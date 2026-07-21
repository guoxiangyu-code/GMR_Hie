import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.flash_vtg_gmr.contracts import (
    SCHEMA_VERSION,
    canonical_json_sha256,
    count_bin,
    key_string,
    normalize_source,
    query_key,
    read_jsonl,
    row_key,
    sha256_file,
    stable_exact_deduplicate_windows,
    validate_encoder_mask,
    video_key,
    video_stem,
    write_json_artifact,
    write_jsonl_artifact,
)


TOKENIZER_ALIASES = {
    "openai-clip-vit-b-32": "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch32": "openai/clip-vit-base-patch32",
}


def _load_tokenizer(name, allow_download=False):
    from transformers import CLIPTokenizer

    model_name = TOKENIZER_ALIASES.get(name, name)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name, local_files_only=not allow_download
    )
    if tokenizer.bos_token_id != 49406 or tokenizer.eos_token_id != 49407:
        raise ValueError(
            f"Unexpected CLIP special IDs: {tokenizer.bos_token_id}/{tokenizer.eos_token_id}"
        )
    return tokenizer, model_name


def _openai_style_ids(tokenizer, query, context_length):
    encoded = tokenizer.encode(query, add_special_tokens=True)
    if len(encoded) > context_length:
        raise ValueError(
            f"Query needs {len(encoded)} tokens, exceeding context length {context_length}: {query!r}"
        )
    if encoded[0] != 49406 or encoded[-1] != 49407 or encoded.count(49407) != 1:
        raise ValueError(f"Invalid SOT/EOT layout for query: {query!r}")
    result = encoded + [0] * (context_length - len(encoded))
    return result, len(encoded)


def _find_phrase_char_span(query, phrase):
    matches = list(re.finditer(re.escape(phrase), query, flags=re.IGNORECASE))
    if len(matches) != 1:
        return None, "missing" if not matches else "ambiguous"
    return matches[0].span(), "unique"


def _find_subsequence(sequence, target):
    return [
        index
        for index in range(len(sequence) - len(target) + 1)
        if sequence[index : index + len(target)] == target
    ]


def _align_phrase(tokenizer, query, phrase, lexical_ids):
    if not phrase:
        return {"label": phrase, "token_indices": [], "status": "missing-label", "method": None}
    char_span, char_status = _find_phrase_char_span(query, phrase)
    if char_status == "unique":
        start, end = char_span
        prefix_count = len(tokenizer.encode(query[:start], add_special_tokens=False))
        end_count = len(tokenizer.encode(query[:end], add_special_tokens=False))
        indices = list(range(1 + prefix_count, 1 + end_count))
        phrase_ids = tokenizer.encode(phrase, add_special_tokens=False)
        selected = [lexical_ids[index - 1] for index in indices]
        if selected == phrase_ids and indices:
            return {
                "label": phrase,
                "token_indices": indices,
                "status": "aligned",
                "method": "prefix-tokenization",
                "char_span": [start, end],
            }

    phrase_ids = tokenizer.encode(phrase, add_special_tokens=False)
    matches = _find_subsequence(lexical_ids, phrase_ids)
    if len(matches) == 1:
        start = matches[0]
        return {
            "label": phrase,
            "token_indices": list(range(1 + start, 1 + start + len(phrase_ids))),
            "status": "aligned",
            "method": "token-subsequence-fallback",
            "char_span": list(char_span) if char_span else None,
        }
    return {
        "label": phrase,
        "token_indices": [],
        "status": "missing" if not matches else "ambiguous",
        "method": "token-subsequence-fallback",
        "char_span": list(char_span) if char_span else None,
    }


def _team_label(row):
    teams = str(row.get("match_info", {}).get("teams", ""))
    candidates = [part.strip() for part in re.split(r"\s+vs\.?\s+", teams, flags=re.IGNORECASE)]
    matches = [team for team in candidates if team and re.search(re.escape(team), row["query"], re.IGNORECASE)]
    return matches[0] if len(matches) == 1 else None


def _template_id(query, action, team):
    value = " ".join(query.lower().split())
    replacements = sorted([item for item in (action, team) if item], key=len, reverse=True)
    for item in replacements:
        placeholder = "{action}" if item == action else "{team}"
        value = re.sub(re.escape(item.lower()), placeholder, value, flags=re.IGNORECASE)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _text_contract(text_path, tokenizer, query, context_length, model_length):
    with np.load(text_path, allow_pickle=False) as archive:
        if "attention_mask" not in archive.files:
            raise ValueError(f"Missing attention_mask in {text_path}")
        npz_mask, npz_valid_length = validate_encoder_mask(
            archive["attention_mask"], context_length, model_length
        )
        stored_input_ids = (
            archive["input_ids"].astype(np.int64, copy=False)
            if "input_ids" in archive.files
            else None
        )
    input_ids, tokenized_valid_length = _openai_style_ids(tokenizer, query, context_length)
    if stored_input_ids is not None:
        if stored_input_ids.shape != (context_length,):
            raise ValueError(
                f"Invalid stored input_ids shape for {text_path}: {stored_input_ids.shape}"
            )
        if not np.array_equal(stored_input_ids, np.asarray(input_ids, dtype=np.int64)):
            raise ValueError(f"Tokenizer/stored input_ids mismatch for {text_path}")
    if tokenized_valid_length != npz_valid_length:
        raise ValueError(
            f"Tokenizer/NPZ valid-length mismatch for {text_path}: "
            f"{tokenized_valid_length} != {npz_valid_length}"
        )
    eot_index = tokenized_valid_length - 1
    if input_ids[eot_index] != 49407 or any(input_ids[eot_index + 1 :]):
        raise ValueError(f"Invalid EOT/padding-fill layout for {text_path}")
    lexical_mask = npz_mask.copy()
    lexical_mask[0] = False
    lexical_mask[eot_index] = False
    lexical_ids = input_ids[1:eot_index]
    lexical_id_zero_count = sum(token == 0 for token in lexical_ids)
    return input_ids, npz_mask.tolist(), lexical_mask.tolist(), lexical_ids, lexical_id_zero_count


def build(args):
    if not args.no_truncate:
        raise ValueError("Canonical manifest generation requires --no_truncate")
    if not args.audit_split_identity:
        raise ValueError(
            "Canonical manifest generation requires --audit_split_identity"
        )
    with open(args.feature_manifest, "r", encoding="utf-8") as handle:
        feature_manifest = json.load(handle)
    if feature_manifest.get("concat_order") != ["slowfast", "clip"]:
        raise ValueError("Feature manifest concat order must be [slowfast, clip]")
    if feature_manifest.get("dataset_setting") != args.dataset_setting:
        raise ValueError("Feature manifest dataset setting mismatch")
    if Path(feature_manifest["text_dir"]).resolve() != Path(args.text_dir).resolve():
        raise ValueError("--text_dir does not match the frozen feature manifest")

    tokenizer, tokenizer_name = _load_tokenizer(args.tokenizer, args.allow_tokenizer_download)
    tokenizer_files = {}
    for key in ("vocab_file", "merges_file"):
        value = tokenizer.init_kwargs.get(key)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"Tokenizer did not expose a local {key}: {value}")
        tokenizer_files[key] = {
            "path": str(Path(value).resolve()),
            "sha256": sha256_file(value),
        }
    tokenizer_snapshot = Path(tokenizer.init_kwargs["vocab_file"]).resolve().parent.name
    video_inventory = feature_manifest["video_inventory"]
    query_inventory = feature_manifest["query_inventory"]
    split_paths = {
        "train": Path(args.train_jsonl),
        "val": Path(args.val_jsonl),
        "test": Path(args.test_jsonl),
    }

    canonical_by_split = {}
    phrase_by_split = {}
    query_keys = set()
    canonical_qid_splits = defaultdict(set)
    row_keys = set()
    video_splits = defaultdict(set)
    vid_query_splits = defaultdict(set)
    lexical_id_zero_count = 0
    phrase_status_counts = defaultdict(int)

    for split, source_path in split_paths.items():
        canonical_rows = []
        phrase_rows = []
        for source_row in read_jsonl(source_path):
            source = normalize_source(source_row.get("dataset_source"))
            qkey_parts = query_key(args.dataset_setting, split, source, source_row["qid"])
            vkey_parts = video_key(args.dataset_setting, source, source_row["vid"])
            rkey_parts = row_key(
                args.dataset_setting, split, source, source_row["qid"], source_row["vid"]
            )
            qkey = key_string(qkey_parts)
            vkey = key_string(vkey_parts)
            rkey = key_string(rkey_parts)
            if qkey in query_keys or rkey in row_keys:
                raise ValueError(f"Duplicate identity key: {qkey} / {rkey}")
            query_keys.add(qkey)
            canonical_qid_splits[(args.dataset_setting, source, str(source_row["qid"]))].add(split)
            row_keys.add(rkey)
            video_splits[vkey].add(split)
            normalized_query = " ".join(source_row["query"].lower().split())
            vid_query_splits[(vkey, normalized_query)].add(split)

            if vkey not in video_inventory or qkey not in query_inventory:
                raise KeyError(f"Feature identity missing for row {rkey}")
            video_record = video_inventory[vkey]
            query_record = query_inventory[qkey]
            raw_windows = source_row.get("relevant_windows", [])
            windows = stable_exact_deduplicate_windows(raw_windows)
            d_label = float(source_row["duration"])
            d_media = float(video_record["D_media"])
            d_grid = float(video_record["D_grid"])
            d_decode = float(video_record["D_decode"])
            for window in windows:
                start, end = window
                if start < 0 or start >= end:
                    raise ValueError(f"Invalid GT window {window} in {rkey}")
                if end > d_label + args.duration_tolerance:
                    raise ValueError(f"GT exceeds D_label in {rkey}: {window} > {d_label}")
                if end > d_decode + args.duration_tolerance:
                    raise ValueError(f"GT exceeds D_decode in {rkey}: {window} > {d_decode}")

            canonical = dict(source_row)
            canonical.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_setting": args.dataset_setting,
                    "split": split,
                    "source": source,
                    "canonical_qid": str(source_row["qid"]),
                    "vid_stem": video_stem(source_row["vid"]),
                    "query_key": qkey_parts,
                    "video_key": vkey_parts,
                    "row_key": rkey_parts,
                    "D_label": d_label,
                    "D_media": d_media,
                    "T": int(video_record["T_final"]),
                    "D_grid": d_grid,
                    "D_decode": d_decode,
                    "relevant_windows_raw": raw_windows,
                    "relevant_windows": windows,
                    "duplicate_gt_removed": len(raw_windows) - len(windows),
                    "count_label": len(windows),
                    "count_bin": count_bin(len(windows)),
                    "exist_label": int(bool(windows)),
                    "feature_identity": {
                        "feature_manifest_sha256": feature_manifest["content_sha256"],
                        "slowfast_sha256": video_record["slowfast_sha256"],
                        "clip_sha256": video_record["clip_sha256"],
                        "text_sha256": query_record["text_sha256"],
                    },
                    "source_row_sha256": canonical_json_sha256(source_row),
                }
            )
            canonical_rows.append(canonical)

            text_path = Path(query_record["text_path"])
            input_ids, encoder_mask, lexical_mask, lexical_ids, zero_count = _text_contract(
                text_path,
                tokenizer,
                source_row["query"],
                args.context_length,
                args.model_length,
            )
            lexical_id_zero_count += zero_count
            action = str(source_row.get("action_type") or "").strip() or None
            team = _team_label(source_row)
            action_alignment = _align_phrase(tokenizer, source_row["query"], action, lexical_ids)
            team_alignment = _align_phrase(tokenizer, source_row["query"], team, lexical_ids)
            phrase_status_counts[f"action:{action_alignment['status']}"] += 1
            phrase_status_counts[f"team:{team_alignment['status']}"] += 1
            phrase_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_setting": args.dataset_setting,
                    "split": split,
                    "source": source,
                    "qid": source_row["qid"],
                    "canonical_qid": str(source_row["qid"]),
                    "vid": source_row["vid"],
                    "vid_stem": video_stem(source_row["vid"]),
                    "query_key": qkey_parts,
                    "query_sha256": hashlib.sha256(source_row["query"].encode("utf-8")).hexdigest(),
                    "input_ids": input_ids,
                    "encoder_valid_mask": encoder_mask,
                    "lexical_mask": lexical_mask,
                    "action": action_alignment,
                    "team": team_alignment,
                    "template_id": _template_id(source_row["query"], action, team),
                    "text_feature_sha256": query_record["text_sha256"],
                }
            )
        canonical_by_split[split] = canonical_rows
        phrase_by_split[split] = phrase_rows

    video_leaks = {key: sorted(value) for key, value in video_splits.items() if len(value) > 1}
    qid_leaks = {
        key_string(key): sorted(value)
        for key, value in canonical_qid_splits.items()
        if len(value) > 1
    }
    vid_query_leaks = {
        f"{key[0]}|{key[1]}": sorted(value)
        for key, value in vid_query_splits.items()
        if len(value) > 1
    }
    if qid_leaks or video_leaks or vid_query_leaks:
        raise ValueError(
            f"Cross-split leakage: qids={len(qid_leaks)}, videos={len(video_leaks)}, "
            f"vid-query={len(vid_query_leaks)}"
        )

    data_output_dir = Path(args.data_output_dir)
    phrase_output_dir = Path(args.phrase_output_dir)
    data_artifacts = {
        split: write_jsonl_artifact(data_output_dir / f"{split}.jsonl", rows)
        for split, rows in canonical_by_split.items()
    }
    phrase_artifacts = {
        "train": write_jsonl_artifact(phrase_output_dir / "train.jsonl", phrase_by_split["train"]),
        "val": write_jsonl_artifact(phrase_output_dir / "val.jsonl", phrase_by_split["val"]),
        "test_diagnostic": write_jsonl_artifact(args.test_phrase_output, phrase_by_split["test"]),
    }
    identity_path = Path(args.index_output).parent / "identity_audit.json"
    write_json_artifact(
        identity_path,
        {
            "artifact_type": "canonical-identity-audit",
            "dataset_setting": args.dataset_setting,
            "query_key_count": len(query_keys),
            "row_key_count": len(row_keys),
            "video_key_count": len(video_splits),
            "cross_split_video_collisions": video_leaks,
            "cross_split_qid_collisions": qid_leaks,
            "cross_split_vid_query_collisions": vid_query_leaks,
            "lexical_token_id_zero_count": lexical_id_zero_count,
            "phrase_alignment_status_counts": dict(sorted(phrase_status_counts.items())),
            "passed": True,
        },
    )
    index = {
        "artifact_type": "canonical-manifest-index",
        "dataset_setting": args.dataset_setting,
        "feature_manifest": {
            "path": str(Path(args.feature_manifest).resolve()),
            "sha256": sha256_file(args.feature_manifest),
            "content_sha256": feature_manifest["content_sha256"],
        },
        "tokenizer": {
            "name": tokenizer_name,
            "class": tokenizer.__class__.__name__,
            "snapshot_revision": tokenizer_snapshot,
            "files": tokenizer_files,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "openai_padding_fill_value": 0,
            "context_length": args.context_length,
            "model_length": args.model_length,
        },
        "data_manifests": data_artifacts,
        "phrase_targets": phrase_artifacts,
        "identity_audit": {
            "path": str(identity_path.resolve()),
            "sha256": sha256_file(identity_path),
        },
    }
    return write_json_artifact(args.index_output, index)


def build_parser():
    parser = argparse.ArgumentParser(description="Build canonical data and isolated phrase targets")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--text_dir", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--dataset_setting", default="standard")
    parser.add_argument("--tokenizer", default="openai-clip-vit-b-32")
    parser.add_argument("--allow_tokenizer_download", action="store_true")
    parser.add_argument("--context_length", type=int, default=77)
    parser.add_argument("--model_length", type=int, default=40)
    parser.add_argument("--no_truncate", action="store_true")
    parser.add_argument("--deduplicate_gt", default="exact", choices=("exact",))
    parser.add_argument("--audit_split_identity", action="store_true")
    parser.add_argument("--duration_tolerance", type=float, default=0.1)
    parser.add_argument("--data_output_dir", required=True)
    parser.add_argument("--phrase_output_dir", required=True)
    parser.add_argument("--test_phrase_output", required=True)
    parser.add_argument("--index_output", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    build(args)


if __name__ == "__main__":
    main()
