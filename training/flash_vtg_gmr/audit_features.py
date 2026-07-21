import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from training.flash_vtg_gmr.contracts import (
    canonical_json_sha256,
    key_string,
    normalize_source,
    query_key,
    read_jsonl,
    resolve_media_path,
    row_key,
    sha256_file,
    validate_encoder_mask,
    video_key,
    video_stem,
    write_json_artifact,
)


LIGHTHOUSE_COMMIT = "d095eaa552cecef240897a8b750306b3b2a08740"
OPENAI_CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_WEIGHT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
SLOWFAST_WEIGHT_SHA256 = "8988deb84b65226669eba1a5da6d14fd170dba374891b21439079c90dd80c026"


def _load_and_validate_provenance(path, clip_length):
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    stored_sha = document.get("content_sha256")
    payload = dict(document)
    payload.pop("content_sha256", None)
    if stored_sha != canonical_json_sha256(payload):
        raise ValueError("Lighthouse extraction provenance content SHA256 mismatch")
    expected = {
        "lighthouse_commit": LIGHTHOUSE_COMMIT,
        "openai_clip_commit": OPENAI_CLIP_COMMIT,
        "clip_weight_sha256": CLIP_WEIGHT_SHA256,
        "slowfast_weight_sha256": SLOWFAST_WEIGHT_SHA256,
        "encoder_mode": "eval",
        "disk_dtype": "float32",
        "concat_order_for_training": ["slowfast", "clip"],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(
                f"Lighthouse extraction provenance {key} mismatch: "
                f"{document.get(key)!r} != {value!r}"
            )
    if float(document.get("clip_length", -1)) != float(clip_length):
        raise ValueError("Lighthouse extraction provenance clip_length mismatch")
    if float(document.get("clip_framerate", -1)) != 1.0 / float(clip_length):
        raise ValueError("Lighthouse extraction provenance clip_framerate mismatch")
    for path_key, sha_key in (
        ("clip_weight", "clip_weight_sha256"),
        ("slowfast_weight", "slowfast_weight_sha256"),
        ("lighthouse_source_archive", "lighthouse_source_archive_sha256"),
        ("openai_clip_source_archive", "openai_clip_source_archive_sha256"),
    ):
        artifact_path = Path(document[path_key])
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if sha256_file(artifact_path) != document[sha_key]:
            raise ValueError(f"Provenance artifact hash mismatch: {artifact_path}")
    return document


def _load_content_hashed_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    stored_sha = document.get("content_sha256")
    payload = dict(document)
    payload.pop("content_sha256", None)
    if stored_sha != canonical_json_sha256(payload):
        raise ValueError(f"Content SHA256 mismatch: {path}")
    return document


def _load_feature(path, expected_dim):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"features"}:
            raise ValueError(f"{path}: expected only key 'features', got {archive.files}")
        features = archive["features"]
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(f"{path}: expected (T,{expected_dim}), got {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"{path}: expected float32, got {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{path}: contains a zero-norm real row")
    return features, norms


def _load_text(path, store_length, model_length, require_input_ids=False):
    with np.load(path, allow_pickle=False) as archive:
        required = {"last_hidden_state", "attention_mask"}
        if require_input_ids:
            required.add("input_ids")
        if not required.issubset(archive.files):
            raise ValueError(f"{path}: missing keys {sorted(required - set(archive.files))}")
        hidden = archive["last_hidden_state"]
        mask = archive["attention_mask"]
        input_ids = archive["input_ids"] if "input_ids" in archive.files else None
    if hidden.shape != (store_length, 512):
        raise ValueError(f"{path}: expected ({store_length},512), got {hidden.shape}")
    if hidden.dtype != np.float32:
        raise ValueError(f"{path}: expected float32 hidden states, got {hidden.dtype}")
    if not np.isfinite(hidden).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    bool_mask, valid_length = validate_encoder_mask(mask, store_length, model_length)
    valid_norms = np.linalg.norm(hidden[bool_mask], axis=1)
    if np.any(valid_norms <= 1e-12):
        raise ValueError(f"{path}: contains a zero-norm valid token row")
    if require_input_ids:
        if input_ids.shape != (store_length,) or not np.issubdtype(
            input_ids.dtype, np.integer
        ):
            raise ValueError(f"{path}: invalid input_ids shape/dtype")
        eot_positions = np.flatnonzero(input_ids == 49407)
        if len(eot_positions) != 1:
            raise ValueError(f"{path}: expected exactly one EOT")
        eot_index = int(eot_positions[0])
        if eot_index + 1 != valid_length or eot_index >= model_length:
            raise ValueError(f"{path}: EOT/mask/model-length mismatch")
        if input_ids[0] != 49406 or np.any(input_ids[eot_index + 1 :] != 0):
            raise ValueError(f"{path}: invalid SOT/EOT/padding layout")
        if np.any(input_ids[1:eot_index] == 0):
            raise ValueError(f"{path}: lexical token ID 0 violates dataset audit")
    return hidden, bool_mask, valid_length, valid_norms


def _media_info(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(result.stdout)
    streams = document.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"ffprobe did not return one video stream for {path}")
    stream = streams[0]
    try:
        duration = float(stream["duration"])
        numerator, denominator = stream["avg_frame_rate"].split("/", 1)
        fps = float(numerator) / float(denominator)
        frame_count = int(stream["nb_frames"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid ffprobe video metadata for {path}: {stream}") from exc
    if duration <= 0 or fps <= 0 or frame_count <= 0:
        raise ValueError(f"Non-positive ffprobe metadata for {path}: {stream}")
    return {"duration": duration, "fps": fps, "frame_count": frame_count}


def _relative_or_absolute(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _audit_batch_invariance(
    reference_root, canary_root, minimum_videos, source_by_stem=None
):
    reference_root = Path(reference_root)
    canary_root = Path(canary_root)
    stream_results = {}
    common_stems = None
    for stream, dimension in (("clip", 512), ("slowfast", 2304)):
        canary_paths = sorted((canary_root / stream).glob("*.npz"))
        if len(canary_paths) < minimum_videos:
            raise ValueError(
                f"Batch-invariance canary has {len(canary_paths)} {stream} videos; "
                f"requires at least {minimum_videos}"
            )
        stems = {path.stem for path in canary_paths}
        common_stems = stems if common_stems is None else common_stems & stems
        row_cosines = []
        relative_l2 = []
        absolute_differences = []
        for canary_path in canary_paths:
            reference_path = reference_root / stream / canary_path.name
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            reference, _ = _load_feature(reference_path, dimension)
            canary, _ = _load_feature(canary_path, dimension)
            if reference.shape != canary.shape:
                raise ValueError(
                    f"Batch-invariance shape mismatch for {stream}/{canary_path.name}: "
                    f"{reference.shape} != {canary.shape}"
                )
            denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(
                canary, axis=1
            )
            row_cosines.extend(
                (np.sum(reference * canary, axis=1) / denominator).tolist()
            )
            relative_l2.extend(
                (
                    np.linalg.norm(reference - canary, axis=1)
                    / np.maximum(np.linalg.norm(reference, axis=1), 1e-12)
                ).tolist()
            )
            absolute_differences.append(np.abs(reference - canary).reshape(-1))
        differences = np.concatenate(absolute_differences)
        stream_results[stream] = {
            "video_count": len(canary_paths),
            "row_count": len(row_cosines),
            "row_cosine_min": float(np.min(row_cosines)),
            "row_cosine_p01": float(np.percentile(row_cosines, 1)),
            "row_cosine_mean": float(np.mean(row_cosines)),
            "relative_l2_median": float(np.median(relative_l2)),
            "relative_l2_p99": float(np.percentile(relative_l2, 99)),
            "absolute_difference_mean": float(differences.mean()),
            "absolute_difference_max": float(differences.max()),
        }
        if stream_results[stream]["row_cosine_min"] < 0.99999:
            raise ValueError(
                f"{stream} batch-invariance row cosine below 0.99999: "
                f"{stream_results[stream]['row_cosine_min']}"
            )
        if stream_results[stream]["relative_l2_median"] > 1e-3:
            raise ValueError(
                f"{stream} batch-invariance median relative L2 above 1e-3: "
                f"{stream_results[stream]['relative_l2_median']}"
            )
    if common_stems is None or len(common_stems) < minimum_videos:
        raise ValueError("Batch-invariance canary streams do not share enough videos")
    source_counts = {}
    if source_by_stem is not None:
        unknown = sorted(stem for stem in common_stems if stem not in source_by_stem)
        if unknown:
            raise ValueError(f"Unknown canary video stems: {unknown[:5]}")
        source_counts = dict(
            sorted(Counter(source_by_stem[stem] for stem in common_stems).items())
        )
        expected_sources = set(source_by_stem.values())
        if set(source_counts) != expected_sources:
            raise ValueError(
                "Batch-invariance canary does not cover every dataset source: "
                f"observed={source_counts}, expected={sorted(expected_sources)}"
            )
    return {
        "artifact_type": "f-lighthouse-batch-invariance-audit",
        "reference_root": _relative_or_absolute(reference_root),
        "canary_root": _relative_or_absolute(canary_root),
        "minimum_video_count": minimum_videos,
        "shared_video_count": len(common_stems),
        "source_counts": source_counts,
        "streams": stream_results,
        "thresholds": {
            "row_cosine_min": 0.99999,
            "relative_l2_median_max": 1e-3,
        },
        "passed": True,
    }


def _balanced_alignment_sample(records, sample_size):
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    for values in by_source.values():
        values.sort(key=lambda item: item["query_key"])
    selected = []
    position = 0
    sources = sorted(by_source)
    while len(selected) < min(sample_size, len(records)):
        added = False
        for source in sources:
            if position < len(by_source[source]):
                selected.append(by_source[source][position])
                added = True
                if len(selected) == min(sample_size, len(records)):
                    break
        if not added:
            break
        position += 1
    return selected


def _alignment_statistics(cosines, relative_l2):
    cosine = np.asarray(cosines, dtype=np.float64)
    rel_l2 = np.asarray(relative_l2, dtype=np.float64)
    return {
        "row_count": int(len(cosine)),
        "row_cosine_mean": float(cosine.mean()),
        "row_cosine_p01": float(np.percentile(cosine, 1)),
        "relative_l2_median": float(np.median(rel_l2)),
        "relative_l2_p99": float(np.percentile(rel_l2, 99)),
    }


def _audit_text_alignment(
    records,
    provenance,
    text_store_length,
    text_model_length,
    device,
    sample_size,
):
    """Independently retokenize every query and replay a fixed text sample twice."""
    if provenance is None:
        raise ValueError("Verified text alignment requires extraction provenance")
    for root_key in ("openai_clip_root", "lighthouse_root"):
        root = str(Path(provenance[root_key]).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    import torch
    from lighthouse.feature_extractor.text_encoders.clip_t import CLIPText

    encoder = CLIPText(str(device), str(Path(provenance["clip_weight"]).resolve()))
    encoder._clip_extractor.eval()
    if encoder._clip_extractor.training:
        raise RuntimeError("Text alignment encoder must remain in eval mode")

    query_results = []
    failures = []
    for record in records:
        input_ids = (
            encoder._tokenizer(
                record["query"],
                context_length=text_store_length,
                truncate=False,
            )[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        with np.load(record["text_path"], allow_pickle=False) as archive:
            stored_ids = archive["input_ids"].astype(np.int64, copy=False)
            stored_mask = archive["attention_mask"].astype(bool, copy=False)
        eot_positions = np.flatnonzero(input_ids == 49407)
        reasons = []
        if not np.array_equal(input_ids, stored_ids):
            reasons.append("input_ids_mismatch")
        if len(eot_positions) != 1:
            reasons.append("non_unique_eot")
            eot_index = -1
        else:
            eot_index = int(eot_positions[0])
            expected_mask = np.arange(text_store_length) <= eot_index
            if not np.array_equal(expected_mask, stored_mask):
                reasons.append("eot_mask_mismatch")
            if eot_index >= text_model_length:
                reasons.append("eot_outside_model_length")
        result = {
            "query_key": record["query_key"],
            "qid": record["qid"],
            "source": record["source"],
            "eot_index": eot_index,
            "passed": not reasons,
            "failures": reasons,
        }
        query_results.append(result)
        if reasons:
            failures.append(result)

    sampled = _balanced_alignment_sample(records, sample_size)
    cosines = []
    relative_l2 = []
    sample_results = []
    with torch.inference_mode():
        for record in sampled:
            with np.load(record["text_path"], allow_pickle=False) as archive:
                stored = archive["last_hidden_state"].astype(np.float32, copy=False)
                mask = archive["attention_mask"].astype(bool, copy=False)
            run_summaries = []
            for _ in range(2):
                replay, replay_mask = encoder(record["query"])
                replay = replay[0].detach().float().cpu().numpy()
                replay_mask = replay_mask[0].detach().cpu().numpy().astype(bool)
                if not np.array_equal(mask, replay_mask):
                    failures.append(
                        {
                            "query_key": record["query_key"],
                            "qid": record["qid"],
                            "failures": ["replay_mask_mismatch"],
                        }
                    )
                    continue
                reference = stored[mask]
                regenerated = replay[mask]
                denominator = np.maximum(
                    np.linalg.norm(reference, axis=1)
                    * np.linalg.norm(regenerated, axis=1),
                    1e-12,
                )
                run_cosines = np.sum(reference * regenerated, axis=1) / denominator
                run_relative_l2 = np.linalg.norm(
                    reference - regenerated, axis=1
                ) / np.maximum(np.linalg.norm(reference, axis=1), 1e-12)
                cosines.extend(run_cosines.tolist())
                relative_l2.extend(run_relative_l2.tolist())
                run_summaries.append(
                    {
                        "row_cosine_min": float(run_cosines.min()),
                        "row_cosine_mean": float(run_cosines.mean()),
                        "relative_l2_median": float(np.median(run_relative_l2)),
                    }
                )
            sample_results.append(
                {
                    "query_key": record["query_key"],
                    "qid": record["qid"],
                    "source": record["source"],
                    "runs": run_summaries,
                }
            )

    statistics = _alignment_statistics(cosines, relative_l2)
    thresholds = {
        "mask_exact_match_fraction": 1.0,
        "valid_row_count_exact_match_fraction": 1.0,
        "row_cosine_mean_min": 0.999,
        "row_cosine_p01_min": 0.995,
        "relative_l2_median_max": 1e-3,
    }
    threshold_failure = (
        statistics["row_cosine_mean"] < thresholds["row_cosine_mean_min"]
        or statistics["row_cosine_p01"] < thresholds["row_cosine_p01_min"]
        or statistics["relative_l2_median"]
        > thresholds["relative_l2_median_max"]
    )
    return {
        "artifact_type": "f-lighthouse-text-alignment-audit",
        "status": "verified" if not failures and not threshold_failure else "failed",
        "tokenizer_query_count": len(records),
        "tokenizer_exact_match_count": sum(item["passed"] for item in query_results),
        "sample_query_count": len(sampled),
        "sample_source_counts": dict(Counter(item["source"] for item in sampled)),
        "device": str(device),
        "openai_clip_commit": provenance["openai_clip_commit"],
        "openai_clip_bpe_sha256": provenance["openai_clip_bpe_sha256"],
        "clip_weight_sha256": provenance["clip_weight_sha256"],
        "thresholds": thresholds,
        "statistics": statistics,
        "query_results": query_results,
        "sample_results": sample_results,
        "failed_qids": sorted({str(item["qid"]) for item in failures}),
        "failure_records": failures,
        "passed": not failures and not threshold_failure,
    }


def audit(args):
    provenance = None
    if args.feature_setting == "f-lighthouse" and not args.extraction_provenance:
        raise ValueError("F-Lighthouse requires --extraction_provenance")
    if args.extraction_provenance:
        provenance = _load_and_validate_provenance(
            args.extraction_provenance, args.clip_length
        )
    runtime_artifacts = {}
    if args.feature_setting == "f-lighthouse":
        if not args.extraction_journal_dir:
            raise ValueError("F-Lighthouse requires --extraction_journal_dir")
        journal_root = Path(args.extraction_journal_dir)
        video_runtime_paths = sorted(journal_root.glob("runtime-video-shard-*.json"))
        text_runtime_paths = sorted(journal_root.glob("runtime-text-shard-*.json"))
        if len(video_runtime_paths) != 2 or len(text_runtime_paths) != 1:
            raise ValueError(
                "F-Lighthouse requires two video runtime sidecars and one text sidecar"
            )
        video_shards = set()
        for path in video_runtime_paths:
            runtime = _load_content_hashed_json(path)
            if (
                runtime.get("mode") != "video"
                or int(runtime.get("num_shards", -1)) != 2
                or int(runtime.get("clip_batch_size", -1)) != 120
                or int(runtime.get("slowfast_batch_size", -1)) != 60
            ):
                raise ValueError(f"Unexpected formal video runtime contract: {path}")
            video_shards.add(int(runtime["shard_id"]))
            runtime_artifacts[f"video_shard_{runtime['shard_id']}"] = {
                "path": _relative_or_absolute(path),
                "sha256": sha256_file(path),
            }
        if video_shards != {0, 1}:
            raise ValueError(f"Unexpected video shard identities: {video_shards}")
        text_runtime = _load_content_hashed_json(text_runtime_paths[0])
        if (
            text_runtime.get("mode") != "text"
            or int(text_runtime.get("num_shards", -1)) != 1
            or int(text_runtime.get("shard_id", -1)) != 0
        ):
            raise ValueError("Unexpected formal text runtime contract")
        runtime_artifacts["text_shard_0"] = {
            "path": _relative_or_absolute(text_runtime_paths[0]),
            "sha256": sha256_file(text_runtime_paths[0]),
        }
    split_paths = [Path(path) for path in args.split_jsonl]
    split_names = [path.stem.lower() for path in split_paths]
    if len(set(split_names)) != len(split_names):
        raise ValueError(f"Split paths have duplicate stems: {split_names}")

    rows_by_split = {}
    all_rows = []
    for split, path in zip(split_names, split_paths):
        rows = read_jsonl(path)
        rows_by_split[split] = rows
        all_rows.extend((split, row) for row in rows)

    query_owners = {}
    canonical_qid_splits = defaultdict(set)
    video_splits = defaultdict(set)
    row_keys = set()
    vid_query_splits = defaultdict(set)
    video_rows = {}
    video_label_durations = defaultdict(set)
    source_counts = Counter()
    for split, row in all_rows:
        source = normalize_source(row.get("dataset_source"))
        source_counts[source] += 1
        qkey = key_string(query_key(args.dataset_setting, split, source, row["qid"]))
        rkey = key_string(row_key(args.dataset_setting, split, source, row["qid"], row["vid"]))
        vkey = key_string(video_key(args.dataset_setting, source, row["vid"]))
        if qkey in query_owners:
            raise ValueError(f"Duplicate query key: {qkey}")
        if rkey in row_keys:
            raise ValueError(f"Duplicate row key: {rkey}")
        query_owners[qkey] = (split, row)
        canonical_qid_splits[(args.dataset_setting, source, str(row["qid"]))].add(split)
        row_keys.add(rkey)
        video_splits[vkey].add(split)
        normalized_query = " ".join(str(row["query"]).lower().split())
        vid_query_splits[(vkey, normalized_query)].add(split)
        video_rows.setdefault(vkey, (source, row["vid"], row))
        video_label_durations[vkey].add(round(float(row["duration"]), 9))

    video_leaks = {key: sorted(value) for key, value in video_splits.items() if len(value) > 1}
    qid_leaks = {
        key_string(key): sorted(value)
        for key, value in canonical_qid_splits.items()
        if len(value) > 1
    }
    query_leaks = {
        f"{key[0]}|{key[1]}": sorted(value)
        for key, value in vid_query_splits.items()
        if len(value) > 1
    }
    if qid_leaks or video_leaks or query_leaks:
        raise ValueError(
            f"Cross-split identity leakage: qids={len(qid_leaks)}, "
            f"videos={len(video_leaks)}, vid-query={len(query_leaks)}"
        )
    inconsistent_durations = {
        key: sorted(values)
        for key, values in video_label_durations.items()
        if len(values) != 1
    }
    if inconsistent_durations:
        raise ValueError(
            "A video has inconsistent annotation durations: "
            f"{list(inconsistent_durations.items())[:5]}"
        )

    slowfast_dir = Path(args.slowfast_dir)
    clip_dir = Path(args.clip_dir)
    text_dir = Path(args.text_dir)
    generation_inventory = {}
    if args.extraction_journal_dir:
        journal_dir = Path(args.extraction_journal_dir)
        journal_paths = sorted(journal_dir.glob("video-shard-*.jsonl"))
        if not journal_paths:
            raise FileNotFoundError(f"No video extraction journals under {journal_dir}")
        for journal_path in journal_paths:
            for record in read_jsonl(journal_path):
                if record.get("status") == "written":
                    generation_inventory[str(record["stem"])] = record
    video_inventory = {}
    numerical = {
        "artifact_type": f"{args.feature_setting}-numerical-audit",
        "video_count": 0,
        "query_count": 0,
        "nonfinite_count": 0,
        "zero_norm_real_row_count": 0,
        "cross_stream_t_mismatch_count": 0,
        "generation_raw_t_mismatch_count": 0,
        "text_mask_noncontiguous_count": 0,
        "padding_hidden_nonzero_row_count": 0,
        "lexical_token_id_zero_count": 0,
        "text_valid_length_min": None,
        "text_valid_length_max": None,
    }

    for vkey, (source, vid, representative) in sorted(video_rows.items()):
        stem = video_stem(vid)
        slowfast_path = slowfast_dir / f"{stem}.npz"
        clip_path = clip_dir / f"{stem}.npz"
        if not slowfast_path.is_file() or not clip_path.is_file():
            raise FileNotFoundError(f"Missing video feature for {vkey}")
        slowfast, slowfast_norms = _load_feature(slowfast_path, 2304)
        clip, clip_norms = _load_feature(clip_path, 512)
        if len(slowfast) != len(clip):
            raise ValueError(
                f"Cross-stream T mismatch for {vkey}: {len(slowfast)} != {len(clip)}"
            )
        if not (0 < len(slowfast) <= args.max_v_l):
            raise ValueError(f"Unexpected T={len(slowfast)} for {vkey}")
        media_path = resolve_media_path(args.video_root, source, vid)
        media_info = _media_info(media_path)
        duration_media = media_info["duration"]
        duration_label = float(representative["duration"])
        duration_grid = len(slowfast) * args.clip_length
        generation = generation_inventory.get(stem)
        if args.extraction_journal_dir and generation is None:
            raise KeyError(f"Missing Lighthouse extraction journal record for {vkey}")
        t_clip_raw = len(clip) if generation is None else int(generation["T_clip_raw"])
        t_slowfast_raw = (
            len(slowfast) if generation is None else int(generation["T_slowfast_raw"])
        )
        t_final = len(clip) if generation is None else int(generation["T_final"])
        if len(clip) != t_final or len(slowfast) != t_final:
            raise ValueError(
                f"Stored/extraction T mismatch for {vkey}: "
                f"stored={len(clip)}/{len(slowfast)}, journal={t_final}"
            )
        if t_clip_raw != t_slowfast_raw:
            numerical["generation_raw_t_mismatch_count"] += 1
        video_inventory[vkey] = {
            "video_key": json.loads(json.dumps(video_key(args.dataset_setting, source, vid))),
            "source": source,
            "vid": vid,
            "vid_stem": stem,
            "slowfast_path": _relative_or_absolute(slowfast_path),
            "clip_path": _relative_or_absolute(clip_path),
            "media_path": _relative_or_absolute(media_path),
            "slowfast_sha256": sha256_file(slowfast_path),
            "clip_sha256": sha256_file(clip_path),
            "media_size_bytes": media_path.stat().st_size,
            "T_slowfast_raw": t_slowfast_raw,
            "T_clip_raw": t_clip_raw,
            "T_final": t_final,
            "generation_trimmed_stream": (
                None if generation is None else generation.get("trimmed_stream")
            ),
            "D_label": duration_label,
            "D_media": duration_media,
            "D_media_source": "ffprobe video_stream.duration",
            "media_avg_frame_rate": media_info["fps"],
            "lighthouse_source_fps_floor": int(np.floor(media_info["fps"])),
            "media_frame_count": media_info["frame_count"],
            "D_grid": duration_grid,
            "D_decode": min(duration_media, duration_grid),
            "duration_delta_label": duration_label - duration_grid,
            "duration_delta_media": duration_media - duration_grid,
            "slowfast_raw_norm_min": float(slowfast_norms.min()),
            "slowfast_raw_norm_max": float(slowfast_norms.max()),
            "slowfast_normalized_norm_min": float(
                np.linalg.norm(
                    slowfast / (slowfast_norms[:, None] + args.normalization_eps),
                    axis=1,
                ).min()
            ),
            "slowfast_normalized_norm_max": float(
                np.linalg.norm(
                    slowfast / (slowfast_norms[:, None] + args.normalization_eps),
                    axis=1,
                ).max()
            ),
            "clip_raw_norm_min": float(clip_norms.min()),
            "clip_raw_norm_max": float(clip_norms.max()),
            "clip_normalized_norm_min": float(
                np.linalg.norm(
                    clip / (clip_norms[:, None] + args.normalization_eps), axis=1
                ).min()
            ),
            "clip_normalized_norm_max": float(
                np.linalg.norm(
                    clip / (clip_norms[:, None] + args.normalization_eps), axis=1
                ).max()
            ),
            "clip_output_pts_first": 0.0,
            "clip_output_pts_last": (t_clip_raw - 1) * args.clip_length,
            "short_video_branch": duration_media < args.clip_length + 0.1,
        }
        numerical["video_count"] += 1

    query_inventory = {}
    valid_lengths = []
    for qkey, (split, row) in sorted(query_owners.items()):
        qid = row["qid"]
        path = text_dir / f"qid{qid}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing text feature for {qkey}: {path}")
        hidden, mask, valid_length, valid_norms = _load_text(
            path,
            args.text_store_length,
            args.text_model_length,
            require_input_ids=args.text_alignment_status == "verified",
        )
        padding_nonzero = int(np.count_nonzero(np.linalg.norm(hidden[~mask], axis=1) > 1e-12))
        query_inventory[qkey] = {
            "query_key": query_key(
                args.dataset_setting,
                split,
                row.get("dataset_source"),
                qid,
            ),
            "qid": qid,
            "source": normalize_source(row.get("dataset_source")),
            "split": split,
            "text_path": _relative_or_absolute(path),
            "text_sha256": sha256_file(path),
            "valid_length": valid_length,
            "valid_norm_min": float(valid_norms.min()),
            "valid_norm_max": float(valid_norms.max()),
            "padding_hidden_nonzero_rows": padding_nonzero,
        }
        numerical["query_count"] += 1
        numerical["padding_hidden_nonzero_row_count"] += padding_nonzero
        valid_lengths.append(valid_length)

    numerical["text_valid_length_min"] = min(valid_lengths)
    numerical["text_valid_length_max"] = max(valid_lengths)
    numerical["passed"] = True

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    numerical_path = output.parent / "numerical_audit.json"
    identity_path = output.parent / "identity_audit.json"
    text_alignment_path = output.parent / "text_alignment_audit.json"
    batch_invariance_path = output.parent / "batch_invariance_audit.json"

    write_json_artifact(numerical_path, numerical)
    write_json_artifact(
        identity_path,
        {
            "artifact_type": f"{args.feature_setting}-identity-audit",
            "dataset_setting": args.dataset_setting,
            "split_row_counts": {key: len(value) for key, value in rows_by_split.items()},
            "source_row_counts": dict(source_counts),
            "query_key_count": len(query_owners),
            "video_key_count": len(video_rows),
            "row_key_count": len(row_keys),
            "cross_split_video_collisions": video_leaks,
            "cross_split_qid_collisions": qid_leaks,
            "cross_split_vid_query_collisions": query_leaks,
            "passed": True,
        },
    )
    text_alignment_verified = args.text_alignment_status == "verified"
    if text_alignment_verified:
        alignment_records = [
            {
                **record,
                "query": query_owners[qkey][1]["query"],
            }
            for qkey, record in sorted(query_inventory.items())
        ]
        text_alignment = _audit_text_alignment(
            alignment_records,
            provenance,
            args.text_store_length,
            args.text_model_length,
            args.text_alignment_device,
            args.text_alignment_sample_size,
        )
    else:
        text_alignment = {
            "artifact_type": f"{args.feature_setting}-text-alignment-audit",
            "status": "unverified",
            "reason": "Locked OpenAI CLIP text re-encoding has not been run.",
            "npz_mask_verified": True,
            "blocks_adapter_aec": False,
            "blocks_token_index_hmsa": True,
            "passed": True,
        }
    write_json_artifact(text_alignment_path, text_alignment)
    if text_alignment_verified and not text_alignment["passed"]:
        raise ValueError(
            f"F-Lighthouse text alignment failed; see {text_alignment_path}"
        )
    batch_invariance = None
    if args.feature_setting == "f-lighthouse":
        if not args.batch_canary_root:
            raise ValueError("F-Lighthouse requires --batch_canary_root")
        batch_invariance = _audit_batch_invariance(
            args.batch_reference_root or output.parent,
            args.batch_canary_root,
            args.batch_canary_min_videos,
            source_by_stem={
                record["vid_stem"]: record["source"]
                for record in video_inventory.values()
            },
        )
        write_json_artifact(batch_invariance_path, batch_invariance)

    manifest = {
        "artifact_type": "feature-manifest",
        "setting": args.feature_setting,
        "dataset_setting": args.dataset_setting,
        "known_provenance": provenance is not None,
        "encoder_mode": None if provenance is None else provenance["encoder_mode"],
        "slowfast_dir": _relative_or_absolute(slowfast_dir),
        "clip_dir": _relative_or_absolute(clip_dir),
        "text_dir": _relative_or_absolute(text_dir),
        "video_root": _relative_or_absolute(args.video_root),
        "split_jsonl": {
            split: {"path": _relative_or_absolute(path), "sha256": sha256_file(path)}
            for split, path in zip(split_names, split_paths)
        },
        "concat_order": ["slowfast", "clip"],
        "stream_dimensions": {"slowfast": 2304, "clip": 512},
        "video_feature_dim": 2816,
        "text_feature_dim": 512,
        "per_stream_normalization": True,
        "normalization_eps": args.normalization_eps,
        "clip_length": args.clip_length,
        "text_context_length_store": args.text_store_length,
        "text_context_length_model": args.text_model_length,
        "text_mask_direction": "1_valid_0_pad",
        "temporal_grid": "half-open, D_grid=T*clip_length",
        "cross_stream_length_policy": (
            "lighthouse-trim-shorter-at-extraction; exact-or-fail-loader"
            if args.extraction_journal_dir
            else "exact-or-fail"
        ),
        "text_token_alignment_status": args.text_alignment_status,
        "video_inventory": video_inventory,
        "query_inventory": query_inventory,
        "inventory_sha256": canonical_json_sha256(
            {"videos": video_inventory, "queries": query_inventory}
        ),
        "extraction_runtime": runtime_artifacts,
        "numerical_audit": {
            "path": _relative_or_absolute(numerical_path),
            "sha256": sha256_file(numerical_path),
        },
        "identity_audit": {
            "path": _relative_or_absolute(identity_path),
            "sha256": sha256_file(identity_path),
        },
        "text_alignment_audit": {
            "path": _relative_or_absolute(text_alignment_path),
            "sha256": sha256_file(text_alignment_path),
        },
    }
    if batch_invariance is not None:
        manifest["batch_invariance_audit"] = {
            "path": _relative_or_absolute(batch_invariance_path),
            "sha256": sha256_file(batch_invariance_path),
            "shared_video_count": batch_invariance["shared_video_count"],
        }
    if args.extraction_provenance:
        provenance_path = Path(args.extraction_provenance)
        manifest["extraction_provenance"] = {
            "path": _relative_or_absolute(provenance_path),
            "sha256": sha256_file(provenance_path),
        }
    return write_json_artifact(output, manifest)


def build_parser():
    parser = argparse.ArgumentParser(description="Audit and freeze a feature corpus")
    parser.add_argument("--mode", default="existing", choices=("existing",))
    parser.add_argument("--dataset_setting", default="standard")
    parser.add_argument("--feature_setting", default="f-old")
    parser.add_argument("--slowfast_dir", required=True)
    parser.add_argument("--clip_dir", required=True)
    parser.add_argument("--text_dir", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--split_jsonl", nargs="+", required=True)
    parser.add_argument("--concat_order", default="slowfast,clip")
    parser.add_argument("--clip_length", type=float, default=2.0)
    parser.add_argument("--max_v_l", type=int, default=75)
    parser.add_argument("--text_store_length", type=int, default=77)
    parser.add_argument("--text_model_length", type=int, default=40)
    parser.add_argument("--normalization_eps", type=float, default=1e-5)
    parser.add_argument("--extraction_provenance")
    parser.add_argument("--extraction_journal_dir")
    parser.add_argument("--batch_reference_root")
    parser.add_argument("--batch_canary_root")
    parser.add_argument("--batch_canary_min_videos", type=int, default=50)
    parser.add_argument(
        "--text_alignment_status",
        choices=("unverified", "verified"),
        default="unverified",
    )
    parser.add_argument("--text_alignment_device", default="cuda:0")
    parser.add_argument("--text_alignment_sample_size", type=int, default=50)
    parser.add_argument("--require_equal_video_lengths", action="store_true")
    parser.add_argument("--fail_on_nonfinite", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.concat_order.split(",") != ["slowfast", "clip"]:
        parser.error("--concat_order must be slowfast,clip")
    audit(args)


if __name__ == "__main__":
    main()
