"""Extract CLIP, SlowFast, and CLIP-text features with a pinned Lighthouse checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from training.flash_vtg_gmr.contracts import (
    normalize_source,
    read_jsonl,
    resolve_media_path,
    sha256_file,
    video_stem,
    write_json_artifact,
)


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def add_external_checkout(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_feature_array(array: np.ndarray, dimension: int, path: Path) -> None:
    if array.ndim != 2 or array.shape[1] != dimension or array.shape[0] == 0:
        raise ValueError(f"{path}: expected (T,{dimension}), got {array.shape}")
    if array.dtype != np.float32:
        raise ValueError(f"{path}: expected float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: contains NaN/Inf")
    if np.any(np.linalg.norm(array, axis=1) <= 1e-12):
        raise ValueError(f"{path}: contains a zero-norm row")


def existing_video_pair_is_valid(slowfast_path: Path, clip_path: Path) -> bool:
    if not slowfast_path.is_file() or not clip_path.is_file():
        return False
    try:
        with np.load(slowfast_path, allow_pickle=False) as archive:
            slowfast = archive["features"]
        with np.load(clip_path, allow_pickle=False) as archive:
            clip = archive["features"]
        validate_feature_array(slowfast, 2304, slowfast_path)
        validate_feature_array(clip, 512, clip_path)
        return len(slowfast) == len(clip)
    except (KeyError, OSError, ValueError):
        return False


def load_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def video_inventory(rows: Iterable[Dict[str, Any]], video_root: Path) -> List[Dict[str, Any]]:
    inventory: Dict[Tuple[str, str], Dict[str, Any]] = {}
    stem_owners: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        source = normalize_source(row.get("dataset_source"))
        vid = str(row["vid"])
        key = (source, vid)
        stem = video_stem(vid)
        owner = stem_owners.setdefault(stem, key)
        if owner != key:
            raise ValueError(f"Flat feature filename collision for {stem}: {owner} vs {key}")
        inventory.setdefault(
            key,
            {
                "source": source,
                "vid": vid,
                "stem": stem,
                "media_path": str(resolve_media_path(video_root, source, vid)),
            },
        )
    return [inventory[key] for key in sorted(inventory)]


def query_inventory(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inventory: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = str(row["qid"])
        value = {
            "qid": qid,
            "source": normalize_source(row.get("dataset_source")),
            "query": str(row["query"]),
        }
        if qid in inventory and inventory[qid] != value:
            raise ValueError(
                f"Flat text filename collision for qid={qid}: "
                f"{inventory[qid]} vs {value}"
            )
        inventory[qid] = value
    return [inventory[key] for key in sorted(inventory, key=lambda value: (len(value), value))]


def append_journal(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _balanced_source_prefix(
    items: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    """Return a deterministic round-robin prefix covering every source."""
    if limit < 0:
        return items
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_source.setdefault(str(item["source"]), []).append(item)
    selected: List[Dict[str, Any]] = []
    position = 0
    sources = sorted(by_source)
    while len(selected) < limit:
        added = False
        for source in sources:
            source_items = by_source[source]
            if position < len(source_items):
                selected.append(source_items[position])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        position += 1
    return selected


def select_shard(
    items: List[Dict[str, Any]],
    shard_id: int,
    num_shards: int,
    limit: int,
    balanced_sources: bool = False,
) -> List[Dict[str, Any]]:
    selected = [item for index, item in enumerate(items) if index % num_shards == shard_id]
    if balanced_sources:
        return _balanced_source_prefix(selected, limit)
    return selected if limit < 0 else selected[:limit]


def extract_videos(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    add_external_checkout(args.openai_clip_root)
    add_external_checkout(args.lighthouse_root)
    from lighthouse.feature_extractor.vision_encoders.clip_v import CLIPVision
    from lighthouse.feature_extractor.vision_encoders.slowfast import SlowFast
    from lighthouse.frame_loaders.clip_loader import CLIPLoader
    from lighthouse.frame_loaders.slowfast_loader import SlowFastLoader

    device = str(args.device)
    clip_loader = CLIPLoader(args.clip_length, 1.0 / args.clip_length, args.size, device)
    slowfast_loader = SlowFastLoader(args.clip_length, 30, args.size, device)
    clip_encoder = CLIPVision(device, str(args.clip_weight.resolve()))
    slowfast_encoder = SlowFast(device, str(args.slowfast_weight.resolve()))
    # Lighthouse's SlowFast loader does not call eval() after loading the model.
    # Lock inference mode here so BatchNorm cannot make features batch-dependent.
    clip_encoder._clip_extractor.eval()
    slowfast_encoder._slowfast_extractor.eval()
    if clip_encoder._clip_extractor.training or slowfast_encoder._slowfast_extractor.training:
        raise RuntimeError("Lighthouse visual encoders must remain in eval mode")

    items = select_shard(
        video_inventory(rows, args.video_root),
        args.shard_id,
        args.num_shards,
        args.limit,
        balanced_sources=args.balanced_sources,
    )
    journal = args.output_root / "journals" / f"video-shard-{args.shard_id:02d}.jsonl"
    for position, item in enumerate(items, start=1):
        started = time.time()
        slowfast_path = args.output_root / "slowfast" / f"{item['stem']}.npz"
        clip_path = args.output_root / "clip" / f"{item['stem']}.npz"
        if args.resume and existing_video_pair_is_valid(slowfast_path, clip_path):
            append_journal(journal, {**item, "status": "skipped-valid", "position": position})
            continue

        clip_frames = clip_loader(item["media_path"])
        slowfast_frames = slowfast_loader(item["media_path"])
        if clip_frames is None or slowfast_frames is None:
            raise RuntimeError(f"Lighthouse frame loader failed for {item['media_path']}")
        with torch.inference_mode():
            clip_tensor = clip_encoder(clip_frames, bsz=args.clip_batch_size)
            slowfast_tensor = slowfast_encoder(
                slowfast_frames, bsz=args.slowfast_batch_size
            )
        clip = clip_tensor.detach().float().cpu().numpy()
        slowfast = slowfast_tensor.detach().float().cpu().numpy()
        validate_feature_array(clip, 512, clip_path)
        validate_feature_array(slowfast, 2304, slowfast_path)
        t_clip_raw = len(clip)
        t_slowfast_raw = len(slowfast)
        t_final = min(t_clip_raw, t_slowfast_raw)
        # Mirrors Lighthouse VisionEncoder._trim_shorter_length at extraction time.
        clip = np.ascontiguousarray(clip[:t_final])
        slowfast = np.ascontiguousarray(slowfast[:t_final])
        atomic_savez(clip_path, features=clip)
        atomic_savez(slowfast_path, features=slowfast)
        append_journal(
            journal,
            {
                **item,
                "status": "written",
                "position": position,
                "T_clip_raw": t_clip_raw,
                "T_slowfast_raw": t_slowfast_raw,
                "T_final": t_final,
                "trimmed_stream": (
                    None
                    if t_clip_raw == t_slowfast_raw
                    else ("clip" if t_clip_raw > t_slowfast_raw else "slowfast")
                ),
                "clip_sha256": sha256_file(clip_path),
                "slowfast_sha256": sha256_file(slowfast_path),
                "elapsed_sec": round(time.time() - started, 3),
            },
        )
        print(
            f"[{args.shard_id}:{position}/{len(items)}] {item['stem']} "
            f"T={t_clip_raw}/{t_slowfast_raw}->{t_final} "
            f"{time.time() - started:.1f}s",
            flush=True,
        )


def existing_text_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            hidden = archive["last_hidden_state"]
            mask = archive["attention_mask"]
            input_ids = archive["input_ids"]
        shape_and_numeric_valid = (
            hidden.shape == (77, 512)
            and hidden.dtype == np.float32
            and mask.shape == (77,)
            and mask.dtype == np.float32
            and input_ids.shape == (77,)
            and np.issubdtype(input_ids.dtype, np.integer)
            and np.isfinite(hidden).all()
            and np.isin(mask, [0.0, 1.0]).all()
        )
        if not shape_and_numeric_valid:
            return False
        eot_positions = np.flatnonzero(input_ids == 49407)
        if len(eot_positions) != 1:
            return False
        eot_index = int(eot_positions[0])
        expected_mask = (np.arange(77) <= eot_index).astype(np.float32)
        return (
            input_ids[0] == 49406
            and eot_index < 40
            and np.array_equal(mask, expected_mask)
            and not np.any(input_ids[1:eot_index] == 0)
            and not np.any(input_ids[eot_index + 1 :] != 0)
            and np.all(np.linalg.norm(hidden[: eot_index + 1], axis=1) > 1e-12)
        )
    except (KeyError, OSError, ValueError):
        return False


def extract_text(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    add_external_checkout(args.openai_clip_root)
    add_external_checkout(args.lighthouse_root)
    from lighthouse.feature_extractor.text_encoders.clip_t import CLIPText

    encoder = CLIPText(str(args.device), str(args.clip_weight.resolve()))
    encoder._clip_extractor.eval()
    if encoder._clip_extractor.training:
        raise RuntimeError("Lighthouse CLIPText encoder must remain in eval mode")
    items = select_shard(
        query_inventory(rows), args.shard_id, args.num_shards, args.limit
    )
    journal = args.output_root / "journals" / f"text-shard-{args.shard_id:02d}.jsonl"
    for position, item in enumerate(items, start=1):
        started = time.time()
        path = args.output_root / "clip_text" / f"qid{item['qid']}.npz"
        if args.resume and existing_text_is_valid(path):
            append_journal(journal, {**item, "status": "skipped-valid", "position": position})
            continue
        with torch.inference_mode():
            hidden_tensor, lighthouse_mask_tensor = encoder(item["query"])
            input_ids_tensor = encoder._tokenizer(item["query"])
        hidden = hidden_tensor[0].detach().float().cpu().numpy()
        lighthouse_mask = lighthouse_mask_tensor[0].detach().float().cpu().numpy()
        input_ids = input_ids_tensor[0].detach().cpu().numpy().astype(np.int64, copy=False)
        eot_positions = np.flatnonzero(input_ids == 49407)
        if len(eot_positions) != 1:
            raise ValueError(f"qid={item['qid']}: expected one EOT, got {eot_positions}")
        eot_index = int(eot_positions[0])
        mask = (np.arange(77) <= eot_index).astype(np.float32)
        if hidden.shape != (77, 512) or mask.shape != (77,):
            raise ValueError(
                f"qid={item['qid']}: unexpected text shapes {hidden.shape}/{mask.shape}"
            )
        if not np.isfinite(hidden).all() or not np.isin(mask, [0.0, 1.0]).all():
            raise ValueError(f"qid={item['qid']}: invalid text numerics or mask")
        if not np.array_equal(mask, lighthouse_mask):
            raise ValueError(f"qid={item['qid']}: EOT and Lighthouse masks differ")
        if eot_index >= 40 or np.any(input_ids[eot_index + 1 :] != 0):
            raise ValueError(f"qid={item['qid']}: EOT/model-length/padding contract failed")
        atomic_savez(
            path,
            last_hidden_state=hidden,
            attention_mask=mask,
            input_ids=input_ids,
        )
        append_journal(
            journal,
            {
                "qid": item["qid"],
                "status": "written",
                "position": position,
                "valid_length": int(mask.sum()),
                "eot_index": eot_index,
                "sha256": sha256_file(path),
                "elapsed_sec": round(time.time() - started, 3),
            },
        )
        if position % 100 == 0 or position == len(items):
            print(f"[{args.shard_id}:{position}/{len(items)}] text", flush=True)


def write_provenance(args: argparse.Namespace) -> None:
    bpe_path = args.openai_clip_root / "clip" / "bpe_simple_vocab_16e6.txt.gz"
    source_dir = args.output_root.parent / "lighthouse" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    lighthouse_commit = git_commit(args.lighthouse_root)
    openai_clip_commit = git_commit(args.openai_clip_root)
    lighthouse_archive = source_dir / f"lighthouse-{lighthouse_commit}.tar.gz"
    openai_clip_archive = source_dir / f"openai-clip-{openai_clip_commit}.tar.gz"
    for checkout, archive in (
        (args.lighthouse_root, lighthouse_archive),
        (args.openai_clip_root, openai_clip_archive),
    ):
        if not archive.is_file():
            subprocess.check_call(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "archive",
                    "--format=tar.gz",
                    f"--output={archive.resolve()}",
                    "HEAD",
                ]
            )
    payload = {
        "schema_version": "hiea2m.lighthouse-extraction.v1",
        "lighthouse_root": str(args.lighthouse_root.resolve()),
        "lighthouse_commit": lighthouse_commit,
        "lighthouse_source_archive": str(lighthouse_archive.resolve()),
        "lighthouse_source_archive_sha256": sha256_file(lighthouse_archive),
        "openai_clip_root": str(args.openai_clip_root.resolve()),
        "openai_clip_commit": openai_clip_commit,
        "openai_clip_source_archive": str(openai_clip_archive.resolve()),
        "openai_clip_source_archive_sha256": sha256_file(openai_clip_archive),
        "openai_clip_bpe_sha256": sha256_file(bpe_path),
        "clip_weight": str(args.clip_weight.resolve()),
        "clip_weight_sha256": sha256_file(args.clip_weight),
        "slowfast_weight": str(args.slowfast_weight.resolve()),
        "slowfast_weight_sha256": sha256_file(args.slowfast_weight),
        "clip_length": args.clip_length,
        "clip_framerate": 1.0 / args.clip_length,
        "clip_ffmpeg_filter": (
            f"fps={1.0 / args.clip_length},scale=short-side-{args.size},"
            f"center-crop={args.size}x{args.size}"
        ),
        "clip_output_pts_contract": (
            "pts_i=i*clip_length seconds in the fps-filter output time base; "
            "the selected source-frame timestamp may differ by fps rounding"
        ),
        "slowfast_decode_framerate": 30,
        "slowfast_sampling_contract": (
            "ffmpeg fps=30; tile-pad to full seconds; group clip_length seconds; "
            "uniformly sample 32 frames using Lighthouse temporal_sampling"
        ),
        "canonical_temporal_bin_contract": (
            "bin_i=[i*clip_length,min((i+1)*clip_length,D_media))"
        ),
        "spatial_size": args.size,
        "concat_order_for_training": ["slowfast", "clip"],
        "stream_storage": "separate-npz",
        "cross_stream_length_policy": (
            "record raw lengths, then mirror Lighthouse "
            "VisionEncoder._trim_shorter_length at extraction"
        ),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ffmpeg": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        "available_gpus": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "disk_dtype": "float32",
        "encoder_mode": "eval",
        "slowfast_eval_override": (
            "Explicitly call Lighthouse SlowFast._slowfast_extractor.eval(); "
            "upstream model_loader leaves the module in training mode."
        ),
        "text_hidden_layer": "final transformer output after ln_final, before text_projection",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "extraction_provenance.json"
    write_json_artifact(path, payload)
    runtime = {
        "schema_version": "hiea2m.lighthouse-runtime.v1",
        "mode": args.mode,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "device": str(args.device),
        "clip_batch_size": args.clip_batch_size,
        "slowfast_batch_size": args.slowfast_batch_size,
        "resume": bool(args.resume),
        "balanced_sources": bool(args.balanced_sources),
    }
    runtime_path = (
        args.output_root
        / "journals"
        / f"runtime-{args.mode}-shard-{args.shard_id:02d}.json"
    )
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(runtime_path, runtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("video", "text", "provenance"), required=True)
    parser.add_argument("--lighthouse_root", type=Path, required=True)
    parser.add_argument("--openai_clip_root", type=Path, required=True)
    parser.add_argument("--clip_weight", type=Path, required=True)
    parser.add_argument("--slowfast_weight", type=Path, required=True)
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--split_jsonl", nargs="+", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_length", type=float, default=2.0)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--clip_batch_size", type=int, default=120)
    parser.add_argument("--slowfast_batch_size", type=int, default=60)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--balanced_sources",
        action="store_true",
        help="Select a deterministic round-robin limit across dataset sources.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    for path in (
        args.lighthouse_root,
        args.openai_clip_root,
        args.clip_weight,
        args.slowfast_weight,
        args.video_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = load_rows(args.split_jsonl)
    write_provenance(args)
    if args.mode == "video":
        extract_videos(args, rows)
    elif args.mode == "text":
        extract_text(args, rows)


if __name__ == "__main__":
    main()
