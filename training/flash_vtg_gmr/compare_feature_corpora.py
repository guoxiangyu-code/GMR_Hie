"""Compare F-Lighthouse with a reference feature corpus for diagnostics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.flash_vtg_gmr.contracts import write_json_artifact


def _load_video(path: Path, dimension: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        array = archive["features"]
    if array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
        raise ValueError(f"Invalid video feature: {path} {array.shape}")
    return array.astype(np.float32, copy=False)


def _load_text(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        hidden = archive["last_hidden_state"].astype(np.float32, copy=False)
        mask = archive["attention_mask"].astype(bool, copy=False)
    if hidden.shape != (77, 512) or mask.shape != (77,):
        raise ValueError(f"Invalid text feature: {path} {hidden.shape}/{mask.shape}")
    return hidden, mask


def _vector_statistics(cosines, relative_l2):
    cosine = np.concatenate(cosines) if cosines else np.empty(0)
    rel_l2 = np.concatenate(relative_l2) if relative_l2 else np.empty(0)
    if len(cosine) == 0:
        return {"row_count": 0}
    return {
        "row_count": int(len(cosine)),
        "cosine_min": float(cosine.min()),
        "cosine_p01": float(np.percentile(cosine, 1)),
        "cosine_mean": float(cosine.mean()),
        "relative_l2_median": float(np.median(rel_l2)),
        "relative_l2_p99": float(np.percentile(rel_l2, 99)),
    }


def _compare_rows(reference, generated, eps=1e-12):
    count = min(len(reference), len(generated))
    reference = reference[:count]
    generated = generated[:count]
    reference_norm = np.linalg.norm(reference, axis=1)
    generated_norm = np.linalg.norm(generated, axis=1)
    cosine = np.sum(reference * generated, axis=1) / (
        reference_norm * generated_norm + eps
    )
    relative_l2 = np.linalg.norm(reference - generated, axis=1) / (
        reference_norm + eps
    )
    return cosine, relative_l2


def compare(args):
    with Path(args.reference_manifest).open("r", encoding="utf-8") as handle:
        reference_manifest = json.load(handle)
    generated_root = Path(args.generated_root)
    streams = {
        "clip": {"dimension": 512, "cosine": [], "relative_l2": [], "t_mismatch": 0},
        "slowfast": {
            "dimension": 2304,
            "cosine": [],
            "relative_l2": [],
            "t_mismatch": 0,
        },
    }
    video_count = 0
    for record in reference_manifest["video_inventory"].values():
        stem = record["vid_stem"]
        for stream, state in streams.items():
            reference = _load_video(Path(record[f"{stream}_path"]), state["dimension"])
            generated = _load_video(
                generated_root / stream / f"{stem}.npz", state["dimension"]
            )
            state["t_mismatch"] += int(len(reference) != len(generated))
            cosine, relative_l2 = _compare_rows(reference, generated)
            state["cosine"].append(cosine)
            state["relative_l2"].append(relative_l2)
        video_count += 1

    text_cosines = []
    text_relative_l2 = []
    text_mask_mismatch = 0
    query_count = 0
    for record in reference_manifest["query_inventory"].values():
        qid = record["qid"]
        reference_hidden, reference_mask = _load_text(Path(record["text_path"]))
        generated_hidden, generated_mask = _load_text(
            generated_root / "clip_text" / f"qid{qid}.npz"
        )
        text_mask_mismatch += int(not np.array_equal(reference_mask, generated_mask))
        valid = reference_mask & generated_mask
        cosine, relative_l2 = _compare_rows(
            reference_hidden[valid], generated_hidden[valid]
        )
        text_cosines.append(cosine)
        text_relative_l2.append(relative_l2)
        query_count += 1

    stream_results = {}
    for stream, state in streams.items():
        stream_results[stream] = {
            "t_mismatch_video_count": state["t_mismatch"],
            **_vector_statistics(state["cosine"], state["relative_l2"]),
        }
    result = {
        "artifact_type": "f-lighthouse-reference-comparison",
        "status": "diagnostic-only",
        "blocks_formal_pipeline": False,
        "reference_manifest": str(Path(args.reference_manifest).resolve()),
        "generated_root": str(generated_root.resolve()),
        "video_count": video_count,
        "query_count": query_count,
        "video_streams": stream_results,
        "text": {
            "mask_mismatch_query_count": text_mask_mismatch,
            **_vector_statistics(text_cosines, text_relative_l2),
        },
    }
    return write_json_artifact(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--output", required=True)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
