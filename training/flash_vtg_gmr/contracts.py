import hashlib
import json
import os
import tempfile
from pathlib import Path


SCHEMA_VERSION = "hiea2m.part1.v1"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
SOURCE_DIRS = {
    "worldcup2022": "WC2022-GMR",
    "sportsmoments": "SportsMoments",
}


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_artifact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    document.setdefault("schema_version", SCHEMA_VERSION)
    document["content_sha256"] = canonical_json_sha256(document)
    _atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def write_jsonl_artifact(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows]
    data = "\n".join(lines) + ("\n" if lines else "")
    _atomic_write_text(path, data)
    return {
        "path": str(path.resolve()),
        "row_count": len(rows),
        "sha256": sha256_file(path),
    }


def _atomic_write_text(path, text):
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def video_stem(vid):
    value = str(vid)
    lower = value.lower()
    for extension in VIDEO_EXTENSIONS:
        if lower.endswith(extension):
            return value[: -len(extension)]
    return value


def normalize_source(source):
    value = str(source or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "worldcup2022": "worldcup2022",
        "wc2022": "worldcup2022",
        "sportsmoments": "sportsmoments",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported dataset source: {source!r}")
    return aliases[value]


def query_key(dataset_setting, split, source, qid):
    return [str(dataset_setting).lower(), str(split).lower(), normalize_source(source), str(qid)]


def video_key(dataset_setting, source, vid):
    return [str(dataset_setting).lower(), normalize_source(source), video_stem(vid)]


def row_key(dataset_setting, split, source, qid, vid):
    return query_key(dataset_setting, split, source, qid) + [video_stem(vid)]


def key_string(key):
    return "|".join(str(part) for part in key)


def stable_exact_deduplicate_windows(windows):
    deduplicated = []
    seen = set()
    for index, window in enumerate(windows or []):
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(f"Invalid temporal window at index {index}: {window!r}")
        start, end = float(window[0]), float(window[1])
        key = (start, end)
        if key not in seen:
            seen.add(key)
            deduplicated.append([start, end])
    return deduplicated


def validate_encoder_mask(mask, store_length=77, model_length=40):
    import numpy as np

    array = np.asarray(mask)
    if array.shape != (store_length,):
        raise ValueError(f"attention_mask must have shape ({store_length},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("attention_mask contains NaN or Inf")
    bool_mask = array.astype(bool)
    valid_length = int(bool_mask.sum())
    if valid_length < 2:
        raise ValueError("attention_mask must contain at least SOT and EOT")
    expected = np.arange(store_length) < valid_length
    if not np.array_equal(bool_mask, expected):
        raise ValueError("attention_mask valid positions must be contiguous from index 0")
    if valid_length > model_length:
        raise ValueError(
            f"Valid token length {valid_length} exceeds model length {model_length}"
        )
    return bool_mask, valid_length


def resolve_media_path(video_root, source, vid):
    root = Path(video_root)
    stem = video_stem(vid)
    source_dir = SOURCE_DIRS[normalize_source(source)]
    candidates = []
    original = Path(str(vid))
    if original.suffix.lower() in VIDEO_EXTENSIONS:
        candidates.append(root / source_dir / original.name)
    candidates.extend(root / source_dir / f"{stem}{ext}" for ext in VIDEO_EXTENSIONS)
    matches = [path for path in candidates if path.is_file()]
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected exactly one media file for source={source!r}, vid={vid!r}; found {unique}"
        )
    return unique[0]


def count_bin(count):
    value = int(count)
    return value if value < 4 else 4
