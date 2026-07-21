# F-Lighthouse Feature Provenance Audit

## 1. Scope

This is a blocking provenance track for Part 1. F-Lighthouse is the only formal feature setting; F-old remains a diagnostic reference. The required provenance artifact is:

```text
artifacts/features/f-lighthouse/extraction_provenance.json
```

F-Lighthouse must never be mixed with F-old within a run. All Standard splits are regenerated and B0/F2 are trained from scratch.

## 2. Dependency Lock

Pin the audited Lighthouse revision:

```text
d095eaa552cecef240897a8b750306b3b2a08740
openai_clip_commit=d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
ViT-B-32.pt sha256=40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
SLOWFAST_8x8_R50.pkl sha256=8988deb84b65226669eba1a5da6d14fd170dba374891b21439079c90dd80c026
```

Lighthouse declares OpenAI CLIP from a Git URL without pinning its revision. Before extraction, record immutable values for:

```text
lighthouse_commit
openai_clip_commit
bpe_simple_vocab_16e6.txt.gz SHA256
ViT-B/32 weight SHA256
SLOWFAST_8x8_R50 weight SHA256
Python/PyTorch/CUDA/cuDNN/ffmpeg versions
device and internal compute dtype
encoder mode
```

If the installed CLIP source revision cannot be proven, the formal extraction gate fails; do not silently use the current GitHub HEAD.

The pinned Lighthouse `slowfast_model_loader()` does not call `eval()`. The formal
wrapper must explicitly call `SlowFast._slowfast_extractor.eval()` and assert that both
visual encoders remain in eval mode. `torch.inference_mode()` alone is insufficient:
BatchNorm would otherwise make features depend on extraction batch composition. A fixed
video must be extracted with SlowFast batch sizes 60 and 16; the minimum row cosine must
be at least `0.99999`. Run this comparison over a fixed 50-video subset and save
`batch_invariance_audit.json`. Training-mode outputs are invalid and must be quarantined.

## 3. Text Contract

Use the raw query string without external lowercasing, stripping, or Unicode normalization. The locked path is equivalent to:

```text
token_ids = clip.tokenize([query], context_length=77, truncate=False)
x = model.token_embedding(token_ids)
x = x + model.positional_embedding
x = model.transformer(x.permute(1,0,2)).permute(1,0,2)
last_hidden_state = model.ln_final(x).float()[0]
eot_index = unique(where(token_ids[0] == 49407))
attention_mask = arange(77) <= eot_index
assert all(token_ids[0, eot_index+1:] == 0)
```

Save all 77 final `ln_final` states, `input_ids`, and the EOT-derived mask. Do not use `encode_text()`, the pooled EOT representation, `text_projection`, or an intermediate layer. `input_ids != 0` is only a dataset consistency statistic, not the normative mask rule. Run with `model.eval()` and `torch.inference_mode()`; save float32 arrays and do not mix CPU-fp32 and CUDA-fp16 outputs within one setting.

For all Standard queries, compare regenerated valid rows against F-old:

```text
mask exact match                  = 100%
valid-token row count exact       = 100%
mean valid-token row cosine       >= 0.999
1st-percentile row cosine         >= 0.995
median valid-token relative L2    <= 1e-3
```

F-Lighthouse saves the token IDs used by the pinned encoder. A tokenizer/ID/mask mismatch blocks every formal experiment; it cannot fall back to F-old.

## 4. Video Contract

Lighthouse produces `[CLIP(512) || SlowFast(2304)]`; split it immediately and save separate streams. The training loader continues to consume `[SlowFast || CLIP]` after per-stream row normalization.

Fixed extraction parameters:

```text
clip_len=2
sampling_fps=0.5
center_crop=true
crop_size=224
CLIP=OpenAI ViT-B/32
SlowFast=SLOWFAST_8x8_R50
float32 on disk
```

Audit the extractor time base rather than inferring it from downstream coordinates:

```text
D_media
T_clip_raw
T_slowfast_raw
ffmpeg command/filter expression
clip output PTS contract
slowfast bin start/end contract
tail-padding policy
short-video branch policy
```

The audited implementation uses ffmpeg `fps=0.5` for CLIP, whose output time-base PTS
is `pts_i=2*i` seconds. This is an output-frame PTS, not a claim that the selected source
frame has exactly the same timestamp; ffmpeg may select the nearest source frame. SlowFast
resamples to 30 fps, groups frames into 2-second clips, tile-pads the final group, and
uniformly samples 32 frames with Lighthouse `temporal_sampling`. Record `T_clip_raw` and
`T_slowfast_raw`, then explicitly mirror Lighthouse `VisionEncoder._trim_shorter_length()`
during generation. Save both streams at `T_final=min(T_clip_raw,T_slowfast_raw)` and
record which stream was trimmed. The training loader requires stored lengths to be exactly
equal and never performs another `min(T)`.

Canonical downstream bin `i` is `[2*i, min(2*(i+1), D_media))`. Every measured CLIP PTS must fall in its bin, but the bin center must not be presented as the observed extraction timestamp.

## 5. Canary

Create a fixed 50-video train canary covering both sources, short/long videos, and encoding variants. Compare pre-normalization streams separately:

```text
row count exact                    = 100%
shape/key/mask exact               = 100%
saved vs regenerated input IDs     = 100%
mean row cosine                    >= 0.999
1st-percentile row cosine          >= 0.995
median relative L2                 <= 1e-3
query mask exact                   = 100%
stored CLIP/SlowFast T equality    = 100%
raw T mismatch records             = 100% auditable
sample PTS inside canonical bin    = 100%
```

Use atomic writes and validate schema/checksum before resuming an existing output. Preserve per-video diagnostics for every mismatch.

The formal gate is binary: the complete inventory, provenance hashes, text IDs/masks,
raw/final length records, and numerical audits either pass or Part 1 remains incomplete.
F-old similarity statistics are diagnostic and do not determine this gate.

## 6. CLI Contract

The extractor accepts pinned checkout paths, immutable weight files, device, shard identity,
and resume mode. Its final audit writes `feature_manifest.json`; partial feature files or
journals never imply a successful feature setting.

The detailed audit may download model weights only after the dependency revision is locked. Authentication tokens must be supplied through process environment or an external credential helper and must never be written to commands, manifests, logs, checkpoints, or repository files.
