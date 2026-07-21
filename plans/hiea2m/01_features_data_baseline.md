# Part 1：特征准备、数据契约与基线锁定

## 1. 任务目标与边界

本任务是 HieA2M 的第一阶段。目标是生成后续所有实验唯一允许使用的数据、特征和 B0 基线产物。

本阶段必须完成：

1. 使用锁定 commit 的 `line/lighthouse` 全量生成 CLIP/SlowFast/CLIP-text corpus；
2. 审计并冻结 F-Lighthouse；现有 F-old 只保留为兼容性参考，不进入主实验；
3. 修复 text mask、GT window 截断和数据顺序问题；
4. 生成 phrase/template supervision manifests，但禁止其进入推理输入；
5. 锁定三个 seeds 的 FlashVTG-GMR B0 checkpoints、预测和评测协议；
6. 输出 Part 2/Part 3 可机器校验的 artifact hashes。

本阶段**不实现** Proposal-to-Event Adapter、AEC、Temporal HMSA 或 V-EPR。除数据契约修复外，不改变 FlashVTG-GMR 模型结构。

### 1.1 当前实现状态（2026-07-20）

已完成并验证：

- F-old 全量 feature/numerical/identity/mask 参考审计，覆盖 1,957 个视频和 5,639 条 query；
- canonical data manifests 与隔离的 phrase targets；action 5,639/5,639、有效 team 4,639/4,639 均由 prefix-tokenization 唯一对齐；
- 固定 40-row text tensor、NPZ 真实 mask、padding row 归零、双视频流 exact-T gate、完整 GT 与 `D_decode` clamp；
- feature/data manifest content hash 和文件 hash gate；
- production/repro-check 模式、局部 epoch RNG、epoch-boundary checkpoint/resume、non-finite fail-fast；
- `B0/B0-mask-only/B0-gt-only/B0-legacy` CLI 语义和 `StepLR.step()` 修复；
- 12 个 hermetic/integration tests，包括 0/1/6/7 mixed batch 的原 FlashVTG forward、criterion 与 finite backward；
- 使用真实 F-old/canonical 数据完成 CPU 严格训练 smoke；两次独立 one-step replay 的模型 hash 一致；不间断 2 epoch 与 epoch-boundary resume 的最终模型 hash一致；
- Lighthouse commit `d095eaa552cecef240897a8b750306b3b2a08740`、OpenAI CLIP commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` 和两份官方权重已锁定；5,639 条正式 text features 已生成，1,957 个视频正在双 GPU 分片生成。

仍未完成，不能伪造为已验收：

- F-Lighthouse 视频特征尚未全量完成；此前启动的 F-old B0 runs 已停止并标记为 `superseded-f-old`，不计入结果；
- 尚未基于 F-Lighthouse 重新生成 canonical manifests、训练 seeds 2024/2025/2026 的正式 B0，也未生成 `baseline_index.json`、正式 val/test predictions 和三 seed mean/std；
- 20-step production-scale replay 尚未执行；当前完成的是 one-step 双运行与 two-epoch resume 集成验证。

## 2. 仓库与已核验事实

工作目录：

```text
/home/guoxiangyu/HieA2G_GMR/GMR_FlashVTGBaseline/generalized-moment-retrieval
```

主要输入：

```text
data/label/Standard/{train,val,test}.jsonl
data/label/Full/{train,val,test}.jsonl
data/Soccer-GMR/feature/standard/{slowfast,clip,clip_text}
data/Soccer-GMR/raw/standard
GMR.pdf
GREC.pdf
HieA2G.pdf
```

`data/Soccer-GMR` 当前是到原始数据目录的符号链接。不要修改或替换链接目标。

已核验的现有特征：

```text
slowfast/*.npz : key=features, shape=(T,2304), dtype=float32
clip/*.npz     : key=features, shape=(T,512),  dtype=float32
clip_text/*.npz:
    last_hidden_state, shape=(77,512), dtype=float32
    attention_mask,    shape=(77,),    dtype=float32
```

进一步审计结果：

```text
Standard query 数                     = 5,639
text valid-token length               = 6..14
valid token after index 39            = 0
non-contiguous text masks             = 0
CLIP/SlowFast raw T mismatch          = 0 / 1,957 videos
video/text NaN or Inf                 = 0
video zero-norm rows                  = 0
text valid-token zero-norm rows       = 0
non-zero hidden rows at padding positions = 373,426
exact duplicate GT windows            = 0 in current Standard/Full
cross-split qid collisions            = 0
cross-split normalized vid collisions = 0
```

最后一项 text 结果很重要：OpenAI CLIP 在 padding positions 上仍可能产生非零 contextual hidden state，因此绝不能用 feature row 是否为零推断 token validity。

Standard 引用 1,957 个视频，视频文件缺失 0、不可读 0、GT 越界 0。两路视频特征的 `T` 对每个视频完全相同。150 秒视频通常对应 75 rows，即一个 row 对应 2 秒；有 9 个视频的 annotation duration 与 `2*T` 不完全相等，因此后文必须区分 annotation duration 和 feature-grid duration。

Standard GT count：

| Split | 0 | 1 | 2 | 3 | 4 | 5 | 6 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2002 | 1423 | 565 | 117 | 23 | 6 | 2 | 6 |
| val | 210 | 165 | 72 | 16 | 2 | 0 | 0 | 4 |
| test | 492 | 384 | 128 | 20 | 10 | 2 | 0 | 5 |

Full 最大 count 为 7。后续 event-mode 数量固定 `M=10`，但本阶段只负责确保 loader 保留完整 GT。

实现边界：HieA2G 论文使用 RoBERTa-base，并通过 noun-phrase 内 word feature 平均构造 phrase feature；GMR/FlashVTG 使用 CLIP+SlowFast 视频特征和 CLIP text encoder。本项目迁移 HMSA 的层次化思想，但不迁移 RoBERTa tokenizer、`[MASK]` token 或其 padding 语义。所有文本契约以已冻结的 OpenAI CLIP `ViT-B/32` pipeline 为准。

## 3. 阶段产物契约

完成后必须存在：

```text
artifacts/features/f-lighthouse/extraction_provenance.json
artifacts/features/f-lighthouse/feature_manifest.json
artifacts/features/f-lighthouse/numerical_audit.json
artifacts/features/f-lighthouse/identity_audit.json
artifacts/features/f-lighthouse/text_alignment_audit.json
artifacts/data/standard/{train,val,test}.jsonl
artifacts/phrase_targets/standard/{train,val}.jsonl
artifacts/diagnostics/phrase_targets/test.jsonl
artifacts/manifests/standard/manifest_index.json
artifacts/baselines/{2024,2025,2026}/
    model_best.ckpt
    opt.json
    command.txt
    environment.txt
    predictions_raw.jsonl
    predictions_legacy_nms.jsonl
    metrics.json
    reproducibility.json
    artifact_manifest.json
artifacts/baselines/baseline_index.json
```

每个 JSON manifest 必须包含 schema version 和 SHA256。Part 2/3 根据 hash fail fast，不允许靠路径名猜测 artifact 身份。

## 4. Feature Gate

### 4.1 F-Lighthouse：冻结主实验特征

仓库 dataset 按 `--v_feat_dirs` 的传入顺序执行：

1. 分别读取每一路 feature；
2. 每一路分别做 row-wise L2 normalization；
3. 要求已存储两路 `T` 完全相等，loader 不再裁剪；
4. 按参数顺序 concat。

现有训练脚本传入 `slowfast clip`，所以 B0 和全部主实验的实际输入契约是：

```text
slowfast_norm = L2Norm(slowfast_feat, dim=-1)
clip_norm     = L2Norm(clip_feat, dim=-1)
video_content = [slowfast_norm(2304) || clip_norm(512)]  # T x 2816
```

不得改成 `[CLIP || SlowFast]`，也不得把 2816-D combined feature 当成单路后统一 normalize。两种变化都会破坏 checkpoint 兼容性。

`feature_manifest.json` 至少记录：

```text
schema_version
setting = "f-lighthouse"
slowfast_dir, clip_dir, text_dir
split_jsonl paths + sha256
file inventory count + inventory sha256
per-file keys, shapes, dtypes
concat_order = ["slowfast", "clip"]
per_stream_normalization = true
normalization_eps = 1e-5
clip_length = 2.0
text_context_length_store = 77
text_context_length_model = 40
text_mask_direction = "1_valid_0_pad"
temporal_grid = "half-open, D_grid=T*clip_length"
cross_stream_length_policy = "lighthouse-trim-shorter-at-extraction; exact-or-fail-loader"
numerical_audit_sha256
identity_audit_sha256
text_alignment_audit_sha256
text_token_alignment_status = "verified" or "unverified"
known_provenance = true
lighthouse_commit, openai_clip_commit
clip_weight_sha256, slowfast_weight_sha256
```

F-Lighthouse 是 B0、G0、P0、H、C、F 主实验的唯一 feature setting。F-old 只用于数值差异诊断，禁止用 F-old checkpoint 初始化任一正式实验。

### 4.2 固定长度 CLIP 文本与真实 Mask

#### 4.2.1 两个长度不能混淆

固定：

```text
L_store = 77    # OpenAI CLIP context_length，NPZ 物理长度
L_model = 40    # 当前 FlashVTG-GMR/B0 的 --max_q_l
```

当前所有有效序列长度只有 `6..14`，所以 `L_model=40` 不截断任何当前有效 token。仍然必须把“张量物理长度”和“语义有效长度”分开：

```text
query_tokens_40 = last_hidden_state[:40]       # 40 x 512，始终固定长度
encoder_valid_40 = attention_mask[:40]         # 40，来自 NPZ
```

loader 不按 `sum(mask)` 裁成变长 tensor，collate 也不得再次根据 tensor 长度生成全 1 mask。它直接 stack 两个固定长度 tensor：

```text
src_txt      : B x 40 x 512
src_txt_mask : B x 40       # bool/float，1/True 表示 valid
```

这种隔离保证 phrase token indices 在不同 batch 中不发生位移，同时阻止固定的 padding hidden rows参与模型。

#### 4.2.2 Token IDs 与三种 mask

OpenAI CLIP 只注册 SOT/EOT 两个特殊 token；固定长度输出张量的未写入位置以数值 0 填充，但不能把词表 ID 0 宣称为专用 PAD token：

```text
padding_fill_value = 0
SOT = 49406
EOT = 49407
```

phrase builder 对原始 query 重新执行同版本 tokenizer，并生成：

```text
input_ids_77       : token ids，shape=(77,)
F-Lighthouse encoder_valid_mask : npz_attention_mask.astype(bool)
EOT-derived audit mask           : arange(77) <= unique_eot_index
lexical_mask       : encoder_valid_mask & positions not in {SOT_position,EOT_position}
padding_mask       : ~encoder_valid_mask
```

约束：

1. `input_ids[0]==SOT`；
2. 恰好有一个 EOT，且 EOT 是最后一个 valid token；
3. SOT 到 EOT 的 valid positions 连续；
4. EOT 后的 token IDs 全部等于 `padding_fill_value`；
5. 模型主 mask 来源始终是 Lighthouse `CLIPText` 保存的 NPZ mask；重新 tokenize/EOT mask 只做一致性审计；
6. EOT-derived mask 必须逐元素等于 NPZ mask，并审计它在当前数据上是否等于 `input_ids!=0`；
7. 全部 5,639 条 query 的 lexical positions 中不得出现 token ID 0；若出现则 `input_ids!=0` 一致性检查预期失败，但 EOT-based mask 仍是规范结果；
8. `encoder_valid_mask` 必须逐元素等于 NPZ `attention_mask.astype(bool)`；
9. B0/FlashVTG 可让 SOT、content、EOT 参与 query encoding/pooling；
10. phrase span、action/team token loss 只能使用 `lexical_mask`，不得包含 SOT/EOT 或 EOT 后的 padding positions。

Mask 方向固定为：

```text
dataset/collate/model API : 1 or True = valid
PyTorch key_padding_mask  : True = padding
conversion                : key_padding_mask = ~src_txt_mask.bool()
```

任何模块若直接把 `src_txt_mask` 传给 `key_padding_mask` 都应在测试中失败。视频 mask 使用相同外部方向：`video_mask==1` 表示真实 clip row。

#### 4.2.3 Padding 隔离与截断失败策略

归一化顺序固定：

```text
valid_rows = rowwise_L2(last_hidden_state[:40], eps=1e-5)
query_tokens_40 = where(encoder_valid_40[:,None], valid_rows, 0)
```

Padding rows 即使原始 hidden state 非零，也在 loader 边界显式置零，并继续由 mask 双重隔离。必须有 padding-value perturbation test：随机替换 NPZ 的 padding rows 后，`src_txt_mask`、query pooling 和 eval outputs 均不变。

`clip.tokenize(..., context_length=77, truncate=False)` 超长时报错。不得静默截到 77，也不得在 40 截断后把丢失的 EOT 伪造为 valid。审计要求当前所有 EOT index `<40`；未来数据违反时必须重新注册新的 `L_model` feature setting，而不是局部放宽。

F-Lighthouse NPZ 由同一锁定的 OpenAI CLIP tokenizer/model 直接生成。仍需对 Standard 全部 5,639 条 query 重新 tokenize，验证 input IDs、EOT 位置和保存 mask；抽样双运行比较 valid token rows：

```text
mask exact match                      = 100%
valid-token row count exact match     = 100%
mean valid-token row cosine           >= 0.999
1st-percentile valid-token cosine     >= 0.995
median valid-token relative L2        <= 1e-3
```

通过后设置 `text_token_alignment_status=verified`，phrase indices 才能用于 H1/H2/F/R。未通过时所有正式任务停止，不允许回退到 F-old。`text_alignment_audit.json` 保存 locked tokenizer/model hashes、逐 query 结果和失败 qids。

### 4.3 Lighthouse 生成与 Provenance Gate

Lighthouse 全量生成是进入 Part 2 的阻塞条件。详细的依赖锁、文本 hidden-layer、视频抽帧 PTS、SlowFast 尾帧 padding、canary 阈值和跨设备数值审计见：

```text
docs/hiea2m/feature_provenance_audit.md
```

主文档只固定以下不可变事实：

```text
lighthouse_commit = d095eaa552cecef240897a8b750306b3b2a08740
lighthouse_output = [CLIP(512) || SlowFast(2304)]
training_output   = [SlowFast(2304) || CLIP(512)]
text_hidden       = final Transformer layer after ln_final, all 77 rows
formal text mask  = Lighthouse CLIPText saved mask, audited against unique EOT
encoder mode      = eval; explicitly override Lighthouse SlowFast loader's train default
generation length = record T_clip_raw/T_slowfast_raw, then use Lighthouse min(T)
loader length gate= stored T_clip == stored T_slowfast == T_final
```

生成时必须逐视频保存 raw 两路长度。官方 `VisionEncoder._trim_shorter_length()` 的语义在生成阶段显式执行，journal 记录被裁 stream 和 `T_final`；训练 loader 只接受已对齐存储，不得再次 `min(T)`。全部 train/val/test 生成、审计和 hash 冻结后，必须从头重跑 B0 与后续所有实验，禁止与 F-old 混用。

### 4.4 数值与跨模态长度审计

`audit_features.py` 在 normalization 前逐文件、逐 row 检查：

```text
isfinite(features).all()
slowfast.ndim==2 and slowfast.shape[1]==2304
clip.ndim==2 and clip.shape[1]==512
text.shape==(77,512)
attention_mask.shape==(77,)
raw row L2 norm > 1e-12 for every video row and valid text row
T_slowfast == T_clip
0 < T <= 75 for Standard
```

文本 padding hidden rows只要求 finite，不要求零范数。归一化公式固定复用当前 helper：

```text
L2Norm(x) = x / (norm(x,dim=-1,keepdim=True) + 1e-5)
```

审计同时检查 normalization 后、concat 后和 batch 构造后的 tensor 全部 finite。任何 NaN/Inf/零范数真实 row 都是 hard failure，不允许用 `nan_to_num`、epsilon replacement、丢弃 row 或跳过文件继续训练。

每个视频在 manifest 记录：

```text
T_slowfast_raw, T_clip_raw, T_final
D_label, D_media, D_grid, duration_delta_label, duration_delta_media
raw norm min/max per stream
normalized norm min/max per stream
NaN/Inf/zero-norm counts
per-stream file SHA256
```

F-Lighthouse 允许生成阶段出现 `T_slowfast_raw != T_clip_raw`，但必须记录并严格执行 `T_final=min(T_slowfast_raw,T_clip_raw)`，与官方 Lighthouse 一致。落盘后的两路 shape 必须都为 `T_final`；正式 loader 遇到存储两路不等必须 fail-fast，不能再次裁剪。text 的 `L_store/L_model` 独立于视频 `T`，绝不能为 concat 或 batching 把 query rows 裁到 video `T`。

训练循环把当前“发现 NaN 只打印”改成：

```text
assert isfinite(each weighted loss)
assert isfinite(total loss)
clip_grad_norm_(..., error_if_nonfinite=True)
```

首次 non-finite 时保存 qid/vid、resolved config 和 batch artifact hash 后立即停止；不得继续 `optimizer.step()`。

## 5. 数据与监督契约

### 5.1 Dataset loader 修复

修改 `training/flash_vtg_gmr/dataset.py`：

1. `max_windows=-1` 表示保留全部 `relevant_windows`；
2. 删除对 GT windows 的 in-place `random.shuffle`；
3. 使用 manifest 中稳定去重后的 GT，保持首次出现顺序；
4. null query 保持在 dataset 中，`span_labels` 为合法空 tensor；
5. 同时读取 `last_hidden_state[:40]` 和真实 `attention_mask[:40]`；
6. query tensor 保持固定 40 rows，collate 直接 stack feature 和真实 mask；
7. text padding feature rows 显式置零，但不用 feature 值推断 mask；
8. 对任意缺失、额外、非有限或 shape 不符的 key fail-fast，不用宽泛 `except` 回退另一格式；
9. 两路已存储视频 `T` 必须相等并等于 manifest 的 `T_final`；禁止训练时静默 `min(T)`；
10. 训练、validation、test 使用相同读取与坐标规则。

数据修复是 B0 和所有新模型的公共输入修复，不能只对新模型启用。

### 5.2 Canonical Data 与 Phrase Targets

新增：

```text
training/flash_vtg_gmr/build_manifests.py
```

Canonical data row 输出：

```text
dataset_setting, split, source
qid, vid, vid_stem, query
D_label, D_media, T, D_grid, D_decode
relevant_windows_raw
relevant_windows               # 稳定 exact-dedup 后的 GT
duplicate_gt_removed
count_label, exist_label
feature_identity               # query/video keys + feature hashes
source_row_sha256
```

Phrase target row 只输出：

```text
dataset_setting, split, source
qid, vid, vid_stem
query_sha256
input_ids                      # OpenAI CLIP, length 77
encoder_valid_mask             # length 77
lexical_mask                   # length 77
action_labels
team_labels
action_char_spans
team_char_spans
action_token_indices
team_token_indices
action_alignment_status        # resolved / ambiguous / unavailable
team_alignment_status          # resolved / ambiguous / unavailable
template_id
```

规则：

- 使用 F-Lighthouse 锁定的 OpenAI CLIP tokenizer；
- `dataset_source=sportsmoments/worldcup2022` 分别规范化为 `SportsMoments/WC2022`，未知值失败；
- token indices 必须通过 tokenizer/decode round-trip，且不包含 SOT/EOT 或 EOT 后的 padding positions；
- 保存的 attention mask 必须与 NPZ 完全一致；
- SportsMoments 只有 action supervision，team 字段为空；
- WC2022 使用 action/team supervision；
- 多标签全部保留，不强行选择单一 action/team；
- 无法唯一对齐的 span 标为 `ambiguous`，不能猜测 token indices；
- `template_id` 通过将 action/team span 替换为 `<ACTION>/<TEAM>` 后执行 lowercase、标点和空格归一化生成。

OpenAI CLIP tokenizer 不直接返回 offset mapping，禁止用 `query.split()` 猜 token indices。主实现优先调用同一个官方 tokenizer，而不是重写完整 BPE：

1. encoder 仍接收完全未修改的 raw query；
2. 用官方 cleaner 得到 alignment copy，并在 cleaned query 中找到唯一 phrase char interval；
3. 分别 tokenize `cleaned_query[:phrase_start]` 与 `cleaned_query[:phrase_end]`，扣除 SOT/EOT 后由 content-token 数量差得到候选 slice；
4. 候选 content indices 加 `+1` 映射到完整 `input_ids_77`；
5. decode 候选 slice、执行同一 clean，必须精确等于 phrase cleaned form；
6. phrase token slice 必须全部满足 `lexical_mask==1`；
7. 同一 phrase 多次出现、cleaning 映射非唯一或 prefix tokenization 在 BPE 边界不稳定时，进入 fallback；
8. fallback 才使用官方 `SimpleTokenizer` 的 regex/byte encoder/BPE 输出建立 cleaned-span 映射，不复制一套修改版 tokenizer；仍无法唯一定位则标为 `ambiguous`，不默认取 first occurrence。

GT 去重只允许一种定义：

```text
canonical_window = (float64(start), float64(end))
is_exact_duplicate(a,b) = (a.start==b.start and a.end==b.end)
```

按 source list 顺序保留首次出现项。`1` 与 `1.0` 视为相同数值；不做 rounding、tIoU merge、NMS 或相邻窗口合并。两个不同 GT 即使 tIoU 很高也必须保留。当前 Standard/Full 审计为 0 个 exact duplicate，但仍固定该规则以防 future/rebuilt manifests 改变 count target。

manifest 生成后必须满足：

```text
count_label == len(relevant_windows)
exist_label == int(count_label > 0)
relevant_windows 中无 exact duplicate
sha256(relevant_windows_raw) 可追溯到 source row
```

canonical `data/` manifest 中 `relevant_windows` 是唯一训练真值。原始 `moment` 字段可保存在 `source_metadata` 供审计，但 loader/criterion 不得再次解析它覆盖 canonical windows。

Phrase target manifest 只能进入训练/带标签 validation 的 criterion targets 或离线诊断。禁止进入 `model.forward()`、inference dataloader 和 postprocessing。

builder 同时输出两套文件：

```text
artifacts/data/standard/{split}.jsonl
  # canonical core row：qid/vid/query/duration/source/dedup windows/count/exist

artifacts/phrase_targets/standard/{train,val}.jsonl
  # criterion-only phrase fields，以 (dataset_setting,split,canonical_qid) join

artifacts/diagnostics/phrase_targets/test.jsonl
  # test 标签诊断；普通 inference 不得打开
```

B0、Part 2 和所有 inference dataloader 只允许打开 `artifacts/data/`；Part 3 train/val criterion 才能额外打开 `artifacts/phrase_targets/`。不要把 phrase fields 嵌回 model-facing row 后再依靠“模型暂时不用”来防泄漏。

`manifest_index.json` 必须为两套文件按 split 记录 `{path, sha256, row_count}`，并记录：

```text
dataset_setting = "standard"
source_jsonl.{train,val,test}.{path,sha256}
data_manifests.{train,val,test}.{path,sha256,row_count}
phrase_manifests.{train,val,test}.{path,sha256,row_count}
tokenizer/model-length contract hash
gt_dedup_policy = "exact-float64-first-occurrence"
temporal_coordinate_contract hash
identity_audit path + sha256
```

后续训练从索引解析实际文件，不能自行构造另一套文件名。

### 5.3 时序坐标映射契约

所有 GT 输入采用秒和半开区间：

```text
window_sec = [start_sec, end_sec)
0 <= start_sec < end_sec
clip_length = c = 2.0 seconds
T = T_clip = T_slowfast
D_grid = T * c
D_media = ffprobe duration
D_label = annotation duration
D_decode = min(D_media, D_grid)
time_tolerance = 0.1 seconds
```

Feature validity 完全由已存储 rows 决定：

```text
video_mask[i] = 1 for every i in [0,T)
```

`D_label` 不生成 video mask。GT 必须同时满足 `end<=D_label+time_tolerance` 和 `end<=D_decode+time_tolerance`；失败时停止，不静默 clamp GT。duration 不一致必须逐视频记录 `D_label/D_media/D_grid` 及 delta。模型归一化始终使用 feature grid：

```text
span_xx_norm = [start_sec / D_grid, end_sec / D_grid]
span_cxw_norm = [(start+end)/(2*D_grid), (end-start)/D_grid]

decoded_xx_sec = clamp(
    [(cx-w/2)*D_grid, (cx+w/2)*D_grid],
    0, D_decode
)
```

TEF 与 clip timestamps 使用同一 grid：

```text
TEF[i] = [i/T, (i+1)/T]
clip_interval_grid[i] = [i*c, (i+1)*c)
clip_center_grid[i] = (i+0.5)*c
```

Token-Clip MIL 等新监督以 center rule 判定内部 clip：

```text
inside(i,[s,e)) iff s <= clip_center_grid[i] < e
```

若需要离散 span indices，唯一规则为：

```text
start_idx = floor(start_sec/c)
end_idx_exclusive = ceil(end_sec/c)
indices = [start_idx, ..., end_idx_exclusive-1] intersect [0,T)
```

不能在不同 loss 中混用 label-duration normalization、`int(end/c)-1` 和 center membership。评估始终在 seconds 上进行；submission 保存 `[start_sec,end_sec,score]` 并 clamp 到 `D_decode`。

### 5.4 QID/VID/Split 隔离

生成 `manifest_index.json` 前运行 identity audit：

```text
dataset_setting = "standard" or "full"
source = normalized dataset_source               # SportsMoments / WC2022
canonical_qid = str(qid)                         # 原始 typed value 另存
vid_stem = strip_one_known_media_extension(vid) # .mp4/.mkv/.webm/.avi/.mov/.m4v
normalized_query = casefold(collapse_ws(NFC(query).strip()))

query_key = (dataset_setting, split, source, canonical_qid)
video_key = (dataset_setting, source, vid_stem)
row_key = (dataset_setting, split, source, canonical_qid, vid_stem)
```

`normalized_query` 只用于 identity audit，不替换送入 CLIP 的 raw query。

缓存、日志、fixture 和 artifact record 必须保存完整 key 或其无碰撞 hash；不得用裸 qid/vid 作为跨 setting 的唯一键。F-old 现有 `qid{qid}.npz` 路径必须通过 feature manifest 的 query-key 映射解析，并额外验证没有两个 query keys 指向内容冲突的同名文件。

必须满足：

1. `query_key/row_key` 全局唯一；
2. 同一 `(dataset_setting,source,canonical_qid)` 不能属于多个 split；
3. 同一 `video_key` 不能属于多个 split；
4. `(dataset_setting,source,vid_stem,normalized_query)` 不跨 split 重复；
5. 每个 row 的显式 `split`（若 source 中存在）与文件名一致；
6. 每个 query_key 只映射到一个 query 和一个 text NPZ；
7. 每个 video_key 在两路视频目录各映射到且只映射到一个 feature file；
8. feature lookup 先以 `source+vid_stem` 查询 feature manifest 映射，再得到物理路径；禁止仅拼接裸 `vid_stem` 猜路径；
9. B0/Part 2 train loader 只打开 `data.train`，validation 只打开 `data.val`；Part 3 criterion 只能 join 同 split 的 phrase file；test GT 只能由离线 evaluator 打开。

重复 query/template 本身是任务设计的一部分，不作为 split leakage 删除；模板依赖在 Part 3 单独诊断。论文声明的是 video-clip-level split，因此不额外要求整场 match ID 互斥，但必须在报告中明确这一粒度。

### 5.5 Batch contract

Part 1 结束时固定以下 batch 输入：

```text
slowfast_feat : B x T x 2304
clip_feat     : B x T x 512
video_content : B x T x 2816  # [normalized SlowFast || normalized CLIP]
video_mask    : B x T
query_tokens  : B x 40 x 512
query_mask    : B x 40        # 1=True=valid，包括 SOT/EOT
targets:
    span_labels: list[Ni x 2], Ni may be 0..7
    exist_label: B
optional criterion-only join in Part 3:
    lexical_mask: B x 40      # 排除 SOT/EOT 和 EOT 后的 padding positions
    phrase_metadata
```

`model.forward()` 只收到显式 model inputs 和经过白名单过滤的普通 targets；不传原始 JSON row、phrase row 或 `batch_meta` 全字典。TEF 仍由 baseline model 在线添加，不写入 `video_content`，也不写入 feature files。

## 6. B0 基线锁定

### 6.1 基线与最小归因记录

先保存一次当前脚本/现有 checkpoint 的 **legacy reference**，不修改其输出。正式父基线固定为 `B0 = B0-fixed`。为区分基础修复的影响，仅用 seed 2024 运行两个诊断变体：

```text
B0-legacy:    当前 checkpoint + 当前脚本，仅作历史参考
B0-mask-only: 真实 text mask/PAD 隔离 + legacy GT 路径，仅作诊断
B0-gt-only:   legacy text mask + 完整 canonical GT，仅作诊断
B0/B0-fixed:  真实 text mask/PAD 隔离 + 完整 canonical GT + F-Lighthouse contract
```

`B0-mask-only` 和 `B0-gt-only` 不进入论文主表，也不要求三个 seeds。`B0-mask-only` 保留 legacy `max_windows=5` 和随机选窗语义，但在 GT 副本上使用 `(seed,split,query_key)` 派生的局部 RNG，不能再次原地 shuffle 或扰动模型初始化 RNG。时序坐标与 fail-fast 检查属于共同数据契约；若它们在当前 corpus 上改变样本，必须单独记录受影响 qid，不能含混归入 mask 或 GT 收益。Part 2/3 只允许从正式 B0 初始化。

### 6.2 B0 模型约束

B0 保持：

- FlashVTG-GMR 模型结构；
- legacy existence head；
- legacy proposal/NMS inference；
- `K=50` proposal 输出；
- 原定位、分类和 saliency loss 结构。

B0 只接受本阶段的数据修复，不包含 Adapter、AEC、HMSA、V-EPR 或新的 count head。

训练基础设施同时修复现有 `StepLR.step(losses)` 调用：当前 scheduler 是 `StepLR`，必须每 epoch 调用无参 `step()`；把 loss 作为位置参数会被解释成 epoch。该差异计入 `B0-legacy -> B0` 的基础修复记录，后续所有变体共享，不作为 HMSA/AEC 收益。

Part 1 不修改 `models/flash_vtg_gmr/model.py` 来导出 Part 2 候选特征。它只保存 B0 checkpoint、raw proposals 和 legacy predictions。`candidate_feat/candidate_mask/candidate_span/candidate_logit/candidate_point/candidate_scale` 是 Part 2 的第一个实现任务；该接口关闭时必须保持 B0 前向和 state dict 不变。

训练 seeds 固定：

```text
2024, 2025, 2026
```

所有 checkpoint/early stopping 只依据 validation。test 只在每个 seed 的 checkpoint 锁定后运行一次。

### 6.3 随机性、两种运行模式与 Resume

两种模式不可混用：

```text
Repro-check:
    --repro_check
    num_workers=0
    cudnn.benchmark=false
    cudnn.deterministic=true
    deterministic_algorithms=true
    固定 reference device/environment
    用于 20-step、resume 和 fixture 测试

Production training:
    --seed {2024,2025,2026}
    num_workers 可配置且写入 opt
    cudnn.benchmark=false
    记录环境、随机源和 worker seeds
    不要求跨设备逐元素相同
```

两种模式都在构造 dataset/model/DataLoader 前设置 Python、NumPy、Torch CPU/CUDA seeds。DataLoader 使用显式 generator；worker 通过 `worker_init_fn` 从 `(seed,worker_id)` 派生 seed。validation/test 不 shuffle，正式 B0 的 `txt_drop_ratio=0`。

saliency 等样本级随机选择使用局部 RNG：

```text
local_seed = Hash(seed, split, epoch, query_key)
dataset.set_epoch(epoch)
```

这样同一 seed 可重放，但不同 epoch 不会永久选择同一组 positives/negatives。dataset 构造不得消费影响 model initialization 的全局 RNG。

第一版只支持 **epoch-boundary resume**：

- checkpoint 仅在完整 epoch 结束后保存；
- `--resume_all` 从 `saved_epoch+1` 开始；
- 恢复 model、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、DataLoader generator、dataset epoch 和 global optimizer step；
- 不承诺 mid-epoch exact resume；CLI 若请求 step-level resume 必须报错；
- AMP 启用时仍保存/恢复 GradScaler state。

Repro-check 在同一 reference environment 比较两次 20-step loss/state/predictions，以及 uninterrupted 两 epoch与 epoch-boundary resume 的下一 epoch结果。Production 只要求配置一致、随机源可追踪并报告三 seeds mean/std，不把 checkpoint 文件字节 hash 当作数值确定性证据。

### 6.4 0/1/6/7 GT 端到端 Fixtures

建立一个 hermetic fixture manifest，包含 count `{0,1,6,7}` 四条样本和确定性的本地 NPZ features。GT windows 均合法、互不完全重复，并覆盖边界 `start=0`、`end=D_decode` 和高度重叠但不同的 moments。

同时用真实记录做数据层回归：

```text
Standard train qid=862   -> count 0
Standard train qid=517   -> count 1
Standard train qid=3432  -> count 6
Full train qid=853       -> count 7，manifest-only audit
```

Full 与 Standard 的 qid namespace 会重复，因此 fixture identity 必须是 `(dataset_setting, split, canonical_qid)`；禁止用裸 qid 跨 setting 查找 text feature。真实 Full count-7 行没有注册到 F-old Standard feature manifest 时，不得误读 `standard/clip_text/qid853.npz`，模型端到端测试使用 hermetic count-7 fixture。

单个混合 batch 必须完成：

```text
dataset load -> collate -> prepare_batch_inputs
-> B0 forward -> criterion -> finite backward
-> raw proposal decode -> legacy inference serialization -> evaluator
```

断言：

- span tensor shapes 精确为 `(0,2)/(1,2)/(6,2)/(7,2)`；
- 6/7 个 GT 不截断、不 shuffle、不 exact-dedup 掉不同 windows；
- null 样本不进入 positive localization reduction，但存在性 loss 有有限梯度；
- mixed batch loss 与逐样本 loss gating 一致；
- evaluator 使用全部 GT，`max_gt_windows=None`；
- 输出 seconds 坐标均落在 `[0,D_decode]`，输入顺序和 qid 原样保留。

### 6.5 B0 需要保存的输出

每个 seed 保存：

```text
raw proposal predictions before NMS
legacy NMS predictions
continuous legacy existence score
mAP, mR+@5, G-mIoU@1/3/5
AUROC, Rej-F1, null false-positive rate
null/single/multi grouped metrics
command, opt, environment, code archive
checkpoint file SHA256
canonical tensor-state SHA256
prediction SHA256
reproducibility.json
```

legacy existence 的固定 `0.4` operating point 只用于 B0 可比结果，同时保存连续分数。不得在 test 上重新选阈值。

## 7. CLI Execution Contract

### 7.1 使用 Lighthouse 生成正式特征

```bash
/home/guoxiangyu/miniconda3/envs/ActiveVideoPerception/bin/python \
  -m training.flash_vtg_gmr.extract_lighthouse_features \
  --mode text \
  --lighthouse_root /tmp/lighthouse \
  --openai_clip_root /tmp/openai-clip \
  --clip_weight artifacts/features/lighthouse/weights/ViT-B-32.pt \
  --slowfast_weight artifacts/features/lighthouse/weights/SLOWFAST_8x8_R50.pkl \
  --video_root data/Soccer-GMR/raw/standard \
  --split_jsonl data/label/Standard/train.jsonl \
                data/label/Standard/val.jsonl \
                data/label/Standard/test.jsonl \
  --output_root artifacts/features/f-lighthouse \
  --device cuda:0 --resume

# 两个 shard 分别使用 cuda:0/cuda:1；shard_id 为 0/1。
/home/guoxiangyu/miniconda3/envs/ActiveVideoPerception/bin/python \
  -m training.flash_vtg_gmr.extract_lighthouse_features \
  --mode video \
  --lighthouse_root /tmp/lighthouse \
  --openai_clip_root /tmp/openai-clip \
  --clip_weight artifacts/features/lighthouse/weights/ViT-B-32.pt \
  --slowfast_weight artifacts/features/lighthouse/weights/SLOWFAST_8x8_R50.pkl \
  --video_root data/Soccer-GMR/raw/standard \
  --split_jsonl data/label/Standard/train.jsonl \
                data/label/Standard/val.jsonl \
                data/label/Standard/test.jsonl \
  --output_root artifacts/features/f-lighthouse \
  --num_shards 2 --shard_id 0 --device cuda:0 --resume

# 固定 inventory 前 50 个视频，以较小 batch 重提；仅用于 batch-invariance gate。
/home/guoxiangyu/miniconda3/envs/ActiveVideoPerception/bin/python \
  -m training.flash_vtg_gmr.extract_lighthouse_features \
  --mode video \
  --lighthouse_root /tmp/lighthouse \
  --openai_clip_root /tmp/openai-clip \
  --clip_weight artifacts/features/lighthouse/weights/ViT-B-32.pt \
  --slowfast_weight artifacts/features/lighthouse/weights/SLOWFAST_8x8_R50.pkl \
  --video_root data/Soccer-GMR/raw/standard \
  --split_jsonl data/label/Standard/train.jsonl \
                data/label/Standard/val.jsonl \
                data/label/Standard/test.jsonl \
  --output_root artifacts/features/f-lighthouse-canary-b16 \
  --clip_batch_size 32 --slowfast_batch_size 16 \
  --limit 50 --balanced_sources --device cuda:0 --resume
```

### 7.2 审计并冻结 F-Lighthouse

```bash
python -m training.flash_vtg_gmr.audit_features \
  --mode existing \
  --feature_setting f-lighthouse \
  --slowfast_dir artifacts/features/f-lighthouse/slowfast \
  --clip_dir artifacts/features/f-lighthouse/clip \
  --text_dir artifacts/features/f-lighthouse/clip_text \
  --video_root data/Soccer-GMR/raw/standard \
  --split_jsonl data/label/Standard/train.jsonl \
                data/label/Standard/val.jsonl \
                data/label/Standard/test.jsonl \
  --concat_order slowfast,clip --clip_length 2 --max_v_l 75 \
  --text_store_length 77 --text_model_length 40 \
  --normalization_eps 1e-5 --fail_on_nonfinite \
  --extraction_provenance artifacts/features/f-lighthouse/extraction_provenance.json \
  --extraction_journal_dir artifacts/features/f-lighthouse/journals \
  --batch_reference_root artifacts/features/f-lighthouse \
  --batch_canary_root artifacts/features/f-lighthouse-canary-b16 \
  --batch_canary_min_videos 50 \
  --text_alignment_status verified \
  --output artifacts/features/f-lighthouse/feature_manifest.json
```

详细 provenance 规则见 `docs/hiea2m/feature_provenance_audit.md`。

F-old 只做非阻塞差异诊断：

```bash
python -m training.flash_vtg_gmr.compare_feature_corpora \
  --reference_manifest artifacts/features/f-old/feature_manifest.json \
  --generated_root artifacts/features/f-lighthouse \
  --output artifacts/features/f-lighthouse/reference_comparison.json
```

该输出不得改变 F-Lighthouse formal gate 或触发回退。

### 7.3 生成 manifests

```bash
python -m training.flash_vtg_gmr.build_manifests \
  --train_jsonl data/label/Standard/train.jsonl \
  --val_jsonl data/label/Standard/val.jsonl \
  --test_jsonl data/label/Standard/test.jsonl \
  --text_dir artifacts/features/f-lighthouse/clip_text \
  --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
  --tokenizer openai-clip-vit-b-32 \
  --context_length 77 \
  --model_length 40 \
  --no_truncate \
  --deduplicate_gt exact \
  --audit_split_identity \
  --data_output_dir artifacts/data/standard \
  --phrase_output_dir artifacts/phrase_targets/standard \
  --test_phrase_output artifacts/diagnostics/phrase_targets/test.jsonl \
  --index_output artifacts/manifests/standard/manifest_index.json
```

### 7.4 训练 B0

实现 `scripts/run_hiea2m.sh` 的 `baseline` 子命令，展开到现有 Python module 入口：

```bash
for SEED in 2024 2025 2026; do
  bash scripts/run_hiea2m.sh baseline \
    --variant B0 \
    --seed "${SEED}" \
    --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
    --data_manifest_index artifacts/manifests/standard/manifest_index.json \
    --bsz "${BATCH_SIZE:-200}" --eval_bsz 1 \
    --num_workers "${NUM_WORKERS:-4}"
done
```

单独运行 Repro-check：

```bash
bash scripts/run_hiea2m.sh baseline-repro-check \
  --variant B0 \
  --seed 2024 \
  --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
  --data_manifest_index artifacts/manifests/standard/manifest_index.json \
  --repro_check
```

wrapper 固定：

```text
--dset_name hl
--ctx_mode video_tef
--train_path <manifest_index.data_manifests.train.path>
--eval_path <manifest_index.data_manifests.val.path>
--v_feat_dirs <manifest.slowfast_dir> <manifest.clip_dir>
--t_feat_dir <manifest.text_dir>
--v_feat_dim 2816
--t_feat_dim 512
--clip_length 2
--max_q_l 40
--max_v_l 75
--max_windows -1
--no_drop_last
--txt_drop_ratio 0
```

`--repro_check` 额外强制 `num_workers=0` 和 deterministic algorithms。布尔参数必须使用 `store_true/store_false` 风格的显式 flags；禁止 `type=bool` 后传字符串 `false`，因为 Python 会将非空字符串解析为真。resolved `opt.json` 必须显示运行模式和所有随机性配置。
`baseline-repro-check` 还固定 `--max_train_steps 20 --n_epoch 1`，并保存
`repro_trace.jsonl`（逐 step batch qids、weighted loss、learning rate）。两次独立运行
必须比较 trace SHA256、最终 canonical tensor-state SHA256 和 val prediction SHA256。

B0 必须从 canonical `artifacts/data/` train/val manifests 读取 exact-dedup 后的 windows，且进程不打开 phrase targets。原始 `data/label/Standard/*.jsonl` 只作为 hash-locked source；不得让训练用 canonical manifest、validation/test 却绕回 raw source。

实际展开后的完整命令必须保存为 `command.txt`。

## 8. 实现文件

修改：

- `training/flash_vtg_gmr/dataset.py`；
- `training/flash_vtg_gmr/config.py`；
- `training/flash_vtg_gmr/train.py`；
- `training/flash_vtg_gmr/inference.py`；
- `scripts/train_flash_vtg_gmr.sh`；
- `scripts/infer_flash_vtg_gmr.sh`。

新增：

- `training/flash_vtg_gmr/audit_features.py`；
- `training/flash_vtg_gmr/build_manifests.py`；
- `training/flash_vtg_gmr/compare_feature_corpora.py`；
- `training/flash_vtg_gmr/reproducibility.py`；
- `scripts/run_hiea2m.sh`；
- `tests/test_part1_contracts.py`（集中覆盖 feature、mask、dataset、时间坐标、
  split identity、极端 GT 数量、evaluator 与 reproducibility contract）。

## 9. 必须通过的测试

### 9.1 Feature 与数值

1. F-Lighthouse loader tensor 精确等于 `[L2Norm(SlowFast) || L2Norm(CLIP)]`，`eps=1e-5`；
2. 颠倒 feature 目录顺序触发 manifest mismatch；
3. 任一 NaN/Inf/真实 zero-norm row 立即失败，训练不执行 `optimizer.step()`；
4. 生成 journal 保留 raw 两路 `T` 与显式 trim；存储两路 `T` 不同立即失败，不允许 loader 静默 `min(T)`；
5. video `T` 与 text `L_store/L_model` 相互独立；
6. normalization、concat、TEF 和 mixed batch 后仍全部 finite。

### 9.2 Text token/mask

7. `src_txt` 固定为 `B x 40 x 512`，`src_txt_mask` 来自 NPZ 而不是 collate 长度；
8. SOT/EOT IDs、padding fill value、连续 valid mask 和唯一 EOT 规则全部通过；
9. 外部 mask `True=valid`，传 PyTorch `key_padding_mask` 前恰好反相一次；
10. SOT/EOT 进入 encoder mask，但不进入 phrase `lexical_mask`；
11. 随机改变 padding hidden rows 后，pooling、attention 和 eval outputs 逐元素不变；
12. EOT index `>=40` 或 tokenizer 超过 77 时 fail-fast，不静默截断；
13. F-Lighthouse text IDs/EOT/mask 全量一致且 `text_token_alignment_status=verified`；失败时阻断全部正式实验。

### 9.3 时间坐标

14. seconds -> normalized cxw -> seconds round-trip 在 `1e-6` 内；
15. label duration 与 `D_grid` 不等时仍按 `D_grid` 编码，并按 `D_decode` 解码；
16. GT 超过 `D_label` 或 `D_decode` 的 tolerance 时失败，不静默 clamp；
17. TEF、center membership 和离散 index mapping 符合唯一公式。

### 9.4 Manifest 与 Split

18. exact duplicate GT 稳定去重并保留 first occurrence；
19. 高 tIoU 但端点不同的 GT 全部保留；
20. `count_label/exist_label` 由去重后 windows 一致生成；
21. phrase token indices 全部通过 round-trip，ambiguous 项有显式状态；
22. phrase indices 不包含 SOT/EOT 或 EOT 后的 padding positions；
23. qid、normalized vid 和 `(vid,query)` 跨 split collision 均触发失败；
24. Standard/Full 相同裸 qid 不会跨 setting 误读 text NPZ；
25. canonical data/phrase targets 物理隔离；B0/inference 不打开 phrase 文件，model forward 不接收 phrase 字段。

### 9.5 B0 与极端基数

26. 0/1/6/7 mixed fixture 完成 loader 到 evaluator 的端到端链路；
27. span shapes 为 `(0,2)/(1,2)/(6,2)/(7,2)`，无截断或 shuffle；
28. null loss gating 和 positive localization reduction 正确且 backward finite；
29. evaluator 不使用 `max_gt_windows=5` 或其他 GT 截断；
30. 两次同 seed 20-step replay 的 loss/state/prediction 一致；
31. 完整 epoch 后 resume 与不中断训练在下一 epoch 的 batch order、loss 和 state hash 一致；
32. 关闭数据修复以外的所有新功能时，B0 架构 state dict keys/shapes 与 legacy 架构一致；
33. `B0-mask-only/B0-gt-only/B0` 的 seed-2024 诊断可区分文本和 GT 修复影响；
34. 三个 B0 run 的 artifact manifests、RNG states 和 hashes 可复算。

## 10. 阶段验收与交接

本任务完成必须同时满足：

- F-Lighthouse manifest 覆盖 Standard train/val/test 引用的全部文件；
- Lighthouse commits、权重、raw/final `T`、数值、时序坐标和 split identity audits 全部通过；
- F-Lighthouse text alignment 为 `verified`；
- loader 使用固定 40-row text tensor 与独立真实 mask，不再依赖 `max_windows=5`、in-place shuffle、`min(T)` 或伪 text mask；
- canonical data manifests 完成 exact GT 去重且通过 qid/vid/split isolation；
- phrase targets 仅作为 HMSA targets/diagnostics；
- 0/1/6/7 GT 端到端测试和 deterministic replay 通过；
- 三个 B0 checkpoints 和 predictions 已锁定；
- `baseline_index.json` 能按 seed 找到 checkpoint、canonical state、RNG、feature 和 prediction hashes；
- 全部测试通过。

向 Part 2/3 交接的最小输入是：

```text
artifacts/features/f-lighthouse/feature_manifest.json
artifacts/manifests/standard/manifest_index.json
artifacts/baselines/baseline_index.json
```

若任一文件缺失或 hash 不匹配，后续任务必须停止，不能自行重提特征或重训另一个 B0。

Part 2 不得把缺失的候选接口误判为 Part 1 未完成；它应在读取上述三个交接文件后实现无参数的 candidate export，再构建公共 Adapter。
