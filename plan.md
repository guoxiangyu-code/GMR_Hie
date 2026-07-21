# HieA2M：从空间对象集合到时序事件集合

## 1. 研究目标

本文只验证两个来自 HieA2G 的核心思想能否迁移到 GMR-FlashVTG：

1. **Temporal HMSA**：把 word-object、phrase-object、text-image 三层对齐改写为 token-clip/event、phrase-event、query-video 三层时序对齐；
2. **Adaptive Event Cardinality（AEC）**：在去重后的 event modes 上联合预测空集、单事件和多事件，并引入 HieA2G 的 count contrastive learning。

论文暂定名为 **HieA2M**。AEC 强调预测的是语义事件数量，而不是 FlashVTG 产生的重叠 window proposals；AWC（Adaptive Window Counter）和 AMC（Adaptive Moment Counter）只作为备选命名，不在正文混用。

核心假设是：

> HieA2G 的层次化语义对齐与自适应目标计数，可以从空间对象集合迁移到时序事件集合，但必须增加一个 proposal-to-event adapter，把密集且重复的 temporal proposals 转换为可计数的 event modes。

## 2. 转移结论与边界

### 2.1 可以完整实现的时序对应关系

```text
HieA2G                         HieA2M
word-object             ->     token-clip/event
phrase-object           ->     phrase-event (multi-positive)
text-image              ->     query-video
adaptive object count   ->     adaptive event cardinality
```

三层 temporal analogue 都能用现有 Soccer-GMR 标注和 Lighthouse 特征实现，并能直接回流到最终 event score、boundary 和 count prediction。

### 2.2 不能声称原样复现的部分

Lighthouse 标准流程对每个 2 秒 clip 输出：

- 一个 512-D CLIP ViT-B/32 全局图像向量；
- 一个 2304-D SlowFast 向量；
- `VisionEncoder("clip_slowfast", ...)` 内部按 `[CLIP(512), SlowFast(2304)]` 拼接为 `(T, 2816)`。

现有文本特征为 `(77, 512)` contextual CLIP rows，并带 `(77,)` attention mask。该流程没有 patch、player、object tokens，Soccer-GMR 也没有 Flickr30K Entities 式 phrase-box 标注。因此：

- 可以实现完整的**时序三级对齐**；
- 不能宣称复现了 HieA2G 的 object-level information path；
- 重新运行相同 Lighthouse pipeline 只能统一特征来源，不会自动增加空间细粒度监督。

多帧 CLIP、patch tokens 或 player tracking 只作为后续 feature-upgrade 消融，不能和主模型结构改动同时引入。

### 2.3 Feature Gate：先确认特征，再开始方法实验

特征方案是阻塞项。完成以下 Gate 前，不启动 P0/H/C/F 训练。

**F-old：冻结现有 corpus。** 当前视频文件分别保存为 `(T,2304)` SlowFast 和 `(T,512)` CLIP，文本为 `(77,512)` contextual rows 加 `(77,)` mask。仓库按 `--v_feat_dirs` 顺序逐路 L2 normalize 后再 concat；现有脚本传入 `slowfast clip`，所以基线实际输入顺序是：

```text
video_content = [SlowFast(2304) || CLIP(512)]
```

所有 B0、AGC-Direct、P0、H、C、F 主实验必须使用这一冻结顺序和同一 corpus checksum，保证旧 checkpoint 可比性。

**F-new：用原视频做可复现性 canary。** 固定已审计 Lighthouse commit `d095eaa552cecef240897a8b750306b3b2a08740`，调用：

```text
VisionEncoder("clip_slowfast", clip_len=2, framerate=0.5,
              size=224, device=DEVICE, slowfast_path=WEIGHT)
TextEncoder("clip_slowfast", device=DEVICE)
```

Lighthouse 的 combined video output 必须拆回两个 NPZ：前 512 维写入 `clip/`，后 2304 维写入 `slowfast/`；训练仍按 `slowfast clip` 顺序读入。文本保存 `last_hidden_state[0]` 和 `attention_mask[0]`。禁止把 combined 2816-D 文件作为单路输入，因为这会把“两路分别 normalize”改成“拼接后统一 normalize”。

先对固定的 50 个 train 视频及其全部 queries 运行 canary，并生成 `feature_manifest.json`，至少记录：

```text
lighthouse_commit, openai_clip_commit
clip_weight_sha256, slowfast_weight_sha256
ffmpeg/python/torch/cuda versions
clip_len, sampling_fps, crop_size, dtype
output_dims, concat_order, per_stream_normalization
video_inventory_sha256, query_inventory_sha256
```

canary 在 loader normalization 前按 CLIP/SlowFast 两路分别比较。通过条件固定为：row count 100% 一致、所有文件 shape/key/mask 一致、逐 row mean cosine `>=0.999`、1% 分位 cosine `>=0.995`、median relative-L2 `<=1e-3`。如果未通过：

1. 论文主实验继续使用 F-old，并明确“旧特征 provenance 不完整”；
2. F-new 只能全量重提 train/val/test 后作为独立 feature-setting 重跑 B0 和 F2；
3. 禁止在同一实验中混用 F-old/F-new，也禁止只重提某个 split。

所有提取文件使用临时文件加原子 rename，支持按 manifest 断点续跑。最终 `feature_manifest.json` 的 SHA256 必须写入每个 run 的 `opt.json`。

## 3. 已核验的实现事实

### 3.1 数据基数

Standard split 的 GT event count 分布为：

| Split | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2002 | 1423 | 565 | 117 | 23 | 6 | 2 | 6 |
| val | 210 | 165 | 72 | 16 | 2 | 0 | 0 | 4 |
| test | 492 | 384 | 128 | 20 | 10 | 2 | 0 | 5 |

Full split 的最大 count 为 7。因此使用 `M=10` 个 event modes 足够覆盖当前数据，但 loader 必须取消 `max_windows=5` 的随机截断，也不能原地 shuffle `relevant_windows`。

同一 query 的不同 GT moments 可能高度重叠，实测最大 tIoU 为 0.857。高 tIoU 不能作为“同一事件”的充分条件，不能在 GT 或预测端直接执行高阈值 union。

### 3.2 当前 FlashVTG-GMR 的关键限制

- 多尺度候选仍是密集 proposal cloud，多个高分 span 可能只是同一事件的边界变体；
- 当前 NMS 后处理忽略部分配置参数，官方评测还会截取前 10 个预测；
- existence head 主要使用 pooled query/video feature，不能可靠表达事件数量；
- `SetCriterion.forward()` 会先过滤 positive query，新加入的 null/count/global loss 必须在该过滤之前计算；
- dataset loader 没有正确使用保存的 text attention mask；
- 预计算 CLIP text rows 已经上下文化，简单把 action token row 置零会产生 masked-recovery 语义泄漏。

这些问题是实现约束，不单独包装成论文贡献。

## 4. 总体架构

```text
Lighthouse video rows + CLIP query rows
                    |
              FlashVTG backbone
                    |
          dense temporal proposals
                    |
       Proposal-to-Event Adapter
                    |
        M=10 event modes {e_m}
                    |
 Temporal HMSA (optional in ablation)
 token / phrase / query-video
                    |
       aligned event representation
                    |
      event prediction + AEC count/null
                    |
       score + boundary + output set
```

方法结构固定为：

- **核心模块一：Temporal HMSA**；
- **核心模块二：AEC**；
- **必要适配层：Proposal-to-Event Adapter**；
- **可选辅助一：Visual Event-to-Phrase Reconstruction（V-EPR）**；
- **可选辅助二：分阶段/多任务训练后小学习率联合微调**。

Proposal relation、semantic/quality score decomposition、NMS 修正都只属于 adapter 或基础设施，不作为 HieA2G 迁移贡献。

## 5. 公共数据与张量接口

### 5.1 输入

```text
slowfast_feat : B x T x 2304
clip_feat     : B x T x 512
video_content : B x T x 2816   # [SlowFast || CLIP]，两路分别 normalize，不含 TEF
video_mask    : B x T
query_tokens  : B x L x 512
query_mask    : B x L
query_global  : B x 256        # mask-aware pooling 后的投影
```

`video_content` 是所有 visual-only 辅助任务的唯一输入。TEF 可以继续用于定位分支，但不得进入 V-EPR 或视觉语义重建目标。

### 5.2 FlashVTG 需要暴露的候选

```text
candidate_feat   : B x K x 256
candidate_mask   : B x K
candidate_span   : B x K x 2      # 排序前、归一化到 [0,1]
candidate_logit  : B x K
candidate_point  : K x 4
candidate_scale  : K
```

主实验取 `K=50`。feature、mask、span、point 和 scale 必须使用同一 flatten/sort index。开启新模型时，`ConvPyramid` 在 train/eval 都返回 mask；关闭时保持 legacy 输出完全不变。

### 5.3 Phrase manifest

为每条 query 离线保存：

```text
qid
source                     # SportsMoments / WC2022
action_labels
team_labels
action_token_indices
team_token_indices
query_attention_mask
template_id
relevant_windows           # 保留全部 GT 和原始顺序
```

action/team indices 必须用与现有 features 相同的 OpenAI CLIP tokenizer 生成并通过 decode round-trip 检查。该 manifest 只进入 criterion targets，不进入 model inputs。SportsMoments 没有 team supervision，其 team 分支在训练和推理时均 mask 为 0。

## 6. 必要适配层：Proposal-to-Event Adapter

AEC 不能直接统计 raw proposals。Adapter 的唯一职责是：

```text
overlapping proposals -> a small set of event modes
```

### 6.1 Event-mode decoder

先用轻量 relation encoder 编码 `candidate_feat/span/scale/logit`。再从 `K=50` 个 proposals 中选择 `M=10` 个 seeds：第一个取最高分候选，后续按下式做 greedy diversity selection，不使用 hard tIoU suppression：

```text
seed_score_i = normalized_candidate_score_i
             + lambda_div * min_j(1 - cosine(r_i, r_seed_j))
```

`lambda_div` 只在 validation 上选择。这样既减少同一边界变体占满 slots，也保留可能高度重叠的不同 GT events。每个 seed 再加入不同的 learned slot embedding：

```text
e_m^0 = W_c candidate_feat(seed_m) + slot_embedding_m + W_q query_global
```

两层轻量 decoder 依次执行：

1. mode self-attention，建模 event 之间的竞争；
2. mode-to-proposal cross-attention，聚合同一事件的边界变体；
3. FFN 与 LayerNorm。

Adapter 输出：

```text
event_feat_m
event_logit_m
quality_logit_m
event_span_m
```

Adapter 明确定义为两个版本。

**P0-selection（公共主版本）**：

```text
event_span_m = stop_gradient(seed_span_m)
```

seed selection 是离散操作，span 来自冻结的 FlashVTG 候选，因此该版本不声称学习边界。

**P0-residual（Adapter 内部消融）**：

```text
rho = 0.5
event_span_m = stop_gradient(seed_span_m)
             + tanh(delta_m) * rho * stop_gradient(seed_duration_m)
```

`rho` 固定为 0.5，不调参。只有 `delta_m` 由 Adapter 预测并接收边界梯度。
residual head 的最后一层权重和 bias 零初始化，使 P0-R 初始边界严格等于 seed span。

### 6.2 一对一集合监督

使用 Hungarian matching 将 modes 与 GT events 一对一匹配：

```text
C(m,j) = 2 * L1(span_m, gt_j)
       + 2 * (1 - tIoU(span_m, gt_j))
       - sigmoid(event_logit_m)
```

- matched mode：event target 为 1；
- unmatched mode：event target 为 0；
- 所有有效 modes 都计算 localization-quality loss；
- null query：所有 modes 都是 no-event、quality target 为 0，不计算 boundary loss。

quality 使用连续、detached target：

```text
quality_target_m = stop_gradient(max_j tIoU(event_span_m, gt_j))
L_quality = SmoothL1(sigmoid(quality_logit_m), quality_target_m)
```

null query 的 `quality_target=0`。该 loss 更新 quality head 及其上游 event feature，但不通过 target 向 span 反传。

P0-selection 的 L1/tIoU 只参与 Hungarian cost，不加入训练 loss，因为 `event_span_m` 不依赖任何可训练参数：

```text
L_adapter_selection = L_event + L_quality
```

P0-residual 才启用 span regression：

```text
L_adapter_residual = L_event
                   + 5 * L_span-L1
                   + 2 * L_span-tIoU
                   + L_quality
```

一对一监督负责把同一 GT 的重复 proposals 压成一个 active mode。推理阶段不再对 event modes 执行普通 NMS，因为几何重叠不等于语义重复。

### 6.3 Adapter 的定位

Adapter 是 Temporal HMSA 和 AEC 共用的输入层，不作为第三个核心创新。论文只需证明：

- event-mode GT coverage 足以承接原 FlashVTG proposals；
- duplicate-event rate 明显低于 raw proposal/NMS 输出；
- adapter 本身没有显著损害 baseline mAP。

这里的 duplicate 和 full coverage 使用唯一、GT-conditioned 定义。对 query `q` 的最终预测集合 `P_q` 和 GT 集合 `G_q`，在边集合 `tIoU(p,g)>=theta` 上求**最大基数、再最大总 tIoU**的一对一 bipartite matching `M_theta(q)`。令：

```text
P_eligible = {p in P_q | exists g in G_q: tIoU(p,g) >= theta}

DuplicateRate_theta(q)
  = (|P_eligible| - |M_theta(q)|) / max(|P_eligible|, 1)

FullCoverage_theta(q)
  = 1[|M_theta(q)| = |G_q|]       # 只在 |G_q| >= 2 上汇总
```

数据集级 `DuplicateRate` 在 positive queries 上做 micro aggregation，即分子、分母分别求和后再相除；不对大量零命中 query 的 0 值做 macro average。`FullCoverage` 在 `|G_q|>=2` 的 queries 上做 macro mean。未命中任何 GT 的 prediction 是 false positive，不计作 duplicate；两个相互高度重叠、但能一对一匹配两个 GT 的 predictions 也不计作 duplicate。主阈值固定 `theta=0.5`，补充报告 `0.3/0.7`。同时报告：

- `Selected-FullCoverage`：在最终输出集合上计算，是主指标；
- `Oracle-Mode-FullCoverage`：在全部 `M=10` modes 上计算，只诊断 adapter 上限；
- `DuplicateRate`：在最终输出集合上计算。

### 6.4 固定公共 Adapter 协议

为避免 HMSA/AEC 的收益混入 adapter 重训差异，P0 是固定公共基础：

1. 对 seeds `{2024,2025,2026}`，分别从对应 B0 checkpoint 训练一次 P0；
2. P0 checkpoint 只依据 validation 的 adapter 指标选择，选择规则在首个实验前锁定；
3. 每个 seed 的 H1/H2/H3/C1/C2/F1/F2/R1 必须加载该 seed 同一个 `adapter_ckpt`；
4. FlashVTG backbone 和 Adapter 参数在主归因矩阵中全部冻结，optimizer 参数组中不得出现它们；
5. 启动时记录并校验 B0、P0、feature manifest 三个 SHA256；
6. 只有单独命名的 `F2-unfreeze` 可以联合解冻，结果放辅助表，不能替代 F2 主结果。

`adapter_ckpt` 是完整 P0 checkpoint，包含其父 B0 weights、Adapter weights、seed、feature manifest hash 和父 B0 hash；下游一次加载后分别冻结 `backbone.*` 与 `event_adapter.*`。它不是只保存若干 adapter keys 的松散 state dict。

主归因矩阵中的 `P0` 固定指 **P0-selection**。`P0-residual` 记为 `P0-R`，只进入 Adapter 内部消融，不得替换 H1-H3/C1-C2/F1-F2 的公共 P0；否则需要把整套主矩阵全部重跑，不能混用。

P0/H1/H2/H3 没有 AEC，其 final set 统一定义为 `sigmoid(event_logit)>=0.5` 的 modes；阈值固定为 0.5，不为各 HMSA 变体单独调参。AEC 中的 `tau_mode` 只处理预测为 4+ 时的集合尾部，不能反向用于重新选择 P0。

P0 checkpoint 选择分数固定为：

```text
AdapterScore = HarmonicMean(
    Oracle-Mode-FullCoverage@0.5,
    1 - DuplicateRate@0.5
)
```

并约束 validation mAP 相对 B0 下降不超过 0.5。下游实验不得根据 H/C/F 的结果重新选择或重训 P0。

## 7. 核心模块一：Temporal HMSA

### 7.1 第一层：Token-Clip/Event Alignment

本计划只实现下面这一种 Token-Clip MIL，不并行尝试 BCE、point supervision 或其他 pooling。把 query tokens 和未融合 query 的视频 rows 投影到相同维度：

```text
A_word[k,t] = (W_q q_k)^T (W_v v_t) / sqrt(d)
```

Temporal MIL 只作用于完整查询。`token_group_full` 包含除 SOT/EOT/PAD 外的全部有效 tokens，并使用 normalized LogMeanExp：

```text
u_full[t] = tau * log(
    mean_{k in token_group_full} exp(A_word[k,t] / tau)
)
```

对 positive query 的每个 GT `g_j=[s_j,e_j]`，按 clip center 构造：

```text
B_j+ = {t | clip_center(t) in [s_j, e_j]}
B-   = {t | clip_center(t) outside union_j [s_j-2s, e_j+2s]}

bag(x, B) = tau * log(mean_{t in B} exp(x[t] / tau))

L_full-temporal(q)
  = 1/J * sum_j softplus(
        margin + bag(u_full, B-) - bag(u_full, B_j+)
    )

L_token-temporal = L_full-temporal
```

固定 `tau=0.1`、`margin=0.2`，不作为搜索超参数。每个 GT 单独形成 positive bag，避免 multi-target query 只激活其中一个时刻；负 bag 使用一整个 clip 的边界 guard，并通过 LogMeanExp 做 hard-negative aggregation，而不是把每个外部 clip 标成负类。`B-` 为空时跳过该 query 的此项 loss。null query 不计算该 loss。

action/team **不使用** `B-`：队伍或动作在完整 GT 外出现是合理现象，把全视频外部区域当作 factor negative 会制造假负监督。主模型对 action/team 只使用 matched event 的 token attention、phrase-event positives，以及标签明确不匹配的 event-mode negatives。

event mode 再对 token features 做独立 factor attention：

```text
R_event(q) = {full, action} + {team if WC2022}

alpha_mr = softmax((W_e_r event_feat_m) @ (W_q_r query_tokens)^T + query_mask)
c_mr     = sum_k alpha_mr[k] * query_tokens[k]

L_r-event
  = mean_{matched m} -mean_{k in token_group_r} log alpha_mr[k]

L_token-event = L_full-event + L_action-event + L_team-event

c_m_word = W_word([c_m_full, c_m_action, c_m_team])
```

full/action/team 使用独立 attention heads；SportsMoments 不计算 `L_team-event`，team head 输出固定为 0。manifest token indices 只构造训练 target；推理时各 head 自己产生 `alpha_mr`，model forward 不接收 token indices。

如果后续增加 action/team temporal negatives，只允许使用同时满足以下条件的 event regions：同视频中具有明确 factor 标签、其标签集合与当前 factor 标签集合无交集、且不与当前 GT windows 重叠。该扩展必须单独命名，不能把所有 `B-` 恢复为 action/team negatives。

### 7.2 第二层：Phrase-Event Alignment

manifest indices 只在 criterion 中从 contextual CLIP rows 构造 stop-gradient 训练 targets：

```text
p_action = Mean(query_tokens[action_token_indices])
p_team   = Mean(query_tokens[team_token_indices])
p_full   = MaskedMean(query_tokens)
```

主预测分支使用上一节模型自己预测的 `alpha_mr` 得到 phrase context：

```text
c_m_phrase = W_p([c_m_full, c_m_action, c_m_team])
```

不得把由 oracle token indices mean-pool 的 `p_action/p_team` 直接送入 event refinement 或 inference。

用独立 projection 得到 128-D normalized phrase/event embeddings。一个 query 的全部 matched GT modes 都是该 phrase 的 positives，因此使用 multi-positive supervised contrastive loss，而不是一对一 phrase-object label：

```text
L_phrase = L_multi-positive(action) + L_multi-positive(team) + L_multi-positive(full)
```

负样本规则保持保守：

- 同 query 中明确 unmatched 且 `max_tIoU < 0.1` 的 mode 可作 full-query negative；
- 跨 query 只有已知 factor label 不匹配时才进入相应 factor denominator；
- unknown、多标签冲突和模糊重叠样本不作为 factor negatives；
- null query 没有 phrase-to-GT-event positive，不计算该 loss。

### 7.3 第三层：Query-Video Alignment

构造全局视频表示：

```text
v_global = MLP([
    MaskedMean(video_content),
    MaskedMax(video_content),
    WeightedMean(event_feat, sigmoid(event_logit))
])

s_qv = cosine(W_q query_global, W_v v_global)
```

训练信号：

- 至少一个 GT event：query-video positive；
- null query：query-video negative；
- 与视频共享 action/team 但完整 query 不成立的 null 样本：hard negative；
- 主损失使用这些显式标注 pair 的 binary matching loss。

对称 query-video contrastive 只作为受控消融，并使用 relation mask：任意 batch-shuffled query-video pair **不默认是 negative**。只有数据标注明确证明不匹配的 pairs 才进入 denominator；其他跨视频组合记为 unknown 并 mask。主方案优先使用原始 positive pairs、显式 null pairs，以及同视频共享部分 action/team 但完整查询不成立的 hard negatives，避免其他视频中的 query 恰好也适用于当前视频而形成假负样本。

该层输出 `c_global` 和 `query_video_logit`，直接进入 event refinement 与 AEC，而不是只做独立辅助分类。

### 7.4 Alignment 必须回流到最终预测

三层 context 共同更新 event mode：

```text
aligned_event_m = LayerNorm(
    event_feat_m
    + gamma_w * c_m_word
    + gamma_p * c_m_phrase
    + gamma_g * c_global
)
```

`gamma_w`、`gamma_p`、`gamma_g` 零初始化，使新增模块初始时严格退化为 adapter 输出。`aligned_event_m` 必须用于：

- event probability；
- localization quality；
- AEC 的 event summary。

使用公共 P0-selection 的 H1-H3/C1-C2/F1-F2 均保持 seed spans，不新增 boundary regression。边界 residual 只属于 P0-R 消融；它不能在 HMSA/AEC 主实验中被隐式重新启用。

HMSA 开启时，以 `aligned_event_m` 重新计算 event/quality logits；HMSA 关闭的 C1/C2 对照则直接使用 adapter logits。下文的 `event_logit_m` 统一指当前分支最终用于推理的 logits。

不接受“FlashVTG prediction branch + 与输出断开的 HMSA auxiliary head”作为主实现。

### 7.5 Temporal HMSA loss

```text
L_HMSA = lambda_word   * L_token-temporal
       + lambda_token  * L_token-event
       + lambda_phrase * L_phrase-event
       + lambda_global * L_query-video
```

初始权重统一设为 `0.1`，只通过 validation 调整一次。每层都要做独立消融，不能只报告全部开启的结果。

## 8. 核心模块二：Adaptive Event Cardinality

### 8.1 集合级计数表示

令 `p_m = sigmoid(event_logit_m)`，从 aligned event modes 构造：

```text
event_mean     = sum_m p_m * aligned_event_m / (sum_m p_m + 1e-6)
event_max      = max_m aligned_event_m
expected_count = sum_m p_m

g_count = MLP([
    query_global,
    c_global,
    event_mean,
    event_max,
    expected_count
])
```

### 8.2 主版本：AEC-CE

为保持与 HieA2G AGC 的对应关系，主分类空间使用：

```text
count_class = {0, 1, 2, 3, 4+}
count_logits = Linear(g_count)
P_CE = softmax(count_logits)
```

class weights 只由 train split 统计，使用 effective-number weighting 并裁剪到 `[0.5, 2.0]`。最终 CE 在 8.5 的统一 `P_AEC` 上计算。

主实验编号锁定为：`C1=AEC-CE`、`C2=AEC-CE+Con`、`F1=HMSA+AEC-CE`、`F2=HMSA+AEC-CE+Con`。PB/Exact 只能使用独立消融名，不能通过参数改变 C1/C2/F1/F2 的含义。

同时报告 `{0,1,...,7}` exact-count head 作为数据集特定消融。由于高 count 样本极少，它不能替代 `{0,1,2,3,4+}` 主结果。

### 8.3 概率版本：AEC-PB

由 event activity probabilities 通过动态规划构造 Poisson-binomial 分布：

```text
P_PB(N=0..M) = PoissonBinomial(p_1, ..., p_M)

P_PB^5 = [
    P_PB(0),
    P_PB(1),
    P_PB(2),
    P_PB(3),
    sum_{n=4..M} P_PB(n)
]

y_5 = bin(n_gt) in {0,1,2,3,4+}
L_PB-bin = -w[y_5] * log P_PB^5[y_5]
```

`w` 与 AEC-CE 完全共享，只从 train 的五类分布估计。这样 count CE、PB 和 count contrastive 使用同一标签空间，不为 count=5/6/7 单独估计不稳定权重。

可选 exact 项不使用稀有类权重：

```text
L_PB-exact = -log(P_PB(n_gt) + 1e-8)
L_PB = L_PB-bin + 0.1 * L_PB-exact       # 仅 C-PB-Exact 消融
```

主 `C-PB/C-PB-Con/F-PB` 只使用 `L_PB-bin`。

比较两种实现：

- **AEC-CE**：独立 count classifier；
- **AEC-PB**：count 完全由 event probabilities 导出。

AEC-CE 中增加 event-count consistency：用 `P_PB^5` 与 CE posterior 做 symmetric KL：

```text
L_consistency = 0.5 * KL(P_CE || P_PB^5) + 0.5 * KL(P_PB^5 || P_CE)
```

这保证 count prediction 和 event activity 表示同一个集合，而不是两个互相矛盾的输出头。AEC-PB 没有独立 CE posterior，因此不计算该 KL。

### 8.4 Count contrastive learning

对 `g_count` 使用 `{0,1,2,3,4+}` 标签和 supervised contrastive loss：

```text
L_count-con(i) = -1/|P(i)| * sum_{p in P(i)}
    log exp(sim(g_i,g_p)/tau)
        / sum_{a != i} exp(sim(g_i,g_a)/tau)
```

使用 class-balanced memory queue，避免 batch 中稀有 multi-count 类没有正样本。必须按 action、source、query template 和 null/single/multi 分层报告结果，检查模型是否只学习 count prior。

### 8.5 No-target 与输出集合

AEC-CE 以 `log P_CE` 为 base logits；AEC-PB 以 `log(P_PB^5+1e-8)` 为 base logits。两者都是固定的 `{0,1,2,3,4+}` 五维空间。`query_video_logit=s_qv` 只在 softmax 前调整“0 vs non-zero”的相对质量：

```text
delta_qv = softplus(a) * s_qv
z_0 = base_logit_0 - 0.5 * delta_qv
z_n = base_logit_n + 0.5 * delta_qv           # n > 0

P_AEC(N=n) = softmax(z / T_count)[n]
L_count    = -class_weight[y] * log P_AEC(N=y)
```

AEC-CE 与 AEC-PB 都使用 `y=bin(n_gt) in {0,1,2,3,4+}` 和同一 `class_weight[y]`；对于 PB，上式中的最终 `P_AEC` 替代 pre-fusion `P_PB^5` 计算 `L_PB-bin`。`a` 可训练，`T_count` 只在 validation 上拟合。

**唯一空集决策**为：

```text
pred_count = argmax_n P_AEC(N=n)
output = empty set iff pred_count == 0
```

不再计算第二个 existence hard gate。`p_nonempty=1-P_AEC(N=0)` 只用于 AUROC、Rej-F1、calibration curve 和部署风险排序，不能再次清空非零 count 的输出。

非空输出规则为：

1. `pred_count=1/2/3`：从 event modes 中按 aligned event score 选择 Top-N；
2. `pred_count=4+`：选择 `p_m >= tau_mode` 的 modes，至少保留 Top-4，最多 `M=10`；
3. 只有 `C-Exact` 的 `pred_count>=4` 直接选择 Top-N；PB 与 CE 一样使用 4+ 规则；
4. 任何 Top-N 都只能作用在 event modes 上，不能回到 raw proposal cloud；
5. event modes 不再经过普通 NMS。

只有 `tau_mode` 和 `T_count` 需要 validation；test 不重新调参。prediction JSONL 同时保存完整 ranked modes、`P_AEC`、`pred_count`、连续 `p_nonempty` 和最终 selected set。

### 8.6 AGC-Direct 简单迁移基线

AGC-Direct 不使用 Proposal-to-Event Adapter，也不使用 Temporal HMSA。它冻结 B0，直接从基线全局表示预测 `{0,1,2,3,4+}`：

```text
g_direct = MLP([
    query_global,
    MaskedMean(baseline_video_memory),
    MaskedMax(baseline_video_memory)
])

P_AGC = softmax(Linear(g_direct) / T_count)
L_G0 = WeightedCE(P_AGC, count_class)
L_G0_con = L_G0 + 0.1 * L_count-con
```

`G0` 只迁移 AGC count classifier；`G0-Con` 才加入 HieA2G count contrastive。两者使用相同 B0、global features、count head 结构和推理规则。

其唯一空集规则同样是 `argmax P_AGC == 0`。预测 1/2/3 时，从 **B0 legacy post-NMS proposals** 取 Top-N；预测 4+ 时保留分数超过 validation 阈值的 proposals，至少 Top-4、最多 10。G0 复用 B0 proposals/NMS，但关闭 B0 existence hard gate；AGC-Direct 禁止调用新 Adapter 或任何额外 existence gate。

该基线不是推荐方法，而是检验“全局 AGC + raw proposal selection”能否解决 GMR；它与 P0/AEC 的差距量化 proposal-to-event 适配的必要性。

## 9. No-target-aware Loss Gating

Null sample 不是所有损失的简单全负样本。每项 loss 必须按其语义决定是否启用：

| Loss | Positive query | Null query |
|---|:---:|:---:|
| FlashVTG boundary / GT matching | 是 | 否 |
| Adapter matched span loss | 是 | 否 |
| Adapter no-event loss | matched/unmatched | 所有 modes |
| Full-query temporal MIL | 是 | 否 |
| Token-to-event attention | matched modes | 否 |
| Phrase-to-GT-event | 是 | 否 |
| V-EPR | 是 | 否 |
| Query-video match | positive | negative/hard negative |
| Count loss | `count>0` | `count=0` |
| Event-count consistency | 是 | 是 |

所有新 loss 必须在当前 criterion 的 positive-only legacy localization filter 之前计算。null query 不能被送入需要视觉正证据的 reconstruction 或 boundary loss。

## 10. 可选辅助：Visual Event-to-Phrase Reconstruction

HieA2G 的 masked text recovery 不能直接套到预计算 contextual CLIP rows。安全的时序替代是从纯视觉事件特征恢复 phrase embedding：

```text
visual_event = TemporalRoIAlign(video_content, matched_event_span)
pred_action  = MLP_action(visual_event)
pred_team    = MLP_team(visual_event)

L_VEPR = 1 - cosine(pred_phrase, stop_gradient(target_phrase))
```

约束：

- 输入只能来自未融合 query 的 `video_content`；
- 不得使用 aligned event、mode-to-token attention 或 query embedding；
- null query 不计算 V-EPR；
- SportsMoments 的 team head、team loss 和 team semantic residual 全部 mask 为 0；
- V-EPR 预测与 phrase context 的相似度通过零初始化 residual 加回 event logit，确保它能影响最终预测。

只有获得与现有 Lighthouse features 完全一致的 OpenAI CLIP implementation/weights，并验证重编码数值一致后，才考虑严格 masked-query recovery。该实验不属于主模型必要组成。

## 11. 训练方案

### 11.1 阶段划分

1. **S-F Feature Gate**：确认 F-old manifest；可并行完成 F-new canary，但 canary 未通过不能替换主特征；
2. **S0 Baseline**：三个 seeds 分别复现 B0，冻结 checkpoint、prediction JSONL 和协议；
3. **S0-D Direct**：在各 B0 上分别训练 G0 和 G0-Con；
4. **S1 Public Adapter**：各 seed 冻结 B0，只训练一次 P0-selection 并锁定 checkpoint SHA256；P0-residual 作为 P0-R 独立消融，不进入下游主矩阵；
5. **S2 Attribution**：从对应 P0 初始化并冻结 B0+P0，分别训练 H1/H2/H3/C1/C2；
6. **S3 Joint**：仍冻结 B0+P0，只联合训练 HMSA 与 AEC，得到 F1/F2；
7. **S4 Optional**：冻结规则不变，在 F2 上增加 V-EPR 得到 R1；
8. **S5 Unfreeze**：仅 `F2-unfreeze` 解冻 adapter/backbone 的最后层，进入辅助表。

P0、H/C/F 新增模块使用 `lr=3e-5`，训练 10-15 epochs 并由 validation early stopping；`F2-unfreeze` 使用 `lr=1e-5` 微调 5-10 epochs。主归因矩阵不允许通过下游模型反向更新公共 Adapter。

### 11.2 总损失

```text
L = L_FlashVTG
  + lambda_adapter * L_adapter
  + lambda_HMSA    * L_HMSA
  + lambda_count   * L_count
  + lambda_con     * L_count-con
  + lambda_cons    * L_consistency
  + lambda_VEPR    * L_VEPR
```

`L_adapter` 由 variant registry 唯一解析：P0/H/C/F/R 使用 `L_adapter_selection`，只有 P0-R 使用 `L_adapter_residual`。主实验不得出现 `L_span-L1/L_span-tIoU` 的 Adapter 权重。

推荐初值：

```text
lambda_adapter = 1.0
lambda_HMSA    = 0.1
lambda_count   = 1.0
lambda_con     = 0.1
lambda_cons    = 0.1
lambda_VEPR    = 0.1   # 仅 R1
```

训练时记录每项 raw loss、加权 loss、梯度范数、matched-mode recall 和 active-mode 数量，避免某个辅助项静默主导优化。

### 11.3 多任务预训练定位

普通 VMR/HD 数据可用于预训练 token-clip、phrase-event、boundary 和 saliency；Soccer-GMR 再学习 null、多事件和 AEC；最后联合微调。该策略属于训练增强，不作为与 Temporal HMSA、AEC 并列的第三个贡献。第一版先在同一 Soccer-GMR 数据上完成可控验证，再决定是否引入额外预训练数据。

## 12. 最小实验矩阵

所有 H/C/F 实验按 seed 使用同一个、已冻结的 S1 public adapter SHA256，避免把去重能力误记为 HMSA 或 AEC 的收益。

| 实验 | Adapter | Temporal HMSA | Count module | Count contrastive | 用途 |
|---|:---:|:---:|:---:|:---:|---|
| B0 |  |  |  |  | 原 FlashVTG-GMR |
| G0 |  |  | AGC-Direct CE |  | 只直接迁移 count classifier |
| G0-Con |  |  | AGC-Direct CE | ✓ | 再迁移原始 count contrastive |
| P0 | ✓ |  |  |  | 公共适配层对照，不作为核心贡献 |
| H1 | ✓ | Token |  |  | 验证细粒度对齐 |
| H2 | ✓ | Token+Phrase |  |  | 验证短语层增益 |
| H3 | ✓ | 全三层 |  |  | 完整 Temporal HMSA |
| C1 | ✓ |  | AEC-CE |  | 验证 event count/null |
| C2 | ✓ |  | AEC-CE | ✓ | 验证 event-level count contrastive |
| F1 | ✓ | ✓ | AEC-CE |  | 两个核心模块组合 |
| F2 | ✓ | ✓ | AEC-CE | ✓ | 完整主模型 |
| R1 | ✓ | ✓ | AEC-CE | ✓ | F2 + V-EPR，可选 |

Adapter 内部消融：

| 实验 | Span | Span regression loss | 下游公共 Adapter |
|---|---|:---:|:---:|
| P0 | frozen seed span |  | ✓ |
| P0-R | seed + `rho=0.5` residual | ✓ |  |

AEC 内部消融：

| 实验 | Count source | Contrastive | 目的 |
|---|---|:---:|---|
| C-PB | PB 聚合五类 |  | 验证显式集合基数 |
| C-PB-Con | PB 聚合五类 | ✓ | PB 的 count representation |
| C-PB-Exact | PB 五类 + `0.1 L_PB-exact` |  | exact PB 辅助项 |
| C-Exact | `{0,...,7}` classifier |  | 数据集特定上界消融 |
| F-PB | HMSA + PB 聚合五类 | ✓ | 联合模型 PB 消融 |

核心归因比较固定为：`G0 -> C1` 衡量 event modes，`G0 -> G0-Con` 衡量原始 AGC contrastive，`C1 -> C2` 衡量 event-level contrastive。

每个关键模型运行 3 个 seeds，报告 mean/std 和 paired bootstrap 95% CI。

## 13. 评测与诊断

### 13.1 主任务指标

- mAP；
- R@1、R@5 或仓库对应的 `mR+@5`；
- G-mIoU@1/3/5；
- AUROC、Rej-F1；
- null false-positive rate；
- multi-target full coverage。

### 13.2 Temporal HMSA 诊断

- score-IoU Spearman correlation；
- token temporal pointing accuracy / inside-outside ranking accuracy；
- action/team token attention accuracy；
- phrase-event retrieval Recall@K；
- query-video AUROC；
- null 中“局部匹配但整体不成立”的 hard-negative accuracy。

### 13.3 AEC 诊断

- exact count accuracy 和 `{0,1,2,3,4+}` accuracy；
- count MAE；
- null/single/multi confusion matrix；
- over-prediction / under-prediction rate；
- count ECE/Brier；
- 按 action、source、query template、GT count 分层的 count accuracy；
- active-mode 数量；
- 第 6.3 节严格定义的 `DuplicateRate@0.5`、`Selected-FullCoverage@0.5` 和 `Oracle-Mode-FullCoverage@0.5`。

### 13.4 Template-dependence 诊断

`build_phrase_manifest.py` 同时生成 `template_id`：在原 query 中把已对齐的 action/team span 分别替换为 `<ACTION>/<TEAM>`，再做 lowercase、标点和连续空格归一化。诊断固定为：

1. 报告 train/val/test 的 template 频数，以及主模型在 seen/unseen template 上的 count accuracy、MAE、null FPR 和 FullCoverage；
2. 训练 **Template-only probe**：输入仅为 `template_id + source` 的 one-hot，不读 query factors 或视频；probe 只在 train 拟合、val 选正则强度、test 只报告一次；
3. 训练 **Label-only probe**：输入仅为 action/team labels、source 和 template_id，不读视频；训练/选择/test 协议与 Template-only probe 相同；
4. 若 val 中没有足够 unseen templates，则在 train+val 上按 `template_id` 做 5-fold GroupKFold，只训练 AGC/AEC head，B0/P0 固定；
5. 报告 `main model - template-only probe` 和 `main model - label-only probe` 的增益，以及 seen-to-unseen drop。

如果 count contrastive 只提高 seen-template 或 label-only 可预测类别，而 unseen-template、视频置换敏感性和 FullCoverage 没有提高，则不认定为有效计数增益。

### 13.5 Phrase-label leakage 诊断

manifest 中的 action/team labels、token indices 和 GT spans 只能进入 criterion targets，不能进入 `model.forward()` 或 inference postprocessing。必须通过：

1. **Manifest invariance**：同一 eval batch 分别使用正确、置空、batch-permute 的 phrase metadata，预测 tensors 和 JSONL byte-identical；
2. **Forward signature**：inference dataloader 不加载 phrase manifest，模型仍能依靠 learned factor attention 完成前向；
3. **Gradient isolation**：V-EPR 输出对 `query_tokens/query_global` 的梯度严格为 0；
4. **Video permutation**：固定 query 并 batch-permute visual-only event features 后，V-EPR phrase retrieval 应降至 chance 附近；若不下降则存在文本路径泄漏；
5. **Oracle separation**：任何使用 manifest token indices 做 pooling 的 oracle 结果单列，不能进入 H1-H3/F1-F2 主表。

B0 的 legacy existence threshold 只用于复现原结果。AGC-Direct 和 HieA2M 主结果均使用 count posterior 的 argmax 作为唯一空集决策；只允许在 validation 拟合 `T_count` 和 4+ 的 `tau_mode`，test 不调 threshold。

## 14. 实施文件与测试

### 14.1 修改文件

- `models/flash_vtg_gmr/model.py`：候选接口、event adapter、Temporal HMSA、AEC；
- `models/flash_vtg_gmr/blocks/loss.py`：Hungarian set loss、HMSA、count、contrastive 和 consistency losses；
- `training/flash_vtg_gmr/dataset.py`：真实 attention mask、完整 GT windows、phrase manifest；
- `training/flash_vtg_gmr/config.py`：模块开关和 loss weights；
- `training/flash_vtg_gmr/inference.py`：完整 event modes、count posterior、existence score；
- `training/flash_vtg_gmr/postprocessing.py`：event-level selection，不复用 raw-proposal NMS；
- `eval/eval_main.py`：count、null/single/multi 和分层诊断。

### 14.2 新增文件

- `models/flash_vtg_gmr/event_adapter.py`；
- `models/flash_vtg_gmr/temporal_hmsa.py`；
- `models/flash_vtg_gmr/event_cardinality.py`；
- `training/flash_vtg_gmr/extract_lighthouse_features.py`；
- `training/flash_vtg_gmr/audit_features.py`；
- `training/flash_vtg_gmr/build_phrase_manifest.py`；
- `training/flash_vtg_gmr/calibrate_count.py`；
- `eval/eval_hiea2m_diagnostics.py`；
- `configs/flash_vtg_gmr/feature_canary_50.txt`；
- `scripts/run_hiea2m.sh`；
- `tests/test_feature_contract.py`；
- `tests/test_candidate_interface.py`；
- `tests/test_event_matching.py`；
- `tests/test_null_loss_gating.py`；
- `tests/test_cardinality.py`；
- `tests/test_event_set_metrics.py`；
- `tests/test_phrase_label_leakage.py`；
- `tests/test_inference_selection.py`；
- `tests/test_cli_contract.py`。

### 14.3 必须通过的测试

1. 关闭全部新模块时，旧 checkpoint 可 `strict=True` 加载，输出与 B0 完全一致；
2. train/eval 的 candidate feature、mask、span 和 score 索引一致；
3. 6/7-event 样本不被截断或 shuffle；
4. null query 的 boundary、phrase-positive 和 V-EPR loss 精确为 0；
5. SportsMoments 的全部 team-related 输出和梯度为 0；
6. Hungarian matching 不会把一个 GT 匹配给多个 modes；
7. Poisson-binomial 概率和为 1，并与枚举结果一致；
8. `pred_count=N` 时只从 event modes 选择 N 个结果；
9. duplicate/full-coverage toy cases 与第 6.3 节定义完全一致；
10. AEC 只有 `argmax P_AEC` 一个空集 hard decision，legacy existence head 在新方法中关闭；
11. eval forward 不读取 phrase labels、token indices、GT spans 或 test-derived calibration；
12. 正确/置空/permuted manifest 下 inference 输出 byte-identical；
13. public adapter 参数不在 H/C/F optimizer 中，运行时 SHA256 与声明一致；
14. prediction JSONL 能从保存的 `P_AEC` 和 modes 确定性重放最终集合；
15. F-old loader 输出精确等于 `[L2Norm(SlowFast) || L2Norm(CLIP)]`，颠倒目录顺序必须触发 manifest mismatch；
16. Lighthouse combined output 拆分再拼接后逐元素一致，canary 阈值与原子写入/断点续跑都有测试；
17. P0-selection 的 loss dict 不含 span regression，seed span 无梯度；P0-R 的 `delta_m` 能收到非零 span 梯度且 `rho==0.5`；
18. `L_token-temporal` 只读取 full-query token group，action/team 不读取全视频 `B-`；
19. C1/C2/F1/F2 registry 固定 AEC-CE，传入 `--aec_type` 或其他 count override 必须失败；
20. `P_PB^5` 概率和为 1，并与 AEC-CE 共享完全相同的五类 weights；
21. G0 的 loss 不含 `L_count-con`，G0-Con 才包含，除此之外两者配置完全一致。

### 14.4 CLI execution contract

保留现有入口：

```text
python -m training.flash_vtg_gmr.train CONFIG [ARGS]
python -m training.flash_vtg_gmr.inference CONFIG [ARGS]
```

新增统一 `--variant`：

```text
{B0,G0,G0-Con,P0,P0-R,H1,H2,H3,C1,C2,F1,F2,R1,
 C-PB,C-PB-Con,C-PB-Exact,C-Exact,F-PB,F2-unfreeze}
```

variant registry 唯一决定启用模块；不允许同时用多个独立 boolean flags 拼出未登记模型，也不提供能改变 C1/C2/F1/F2 count head 类型的 `--aec_type`。新增参数契约为：

| 参数 | 训练 | 推理 | 约束 |
|---|---|---|---|
| `--feature_manifest PATH` | 必需 | 必需 | 校验 corpus/order/dim/hash |
| `--phrase_manifest_dir DIR` | H/F/R 训练必需 | 禁止 | 按 split 读取，只能进入 criterion targets |
| `--init_backbone_ckpt PATH` | G0/G0-Con/P0/P0-R 必需 | 禁止 | 只初始化 B0 prefixes |
| `--adapter_ckpt PATH` | H/C/F/R 必需 | checkpoint 内置 | 必须匹配 seed 和 SHA256 |
| `--freeze_adapter` | H/C/F/R 主实验必需 | 不适用 | `F2-unfreeze` 除外 |
| `--count_calibration PATH` | 禁止 | 所有 count variants 必需 | 只能由 val 生成 |
| `--max_windows -1` | 必需 | 必需 | `-1` 表示保留全部 GT |

`--resume` 只表示**同构模型**的严格续训或推理，必须 `strict=True`；不得再承担“从 B0 初始化新结构”的职责。现有 `--resume_adapter` 废弃。partial initialization 必须只允许 registry 声明的 prefixes，unexpected/missing keys 列表写入日志并在超出白名单时失败。

标准执行顺序为：

```bash
# 1. 为现有 F-old corpus 生成冻结 manifest
python -m training.flash_vtg_gmr.audit_features \
  --mode existing \
  --slowfast_dir data/Soccer-GMR/feature/standard/slowfast \
  --clip_dir data/Soccer-GMR/feature/standard/clip \
  --text_dir data/Soccer-GMR/feature/standard/clip_text \
  --split_jsonl data/label/Standard/train.jsonl \
                data/label/Standard/val.jsonl \
                data/label/Standard/test.jsonl \
  --concat_order slowfast,clip \
  --output artifacts/features/f-old/feature_manifest.json

# 2. 用原视频运行 F-new canary；通过后才允许 full extraction
python -m training.flash_vtg_gmr.extract_lighthouse_features \
  --mode canary \
  --lighthouse_root /tmp/lighthouse-audit \
  --video_root data/Soccer-GMR/raw/standard \
  --query_jsonl data/label/Standard/train.jsonl \
  --canary_list configs/flash_vtg_gmr/feature_canary_50.txt \
  --slowfast_weight "$SLOWFAST_WEIGHT" \
  --output_root artifacts/features/f-new-canary

# 3. 生成训练监督；每个 split 单独输出并记录 hash
python -m training.flash_vtg_gmr.build_phrase_manifest \
  --input_jsonl data/label/Standard/train.jsonl \
  --split_name train \
  --text_dir data/Soccer-GMR/feature/standard/clip_text \
  --output_dir artifacts/manifests/standard
# 对 val 重复执行并写入同一目录；test manifest 仅供离线诊断，不传给 inference

# 4. 每个 seed 从同一 B0 分别训练 G0 和一次性公共 P0
bash scripts/run_hiea2m.sh train \
  --variant G0 --seed 2024 \
  --feature_manifest artifacts/features/f-old/feature_manifest.json \
  --init_backbone_ckpt "$B0_CKPT"
# G0-Con 使用相同命令，仅把 --variant 改为 G0-Con

bash scripts/run_hiea2m.sh train \
  --variant P0 --seed 2024 \
  --feature_manifest artifacts/features/f-old/feature_manifest.json \
  --init_backbone_ckpt "$B0_CKPT"

# 5. 下游变体加载并冻结同一个 P0；例：H3/F2
bash scripts/run_hiea2m.sh train \
  --variant H3 --seed 2024 \
  --feature_manifest artifacts/features/f-old/feature_manifest.json \
  --phrase_manifest_dir artifacts/manifests/standard \
  --adapter_ckpt "$P0_CKPT" --freeze_adapter

bash scripts/run_hiea2m.sh train \
  --variant F2 --seed 2024 \
  --feature_manifest artifacts/features/f-old/feature_manifest.json \
  --phrase_manifest_dir artifacts/manifests/standard \
  --adapter_ckpt "$P0_CKPT" --freeze_adapter

# 6. 只在 val 拟合 count temperature/tau_mode
python -m training.flash_vtg_gmr.calibrate_count \
  --checkpoint "$F2_CKPT" \
  --split_jsonl data/label/Standard/val.jsonl \
  --output "$F2_RUN/calibration.json"

# 7. test inference 不接收 phrase manifest
bash scripts/run_hiea2m.sh infer \
  --variant F2 \
  --checkpoint "$F2_CKPT" \
  --feature_manifest artifacts/features/f-old/feature_manifest.json \
  --count_calibration "$F2_RUN/calibration.json" \
  --split_jsonl data/label/Standard/test.jsonl
```

`scripts/run_hiea2m.sh` 必须展开为当前 Python module 入口，并固定公共数据参数：`--dset_name hl`、`--ctx_mode video_tef`、`--v_feat_dirs slowfast clip`、`--v_feat_dim 2816`、`--t_feat_dim 512`、`--clip_length 2`、`--max_windows -1`。P0/H/C/F/R 强制 `--nms_thd -1` 且 `use_exist_head=False`；只有 B0/G0/G0-Con 使用 legacy post-NMS。

每次运行必须产出：

```text
opt.json, code.zip, command.txt, environment.txt
feature_manifest.json + sha256
phrase_manifests.sha256         # 仅训练
parent_checkpoints.json         # path + sha256
calibration.json                # 推理计数模型
predictions_raw.jsonl           # 全部 modes/posterior
predictions_selected.jsonl      # 唯一 count rule 后的集合
metrics.json, diagnostics.json
```

inference 从 checkpoint 的 `opt.json` 恢复全部架构参数；CLI 只能覆盖 device、split path、feature physical path、output path 和 calibration path。任何 architecture override、feature hash/order 不匹配、P0 seed/hash 不匹配都必须 fail fast。

## 15. 风险与决策规则

| 风险 | 检查 | 决策 |
|---|---|---|
| F-new 无法复现旧特征 | 50-video canary 与 manifest | 主实验锁定 F-old；F-new 全量独立重跑 |
| 特征拼接顺序/normalize 改变 | loader tensor unit test + hash | fail fast，不加载 checkpoint |
| Selection-only span loss 无梯度 | loss/gradient unit test | P0 只保留 event+quality；span loss 仅 P0-R |
| Adapter 丢失原 proposal recall | 比较 raw proposal 与 mode oracle recall | 先修 adapter，不继续归因给 HMSA/AEC |
| 公共 Adapter 被下游更新 | optimizer/gradient/SHA256 audit | 该 run 不进入归因矩阵 |
| Count 学习 action/template 先验 | 分层 count accuracy、打乱模板诊断 | 只提升总体 accuracy 不进入主模型 |
| HMSA 只是辅助 loss | 比较 zero-init residual 开/关及梯度 | 必须证明 aligned event 改变最终预测 |
| Phrase negatives 含假负例 | 审计多标签和 overlapping events | unknown/ambiguous 从 denominator 移除 |
| Action/team temporal 假负样本 | 检查 `B-` 的计算图 | 主 MIL 只允许 full-query tokens |
| Query-video batch 假负样本 | relation-mask audit | 未经标注明确否定的跨视频 pair 一律 mask |
| Phrase metadata 进入推理 | manifest invariance + forward signature | 任一失败视为 label leakage |
| V-EPR 文本泄漏 | 检查输入计算图 | 任何 query-conditioned feature 都禁用 |
| 高 count 类过稀疏 | CE 与 PB、4+ 与 exact 对照 | 主结果保留稳定的 `{0,1,2,3,4+}` |
| 主 AEC 编号发生漂移 | variant registry/CLI test | C1/C2/F1/F2 永远锁定 AEC-CE |
| count/existence 双重门控 | inference-selection unit test | 新方法只允许 `argmax P_AEC` |
| calibration 过拟合 test | 保存 calibration provenance | test 上拟合 `T_count/tau_mode` 的结果无效 |

## 16. 最终论文主线

论文只提出两个核心模块：

1. **Temporal HMSA** 通过 token-clip/event、phrase-event 和 query-video 对齐，生成直接用于定位与计数的 aligned event representations；
2. **AEC** 在 event modes 上联合预测 null/single/multi cardinality，并用 count contrastive learning 塑造集合级数量表示。

Proposal-to-Event Adapter 是让密集 proposals 变成可计数事件的必要工程适配；V-EPR 和分阶段多任务训练是辅助实验。

最终表述为：

> **HieA2M 将 HieA2G 的层次化语义对齐与自适应目标计数，从空间对象集合迁移到时序事件集合。**

主要参考：[HieA2G](https://ojs.aaai.org/index.php/AAAI/article/view/32867)、[GMR repository](https://github.com/dymm9977/generalized-moment-retrieval)、[FlashVTG](https://openaccess.thecvf.com/content/WACV2025/papers/Cao_FlashVTG_Feature_Layering_and_Adaptive_Score_Handling_Network_for_Video_WACV_2025_paper.pdf)、[Lighthouse](https://github.com/line/lighthouse)、[HERO feature extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor)。
