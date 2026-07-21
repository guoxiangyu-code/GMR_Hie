# Part 2：Proposal-to-Event Adapter 与事件基数预测

## 1. 任务目标与边界

本任务实现并验证：

1. Proposal-to-Event Adapter：把 FlashVTG dense proposals 转成 `M=10` 个可计数 event modes；
2. AGC-Direct：用 Threshold -> Classifier -> Contrastive 的忠实链验证 HieA2G AGC 直接迁移到 raw proposals 是否足够；
3. Adaptive Event Cardinality（AEC）：在 event modes 上预测 `{0,1,2,3,4+}`；
4. 唯一空集决策、event-level selection、duplicate/full-coverage 指标；
5. 固定公共 P0 Adapter，供 Part 3 的 HMSA/联合模型使用。

本阶段**不实现** Token-Clip、Phrase-Event、Query-Video alignment 或 V-EPR，不读取 phrase manifest。基础 AEC 必须在没有 Temporal HMSA 的情况下独立训练和推理。

## 2. 前置产物与 Fail-Fast

只接受 Part 1 的以下产物：

```text
artifacts/features/f-lighthouse/feature_manifest.json
artifacts/manifests/standard/manifest_index.json
artifacts/baselines/baseline_index.json
artifacts/baselines/{2024,2025}/model_best.ckpt
```

启动时必须校验：

- feature manifest schema/hash；
- `setting="f-lighthouse"`、`known_provenance=true`、`encoder_mode="eval"`；
- `text_token_alignment_status="verified"`；
- `concat_order=[slowfast,clip]`；
- `per_stream_normalization=true`；
- `clip_length=2`；
- B0 seed、checkpoint hash 和 feature hash；
- canonical manifest hash 与 baseline index 一致；
- dataset 已启用 `max_windows=-1`、真实 text mask 和 null queries。

任一检查失败则停止。不得在本任务中重提特征、重训另一个 B0 或修改 phrase supervision。

## 3. 实验编号锁定

```text
G0-Threshold = raw proposals + validation-only single threshold，无 count head
G0      = AGC-Direct CE
G0-Con  = AGC-Direct CE + count contrastive
P0      = P0-selection，公共 Adapter
P0-R    = P0-residual，仅 Adapter 消融
C1      = P0 + AEC-CE
C2      = P0 + AEC-CE + count contrastive
```

内部消融：

```text
C1-Enhanced  = P0 + event max/expected count + consistency
C-PB        = P0 + AEC-PB-bin
C-PB-Con    = P0 + AEC-PB-bin + count contrastive
C-PB-Exact  = P0 + AEC-PB-bin + 0.1 AEC-PB-exact
C-Exact     = P0 + exact {0,...,7} classifier
```

C1/C2 永远使用 AEC-CE。CLI 不提供能把它们切换成 PB/Exact 的参数。

### 3.1 必跑与非阻塞实验

Part 2 的完成只强制以下主实验，且每个实验都运行 seeds `2024/2025`：

```text
G0-Threshold, G0, G0-Con, P0, C1, C2
```

以下均为非阻塞内部消融，不得因为尚未运行而阻止 Part 2 交接：

```text
P0-R, C1-Enhanced, C-PB, C-PB-Con, C-PB-Exact, C-Exact
```

非阻塞表示可以在 Part 3 开始后补跑，但其实现若已合入，仍必须通过相关单元测试。主表、模型选择和 Part 3 初始化均不得依赖这些可选结果。

## 4. FlashVTG Candidate Contract

### 4.1 需要暴露的张量

在 `models/flash_vtg_gmr/model.py` 中增加 opt-in candidate outputs：

```text
candidate_feat   : B x K x 256
candidate_mask   : B x K
candidate_span   : B x K x 2      # [start,end]，归一化 [0,1]
candidate_logit  : B x K          # sigmoid 前
candidate_topk_idx : B x K
candidate_point  : B x K x 4
candidate_scale  : B x K
query_global     : B x 256
```

固定：

```text
K=50
M=10 event modes
```

要求：

1. span 必须在任何 score sort 前解码；
2. top-K 后的 feature/mask/span/logit/point/scale 必须用 `candidate_topk_idx` 从同一 flatten 维度 gather，batch 间不得共享 top-K index；
3. 开启 Adapter 时 `ConvPyramid` 在 train/eval 都返回 mask；
4. padding candidates 不参与 seed selection、attention 或 loss；
5. 关闭新模块时不构造新增参数，B0 checkpoint 可 `strict=True` 加载且输出不变。

### 4.2 Criterion 顺序

当前 `SetCriterion.forward()` 会先过滤 positive queries。以下损失必须在该过滤之前计算：

```text
L_event
L_quality
L_count
L_count-con
L_consistency
```

legacy FlashVTG localization loss 仍只处理 positive queries。

## 5. Proposal-to-Event Adapter

### 5.1 Relation features 与 seeds

对每个 candidate 构造可训练 relation feature：

```text
r_i = RelationEncoder([
    candidate_feat_i,
    normalized_span_i,
    candidate_scale_i,
    candidate_logit_i
])
```

`RelationEncoder` 不能只服务离散 seed selection。它必须同时进入 mode 初始化和 decoder proposal memory，使 `L_event/L_quality` 能对其产生梯度：

```text
e_m^0 = W_seed r_seed_m
      + learned_slot_embedding_m
      + W_q query_global

decoder_proposal_memory = {r_i}
```

从 K=50 中 greedy 选择最多 M=10 seeds。为避免可训练 relation feature 导致 seed index 在训练中跳变，离散 diversity 只使用 detached B0 量：

```text
z_i = stop_gradient(L2Norm(candidate_feat_i))
s_i = stop_gradient(normalized_candidate_score_i)

seed_1 = argmax_i s_i

seed_score_i = s_i
             + lambda_div * min_j(1 - cosine(z_i, z_seed_j))

lambda_div = 0.5
```

`lambda_div` 固定为 0.5，不按 seed/下游模型调参。禁止 hard tIoU suppression，因为同一 query 的不同 GT 可能达到 tIoU 0.857。

两层 decoder：

1. mode self-attention；
2. mode-to-all-proposals cross-attention；
3. FFN + LayerNorm。

输出：

```text
event_feat_m
event_logit_m
quality_logit_m
event_span_m
event_mask_m
```

若样本有 `K_valid < M`，只生成 `K_valid` 个真实 seeds，其余 slots padding；定义：

```text
event_mask[b,m] = True  iff slot m has a real valid seed
```

mask 方向固定为 `True=valid`。mode self-attention 同时 mask 无效 query/key，proposal cross-attention 用 `event_mask` mask query、用 `candidate_mask` mask memory；无效 slot 输出清零。Hungarian、event/quality loss、集合统计和 selection 均只能访问 `event_mask=True` 的 modes。

### 5.2 P0-selection：公共主版本

```text
event_span_m = stop_gradient(seed_span_m)
```

seed selection 离散，seed span 来自冻结 B0 candidates。Hungarian matching 可以使用 span L1/tIoU，但 span loss 不进入训练总损失：

```text
L_adapter_selection = L_event + L_quality
```

主实验中的 `P0` 只指 P0-selection。Part 3 的 H/C/F 模型也保持 seed spans，不得重新启用 boundary regression。

### 5.3 P0-R：Residual 消融

```text
rho = 0.5
event_span_m = stop_gradient(seed_span_m)
             + tanh(delta_m) * rho * stop_gradient(seed_duration_m)
```

`rho` 固定，不搜索。residual head 最后一层 weights/biases 零初始化。

```text
L_adapter_residual = L_event
                   + 5 * L_span-L1
                   + 2 * L_span-tIoU
                   + L_quality
```

P0-R 只进入 Adapter 内部消融，不能替换下游公共 P0。若以后决定使用 P0-R，必须完整重跑所有 H/C/F，不能混用。

### 5.4 Hungarian matching

使用 `scipy.optimize.linear_sum_assignment` 对 detached cost 做一对一 matching：

```text
C(m,j) = 2 * L1(event_span_m, gt_j)
       + 2 * (1 - tIoU(event_span_m, gt_j))
       - sigmoid(event_logit_m)
```

cost matrix 只包含 `event_mask=True` 的 rows；padding modes 不得进入 Hungarian 后再靠 loss weight 清零。

监督：

- matched mode：event target=1；
- unmatched mode：event target=0；
- null query：所有 `event_mask=True` modes 的 event target 为 0，padding modes 不进入 loss；
- P0-selection 不计算 boundary regression；
- P0-R 只对 matched modes 计算 boundary regression。

`L_event` 使用 focal loss。quality target 为：

```text
quality_target_m = stop_gradient(max_j tIoU(event_span_m, gt_j))
L_quality = SmoothL1(sigmoid(quality_logit_m), quality_target_m)
```

null query 的 quality target 为 0。mode 排序分数固定：

```text
mode_score_m = sigmoid(event_logit_m) * sigmoid(quality_logit_m)
```

不引入可调 exponent。

### 5.5 P0 推理

P0 没有 count head。最终集合固定为：

```text
selected_modes = {m | sigmoid(event_logit_m) >= 0.5}
```

集合还必须满足 `event_mask_m=True`。阈值固定 0.5，不按 seed 或下游模型调整。event modes 不再执行普通 NMS。

### 5.6 `EventInterfaceV1`：Part 2 到 Part 3 的唯一接口

冻结 B0 和同 seed public P0 在每次 forward 内部生成：

```text
EventInterfaceV1:
    event_feat              B x M x 256
    event_span              B x M x 2
    adapter_event_logit     B x M
    adapter_quality_logit   B x M
    event_mask              B x M       # bool, True=valid
    query_global            B x 256

    schema_version          = "EventInterfaceV1"
    M                       = 10
    feature_dim             = 256
    span_format             = "normalized_start_end"
    mask_direction          = "true_is_valid"
    baseline_checkpoint_sha256
    public_p0_checkpoint_sha256
    feature_manifest_sha256
```

这是模型内部的强类型返回对象，不是 dataset 输入或一套新的预计算特征。正式 Part 3 forward 必须保持：

```text
video/query inputs -> frozen B0 -> candidate tensors
                   -> frozen public P0 -> EventInterfaceV1
                   -> Temporal HMSA
```

禁止主方案让 dataloader 加载预计算 candidate/event tensors。允许为离线诊断缓存 `EventInterfaceV1`，但缓存必须绑定上述三个 hashes，且不能用于正式训练、validation、test inference 或 checkpoint 选择。Part 3 启动时逐项验证 schema/shape/mask/hash；HMSA gates 全零时必须严格退化为该接口中的 P0 输出。

## 6. Event Set Metrics：Duplicate、Full Coverage 与 SetSuccess

对 query `q` 的 prediction set `P_q` 和 GT set `G_q`，在：

```text
E_theta = {(p,g) | tIoU(p,g) >= theta}
```

上求最大基数、再最大总 tIoU 的一对一 matching `M_theta(q)`。

```text
P_eligible(q) = {p in P_q | exists g in G_q: tIoU(p,g)>=theta}

DuplicateCount_theta(q)
  = |P_eligible(q)| - |M_theta(q)|

DuplicateRate_theta
  = sum_q DuplicateCount_theta(q)
    / max(sum_q |P_eligible(q)|, 1)       # positive queries 上 micro

FullCoverage_theta(q)
  = 1[|M_theta(q)| = |G_q|]              # 只对 |G_q|>=2

FullCoverage_theta
  = mean_{q:|G_q|>=2} FullCoverage_theta(q)
```

严格集合成功指标定义为：

```text
SetSuccess_theta(q) = 1
iff |P_q| = |G_q| = |M_theta(q)|
```

null query 采用同一定义：`P_q` 和 `G_q` 都为空时为 1，否则为 0。主指标固定 `SetSuccess@0.5`，在全部 queries 上做 macro mean。该指标同时惩罚 count error、漏检、多检、duplicate 和低 tIoU，不用 score threshold 之外的补救规则。

规则：

- false positive 不算 duplicate；
- 两个预测若分别匹配两个高度重叠 GT，不算 duplicate；
- 主阈值 `theta=0.5`；补充 `0.3/0.7`；
- `Selected-FullCoverage` 在最终输出计算；
- `Oracle-Mode-FullCoverage` 在全部 10 modes 计算，仅表示 adapter 上限。
- `Raw-Proposal-Oracle-FullCoverage` 使用全部 K 个 valid raw proposals，并按同一 matching 和阈值计算；它只作为 Adapter 压缩前的 coverage 上限，不作为模型输出。

## 7. 固定公共 P0 协议

对 seeds `{2024,2025}`：

1. 从对应 seed 的正式 B0 初始化；
2. 冻结全部 B0 参数；
3. 只训练一次 P0-selection；
4. checkpoint 只用 validation 的 AdapterScore 选择；
5. 下游 C1/C2 和 Part 3 H/F/R 加载同一 seed 的 P0；
6. 下游 optimizer 不得包含 `backbone.*` 或 `event_adapter.*`；
7. P0 hash 由 downstream run 启动时校验。

```text
AdapterScore = HarmonicMean(
    Selected-FullCoverage@0.5,
    1 - DuplicateRate@0.5
)
```

`Oracle-Mode-FullCoverage@0.5` 只作为第 15 节的硬性 coverage-retention 约束，不进入 checkpoint score。

约束 validation mAP 相对 B0 下降不超过 0.5。

P0 checkpoint 是完整 checkpoint，包含：

```text
B0 weights + hash
Adapter weights
seed
feature manifest hash
training command/opt
P0 predictions/metrics hashes
EventInterfaceV1 schema metadata
```

输出索引：

```text
artifacts/adapters/public_adapter_index.json
```

## 8. AGC-Direct Baselines

AGC-Direct 不使用 Adapter，并保持 B0 冻结。先建立无 count classifier 的直接阈值对照：

```text
G0-Threshold:
    raw_score_i = sigmoid(candidate_logit_i)
    output = all valid raw proposals with raw_score_i >= tau_raw
```

`tau_raw` 是整个 validation split 上选择的单一阈值，以 `SetSuccess@0.5` 最大为准，平分时选 positive-query mAP 更高者；每个 seed 只拟合一次并冻结到 test。它不使用 legacy existence gate、动态 count、额外 NMS 或按 query 调阈值。预测 count 直接等于输出集合大小，再映射到 `{0,1,2,3,4+}` 计算 `Count-Acc-5`。

忠实 AGC classifier 使用 HieA2G 的 average-pool text/object 结构：

```text
text_mean_raw     = MaskedMean(query_tokens, query_mask)
proposal_mean_raw = MaskedMean(candidate_feat, candidate_mask)

g_raw = CountHeadV1.encode(text_mean_raw, proposal_mean_raw)
P_AGC = softmax(CountHeadV1.classifier(g_raw) / T_count)
```

`CountHeadV1` 是 Part 2/3 共享的唯一主计数结构：

```text
t = LayerNorm(GELU(Linear(512,256)(text_mean)))
s = LayerNorm(GELU(Linear(256,256)(set_mean)))
g = LayerNorm(GELU(Linear(512,256)(concat(t,s))))
count_logits = Linear(256,5)(g)
```

主结构不读取 `query_global`、video-memory max、set max、proposal/event logits 或 expected count。dropout 固定为 0，确保 G0/C1 和 C1/F1 可以使用逐元素相同的初始化。

五类标签固定 `{0,1,2,3,4+}`，class weights 由 train 的 effective-number weighting 得到并裁剪到 `[0.5,2.0]`。

```text
L_G0     = WeightedCE(P_AGC, count_class)
L_G0_con = L_G0 + 0.1 * L_count-con
```

G0-Con 的 contrastive embedding 固定为 `L2Norm(Projection(g_raw))`；不得改用 video max/query_global 等 G0 classifier 未使用的特征。其 temperature 和 detached class-balanced queue 与 C2 完全同构。

推理：

- `argmax P_AGC==0`：空集；
- count 1/2/3：从 valid raw proposals 按 `raw_score` 取 Top-N；
- count 4+：复用 G0-Threshold 冻结的 `tau_raw`，至少 Top-4、最多 10；
- 关闭 B0 existence hard gate；
- 不增加其他 empty-set gate，不执行额外 NMS。

G0 与 G0-Con 除 contrastive loss 外必须完全一致。

## 9. Adaptive Event Cardinality

### 9.1 Part 2 的同构 event 输入

Part 2 AEC 不依赖 `c_global` 或 `query_video_logit`。C1/C2 与 G0/G0-Con 使用同一个 `CountHeadV1`，唯一结构性替换是 raw proposals -> public P0 event modes：

```text
text_mean_event = MaskedMean(query_tokens, query_mask)
event_mean      = MaskedMean(event_feat, event_mask)

g_event = CountHeadV1.encode(text_mean_event, event_mean)
P_CE = softmax(CountHeadV1.classifier(g_event) / T_count)
```

若整行 `event_mask` 为空，`MaskedMean` 返回确定的全零 set vector 并记录审计计数，不能产生 NaN。G0 与 C1 的 `CountHeadV1` 参数使用相同 module-init key `(seed, "CountHeadV1")`，并通过 isolated RNG fork/local generator 初始化，因此前面创建了多少其他模块不能改变其参数；训练开始前必须逐元素一致。它们的 batch order、optimizer、学习率、class weights 和 checkpoint 规则也相同。

### 9.2 主模型 AEC-CE

```text
count_class = {0,1,2,3,4+}
L_count = -w[bin(n_gt)] * log P_CE[bin(n_gt)]
```

主编号锁定：

```text
C1 = AEC-CE
C2 = AEC-CE + 0.1 * L_count-con
```

### 9.3 非阻塞 `C1-Enhanced`

以下增强不进入 C1/C2：

```text
p_m            = where(event_mask, sigmoid(adapter_event_logit_m), 0)
event_max      = MaskedMax(event_feat, event_mask)
expected_count = sum_m p_m
```

`C1-Enhanced` 使用独立增强 head `[text_mean,event_mean,event_max,expected_count]`，并可加入 event-count consistency：

由 slot activity 做 differentiable Poisson-binomial DP：

```text
P_PB(N=0..M) = PoissonBinomial(p_1,...,p_M)
P_PB^5 = [P(0),P(1),P(2),P(3),sum_{n=4..M}P(n)]

L_consistency = 0.5 * KL(P_CE || P_PB^5)
              + 0.5 * KL(P_PB^5 || P_CE)
```

invalid modes 的 Bernoulli probability 固定为 0，不能进入 DP。`C1-Enhanced` 使用 `0.1 * L_consistency`；主 C1/C2 不使用该项，从而保证 `G0 -> C1` 只改变 raw proposals -> event modes。

### 9.4 Count contrastive

```text
g_con = L2Norm(Projection(g_event))

L_count-con(i) = -1/|Pos(i)| * sum_{p in Pos(i)}
    log exp(sim(g_i,g_p)/tau)
        / sum_{a!=i} exp(sim(g_i,g_a)/tau)
```

标签固定 `{0,1,2,3,4+}`。使用 class-balanced memory queue；queue 只保存 detached embeddings/labels，不跨 feature setting 或 seed。

C1 不含该 loss，C2 权重固定 0.1。

### 9.5 PB/Exact 消融

PB 使用相同五类空间和相同 weights：

公共 P0 冻结，因此 PB 不能直接把 frozen `adapter_event_logit` 当作唯一输入。定义可训练 activity residual：

```text
delta_activity_m = ActivityResidual(event_feat_m, text_mean_event)
aec_event_logit_m = stop_gradient(adapter_event_logit_m) + delta_activity_m
p_m = where(event_mask, sigmoid(aec_event_logit_m), 0)
```

`ActivityResidual` 最后一层 weight/bias 零初始化，只由 PB/count loss 更新，且不得更新 public P0。随后由有效 modes 计算 `P_PB(N=0..M)` 和五类合并分布：

```text
y_5 = bin(n_gt)
L_PB-bin = -w[y_5] * log P_PB^5[y_5]
```

可选 exact 辅助不使用稀有类别权重：

```text
L_PB-exact = -log(P_PB(n_gt) + 1e-8)
L_C-PB-Exact = L_PB-bin + 0.1 * L_PB-exact
```

`C-Exact` 是独立 `{0,...,7}` classifier，只作为数据集上界消融，不改变主实验。

### 9.6 唯一空集与 selection

所有 AEC variants 的唯一空集硬决策：

```text
pred_count = argmax P_count
output = empty set iff pred_count == 0
```

`p_nonempty=1-P_count[0]` 只用于 AUROC、Rej-F1、ECE/Brier 和排序诊断，不能再次清空输出。

非空规则：

- count 1/2/3：按 `mode_score` 取 Top-N；
- count 4+：保留 `sigmoid(event_logit)>=tau_mode`，至少 Top-4、最多 10；
- `C-Exact` 的 count 4..7：直接取 Top-N；
- Top-N 和 threshold 只能作用在 `event_mask=True` 的 event modes；
- 不再执行 NMS。

`T_count` 和 `tau_mode` 只在 validation 拟合。正温度不改变 argmax；test 不调参数。

## 10. No-Target Loss Gating

| Loss | Positive query | Null query |
|---|:---:|:---:|
| B0 localization | 是 | 否 |
| Adapter event focal | matched/unmatched | 所有 modes 为 no-event |
| Adapter quality | continuous IoU | target=0 |
| P0-selection span regression | 否 | 否 |
| P0-R span regression | matched | 否 |
| Count CE/PB | `count>0` | `count=0` |
| Event-count consistency（C1-Enhanced only） | 是 | 是 |
| Count contrastive | 是 | 是 |

新 losses 必须在 criterion 的 positive-only legacy localization filter 之前计算。

## 11. 训练与模型选择

### 11.1 顺序

每个 seed 独立执行：

1. 从 Part 1 B0 的 validation predictions 拟合一次 G0-Threshold `tau_raw`；
2. 从 Part 1 B0 训练 G0；
3. 从同一 B0 和相同 `CountHeadV1` 初始化训练 G0-Con；
4. 从同一 B0 训练一次 P0-selection；
5. 冻结 B0+P0，从相同 `CountHeadV1` 初始化训练 C1；
6. 冻结相同 B0+P0 和相同初始化训练 C2；
7. 可选训练 P0-R/C1-Enhanced/PB/Exact，不用于下游公共接口。

推荐：

```text
new-module lr=3e-5
weight_decay=1e-4
epochs=10..15
early stopping=validation only
grad_clip=0.1
```

所有主结果运行 seeds `2024/2025`。checkpoint 选择不读取 test。该集合与第 3.1 节及 Part 1 已锁定 B0 seeds 一致。

G0/G0-Con/C1/C2 的 checkpoint 统一按 validation `SetSuccess@0.5` 最大选择；平分时依次使用 positive-query mAP、`Count-Acc-5`。P0 仍只使用第 7 节 AdapterScore 和 oracle coverage 约束。

### 11.2 关键比较

必报主线比较：

```text
G0-Threshold -> G0 : 原始 AGC count classifier
G0 -> G0-Con   : 原始 AGC count contrastive
B0 -> P0       : proposal-to-event conversion
G0 -> C1       : event modes 对 count/set prediction 的作用
C1 -> C2       : event-level count contrastive
```

可选 Adapter 消融：

```text
P0 -> P0-R     : boundary residual 是否必要
```

## 12. CLI Execution Contract

本阶段完成门槛及当前 CLI 支持 variants：

```text
{G0-Threshold,G0,G0-Con,P0,P0-R,C1,C2}
```

`C1-Enhanced/C-PB/C-PB-Con/C-PB-Exact/C-Exact` 是非阻塞的保留消融设计，
未实现前不得出现在 CLI choices，也不计入 Part 2 完成门槛；后续若实现，必须以新
variant 注册并补齐同等级 contract tests，不能复用或覆盖 C1/C2。

参数：

| 参数 | 约束 |
|---|---|
| `--feature_manifest` | 必需，必须匹配 B0/P0 hash |
| `--data_manifest_index` | 必需，train/val/test 只能由 canonical index 解析 |
| `--baseline_index` | 必需，必须匹配 seed/checkpoint/feature/data hashes |
| `--init_backbone_ckpt` | G0/G0-Con/P0/P0-R 必需 |
| `--adapter_ckpt` | C variants 必需，必须是同 seed public P0 |
| `--freeze_adapter` | C variants 必需 |
| `--count_calibration` | count inference 必需，只能来自 val |
| `--max_windows -1` | train/eval 必需 |

`--resume` 只用于同构 checkpoint 严格续训/推理。不得使用旧 `--resume_adapter`；partial init 必须按 registry 白名单加载并记录 missing/unexpected keys。

训练：

```bash
for SEED in 2024 2025; do
  B0_CKPT=$(python -m training.flash_vtg_gmr.resolve_artifact \
    --index artifacts/baselines/baseline_index.json --seed "${SEED}")

  bash scripts/run_hiea2m.sh calibrate-threshold \
    --variant G0-Threshold \
    --seed "${SEED}" \
    --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
    --data_manifest_index artifacts/manifests/standard/manifest_index.json \
    --baseline_index artifacts/baselines/baseline_index.json \
    --init_backbone_ckpt "${B0_CKPT}" \
    --split val

  for VARIANT in G0 G0-Con P0; do
    bash scripts/run_hiea2m.sh train \
      --variant "${VARIANT}" \
      --seed "${SEED}" \
      --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
      --data_manifest_index artifacts/manifests/standard/manifest_index.json \
      --baseline_index artifacts/baselines/baseline_index.json \
      --init_backbone_ckpt "${B0_CKPT}"
  done

  P0_CKPT=$(python -m training.flash_vtg_gmr.resolve_artifact \
    --index artifacts/adapters/public_adapter_index.json --seed "${SEED}")

  for VARIANT in C1 C2; do
    bash scripts/run_hiea2m.sh train \
      --variant "${VARIANT}" \
      --seed "${SEED}" \
      --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
      --data_manifest_index artifacts/manifests/standard/manifest_index.json \
      --baseline_index artifacts/baselines/baseline_index.json \
      --adapter_ckpt "${P0_CKPT}" \
      --freeze_adapter
  done
done
```

`P0-R`、C1-Enhanced 和 PB/Exact 使用相同 CLI 单独启动；它们不属于上述必跑闭环。G0/G0-Con 的 count=4+ inference 必须读取同 seed G0-Threshold calibration hash。

Calibration/inference：

```bash
python -m training.flash_vtg_gmr.calibrate_count \
  --checkpoint "$C2_CKPT" \
  --data_manifest_index artifacts/manifests/standard/manifest_index.json \
  --split val \
  --output "$C2_RUN/calibration.json"

bash scripts/run_hiea2m.sh infer \
  --variant C2 \
  --checkpoint "$C2_CKPT" \
  --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
  --data_manifest_index artifacts/manifests/standard/manifest_index.json \
  --baseline_index artifacts/baselines/baseline_index.json \
  --count_calibration "$C2_RUN/calibration.json" \
  --split test
```

inference 从 checkpoint `opt.json` 恢复架构。CLI 只能覆盖 device、split/feature physical path、output 和 calibration path；架构/hash/seed mismatch 必须失败。

## 13. 实现文件

修改：

- `models/flash_vtg_gmr/model.py`；
- `models/flash_vtg_gmr/blocks/loss.py`；
- `training/flash_vtg_gmr/config.py`；
- `training/flash_vtg_gmr/train.py`；
- `training/flash_vtg_gmr/inference.py`；
- `training/flash_vtg_gmr/postprocessing.py`；
- `eval/eval_main.py`；
- `scripts/run_hiea2m.sh`。

新增：

- `models/flash_vtg_gmr/event_interface.py`；
- `models/flash_vtg_gmr/event_adapter.py`；
- `models/flash_vtg_gmr/event_cardinality.py`；
- `training/flash_vtg_gmr/calibrate_count.py`；
- `training/flash_vtg_gmr/resolve_artifact.py`；
- `training/flash_vtg_gmr/finalize_part2.py`；
- `eval/eval_hiea2m_diagnostics.py`；
- `tests/test_candidate_interface.py`；
- `tests/test_event_matching.py`；
- `tests/test_event_set_metrics.py`；
- `tests/test_adapter_gradient.py`；
- `tests/test_cardinality.py`；
- `tests/test_inference_selection.py`；
- `tests/test_public_adapter_contract.py`；
- `tests/test_part2_completion.py`。

## 14. 必须通过的测试

1. B0 mode off 时 prediction JSONL 完全不变；
2. `candidate_topk_idx/point/scale` 保留 batch 维，candidate tensors 的 K/index/mask/span 一一对应；
3. `EventInterfaceV1` schema/shape/mask/hash 校验通过，正式数据路径不能注入预计算 events；
4. `event_mask=True` 仅对应有效 seed，并传播到 attention、matching、loss、pooling、PB、selection 和 JSONL；
5. null query 的所有 valid modes 都是 no-event，padding modes 不进入 loss；
6. Hungarian 不会把一个 GT 匹配给多个 modes，且 padding mode 不进入 cost matrix；
7. P0-selection loss 不含 span regression，seed span 无梯度；
8. `RelationEncoder` 从 event/quality loss 获得非零梯度，seed index 不依赖其可训练输出；
9. P0-R 的 `delta_m` 有非零 span 梯度，`rho==0.5` 且初始 residual 为 0；
10. quality target detached，null target 为 0；
11. duplicate/full-coverage/SetSuccess toy cases 与第 6 节定义一致，包括 null success；
12. public P0 参数不在 C1/C2 optimizer，训练前后 hash 不变；
13. G0-Threshold 无 count head/额外空集门控，threshold 只从 val 拟合；
14. G0/C1 使用同一个 `CountHeadV1` 结构与逐元素相同初始化，主 C1/C2 不含 max/expected-count/consistency；
15. G0 loss 不含 count contrastive，G0-Con 才包含；C1 不含，C2 才包含；
16. Poisson-binomial DP 与枚举一致且概率和为 1；
17. PB activity residual 零初始化、获得非零梯度且 public P0 hash 不变；
18. `P_PB^5` 与 CE 共用完全相同的 weights；
19. C1/C2 registry 固定 AEC-CE，count-head override 必须失败；
20. AEC 只有 `argmax P_count` 一个空集 hard decision；
21. count 1/2/3 只从 valid event modes 选 Top-N；
22. prediction raw artifact 可确定性重放 selected set；
23. train/eval 不读取 phrase manifest。

## 15. 阶段验收与 Part 3 交接

### 15.1 “Part 2 完成”的唯一含义

只有以下四组条件**同时满足**，才可以把 Part 2 标记为 `COMPLETE`。代码写完、单个 seed 跑通或只得到一张指标表都不算完成。

#### A. 工程闭环

- 第 13 节中主线所需的 candidate interface、P0-selection、AGC-Direct、AEC-CE、calibration、selection 和 diagnostics 已实现；
- `G0-Threshold` 能通过 `scripts/run_hiea2m.sh` 完成 validation threshold calibration 和 test inference；`G0/G0-Con/P0/C1/C2` 能通过同一脚本完成 train 和 test inference；G0/G0-Con/C1/C2 另须完成 validation-only count calibration，P0 明确不生成 count calibration；
- B0 mode-off 的预测与 Part 1 锁定产物一致；
- C1/C2 启动前后 B0 与 public P0 参数 hash 不变；
- inference 只从 checkpoint 恢复架构与固定规则，不读取 test 标签，不接受改变模型语义的 CLI override。

#### B. 验证闭环

- 第 14 节全部适用于主线的测试通过；
- 至少有一个覆盖 `count=0/1/multi` 的小批次完成 forward、backward、checkpoint resume 和 inference；
- 两个 seeds 的所有必跑训练/校准均无 NaN/Inf；G0-Threshold 可由 raw proposals 和 `tau_raw` 重放，G0/G0-Con 可由 raw proposals、count probabilities、calibration 和固定 selection rule 重放，P0 可由 `EventInterfaceV1` 和固定 `0.5` event threshold 重放，C1/C2 可由 `EventInterfaceV1`、count probabilities、calibration 和固定 selection rule 重放；
- G0/G0-Con/C1/C2 的每个 test prediction 数量等于保存的 `pred_count` 所规定的数量：`0` 为空，`1/2/3` 为精确 Top-N，`4+` 满足至少 4、最多 10；G0-Threshold 保存实际 `selected_count`，不得伪造 count class 决策。

#### C. 必跑实验闭环

以下 12 个 required run records 必须全部有 test predictions、metrics 和配置/hash 记录：

```text
2 seeds x {G0-Threshold, G0, G0-Con, P0, C1, C2}
```

其中 G0/G0-Con/P0/C1/C2 共 10 个训练 runs 必须有正式 checkpoint；2 个 G0-Threshold records 必须引用对应 B0 checkpoint hash，并显式记录 `train_status=not_applicable`。G0-Threshold/G0/G0-Con/C1/C2 共 10 个 count/threshold records 必须各有一份仅由 validation 拟合的 calibration；2 个 P0 records 必须显式记录 `calibration=null` 和 `selection_threshold=0.5`，不能放置伪 calibration 文件来满足检查。

还必须生成跨 seed 的 mean/std 汇总和下列预注册比较：

```text
G0-Threshold -> G0
G0 -> G0-Con
B0 -> P0
G0 -> C1
C1 -> C2
```

不得用某个失败 seed 的重跑结果替换原结果而不保留原因和旧 run ID。`P0-R`、C1-Enhanced、PB 和 Exact 不属于完成条件。

#### D. 产物闭环

以下三个索引必须存在、能从空进程重新解析，并通过其中记录的 SHA256 校验：

```text
artifacts/adapters/public_adapter_index.json
artifacts/cardinality/cardinality_index.json
artifacts/cardinality/part2_completion.json
```

`part2_completion.json` 至少包含：

```text
schema_version
status = COMPLETE | INCOMPLETE
part3_handoff = READY | NOT_READY
research_outcome = POSITIVE | MIXED | NEGATIVE
feature_manifest_hash
data_manifest_hash
baseline_index_hash
required_variants = [G0-Threshold, G0, G0-Con, P0, C1, C2]
required_seeds = [2024, 2025]
run_ids/checkpoint_hashes/calibration_hashes_or_null/prediction_hashes/metric_hashes
test_command + test_result
aggregate_metrics_path + report_path
unmet_requirements
```

`training.flash_vtg_gmr.finalize_part2` 必须从磁盘实际产物核验，不允许仅凭训练目录存在就写入 `COMPLETE`。任一必跑 run、hash、适用的 calibration、prediction、metric 或测试缺失时，`status` 必须为 `INCOMPLETE` 并列出 `unmet_requirements`。

唯一收口命令为：

```bash
python -m training.flash_vtg_gmr.finalize_part2 \
  --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
  --data_manifest_index artifacts/manifests/standard/manifest_index.json \
  --baseline_index artifacts/baselines/baseline_index.json \
  --adapter_index artifacts/adapters/public_adapter_index.json \
  --cardinality_root artifacts/cardinality \
  --required_seeds 2024 2025 \
  --required_variants G0-Threshold G0 G0-Con P0 C1 C2 \
  --test_result artifacts/cardinality/test_result.json \
  --output artifacts/cardinality/part2_completion.json
```

命令在 `status=COMPLETE` 时退出码为 0；否则仍写出诊断 JSON，但退出码非 0。恢复上下文后，以该文件和命令为唯一完成判据，不根据目录数量或训练日志人工判断。

### 15.2 必须报告的指标

必须报告：

```text
Count-Acc-5
SetSuccess@0.5
mAP, mR+@5, G-mIoU@1/3/5
AUROC, Rej-F1, null FPR
five-class count accuracy, exact accuracy, MAE
over/under prediction rate
DuplicateRate@0.5
Selected-FullCoverage@0.5
Oracle-Mode-FullCoverage@0.5
null/single/multi grouped metrics
```

论文对应的两个主指标固定为 `Count-Acc-5` 和 `SetSuccess@0.5`；其余指标只解释数量、空集、重复、覆盖和定位误差来源。

P0 通过条件：

- validation mAP 相对同 seed B0 下降不超过 `0.5` 个百分点；
- AdapterScore 按预先定义选择；
- `Oracle-Mode-FullCoverage@0.5` 相对同 seed `Raw-Proposal-Oracle-FullCoverage@0.5` 下降不超过 `5.0` 个百分点；
- 两个 public P0 hashes 已锁定。

上述 P0 门槛必须对两个 seeds 分别成立，不能只看均值。若某个 seed 不成立，实验执行仍可形成完整的负面诊断，但 `part3_handoff` 必须为 `NOT_READY`，不能让 Part 3 消费该 P0。

### 15.3 工程完成、研究结论与交接状态分离

C1/C2 不设置拍脑袋的绝对提升门槛。只要预注册实验完整执行并如实报告，AEC 没有提升也可以把工程/实验状态标为 `COMPLETE`，同时将 `research_outcome` 标为 `MIXED` 或 `NEGATIVE`。不得为了得到正结果临时改变 count 类别、空集门控、threshold 或主实验编号。

`research_outcome` 只总结第 11.2 节预注册比较的方向和失败模式，不参与 `status` 的计算；具体结论及逐 seed 数值必须写入 `part2_report.md`，不能只保留一个枚举标签。

`part3_handoff=READY` 的必要条件是：

- `status=COMPLETE`；
- 两个 public P0 都通过第 15.2 节 P0 门槛；
- 唯一 count rule 可重放；
- C1/C2 的 positive-query mAP 相对对应 P0 下降均不超过 `0.5` 个百分点；
- count/full-coverage/null 指标与失败模式已完整记录。

若只有 AEC 的 mAP 门槛未满足，Part 3 可以先实施 H1/H2/H3，但不得启动或宣称 F1/F2 的正式联合结果；修复必须注册为新 variant，不能覆盖 C1/C2。若 P0 门槛未满足，则整个 Part 3 的 H/F 主实验均不得使用该 Adapter。

### 15.4 交接产物

交接产物：

```text
artifacts/adapters/public_adapter_index.json
artifacts/cardinality/cardinality_index.json
artifacts/cardinality/part2_completion.json
artifacts/cardinality/{seed}/{G0-Threshold,G0,G0-Con,C1,C2,...}/
    checkpoint + opt + predictions + metrics + hashes
    calibration                         # G0/G0-Con/C1/C2 only
artifacts/cardinality/aggregate_metrics.json
artifacts/cardinality/part2_report.md
```

Part 3 只能消费 public P0；不得根据 HMSA/F2 结果回头重选 P0。

一句话判据：**12 个 required run records（其中 10 个训练 runs）和全部验证/索引产物齐全且 finalizer 写出 `status=COMPLETE`，表示 Part 2 已完成；只有同一文件同时写出 `part3_handoff=READY`，才允许启动依赖 P0/AEC 的 Part 3 正式联合实验。**
