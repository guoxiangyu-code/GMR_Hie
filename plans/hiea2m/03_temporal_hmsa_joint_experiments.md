# Part 3: Temporal HMSA、联合模型与最终实验

## 1. 任务目标与边界

本任务只完成以下内容：

1. 实现 Token-Clip/Event、Phrase-Event、Query-Video 三层 Temporal HMSA；
2. 让 alignment context 回流到 event prediction，而不是只增加辅助 loss；
3. 将 Temporal HMSA 与固定的 AEC-CE 组成最终模型；
4. 实现可选的 Visual Event-to-Phrase Reconstruction（V-EPR）；
5. 完成主实验、消融、模板依赖和 phrase-label 泄漏诊断。

本阶段不得重新选择特征、修改 Standard split、重训另一个 B0，或根据最终结果重选 Proposal-to-Event Adapter。主实验始终使用 Part 2 冻结的 `P0-selection`；event spans 是固定 seed spans，不启用 boundary residual 或 span regression。

## 2. 前置产物与 Fail-Fast

运行前必须存在：

```text
artifacts/features/f-lighthouse/feature_manifest.json
artifacts/manifests/standard/manifest_index.json
artifacts/baselines/baseline_index.json
artifacts/adapters/public_adapter_index.json
artifacts/cardinality/cardinality_index.json
artifacts/cardinality/part2_completion.json
```

启动时验证：

```text
feature_setting == "f-lighthouse"
known_provenance == true
encoder_mode == "eval"
text_token_alignment_status == "verified"
feature_order == [slowfast, clip]
feature_dim == 2816
text_dim == 512
clip_length == 2
split == Standard
seeds == [2024, 2025]
adapter_variant == P0-selection
adapter_frozen == true
adapter_hash == public_adapter_index[seed].sha256
baseline_hash == baseline_index[seed].sha256
event_interface_schema == "EventInterfaceV1"
part2_completion.status == "COMPLETE"
part2_completion.part3_handoff == "READY"
phrase_target_hashes == manifest_index.phrase_targets.{train,val}.sha256
```

任一检查失败则停止。`cardinality_index.json` 用于复现 G0/C1/C2、验证 `CountHeadV1` config/init hash，并提供公平比较；F1/F2 不继承 C1/C2 的已训练参数，但必须复用同构结构和相同初始参数。
test phrase targets 只允许由离线诊断 evaluator 读取；模型训练、普通 validation/test
inference 和模型选择均不得读取 `phrase_targets.test_diagnostic`。

## 3. 实验编号锁定

主实验名称是稳定接口：

```text
H1 = P0 + Token-Clip/Event alignment
H2 = H1 + Phrase-Event alignment
H3 = H2 + Query-Video alignment
F1 = H3 + AEC-CE
F2 = H3 + AEC-CE + count contrastive
R1 = F2 + V-EPR
```

内部消融使用独立名称：

```text
F-PB        = H3 + AEC-PB-bin
F1-Global   = F1 + zero-init global residual into CountHeadV1
F2-unfreeze = F2，但解冻 P0；不进入主表
H3-aux-only = 三层 alignment 只算辅助 loss，不回流 event；机制消融
```

固定语义：

- `F1/F2` 永远使用五类 `AEC-CE`；
- `F1` 不含 count contrastive，`F2` 固定包含；
- `R1` 只比 `F2` 多 V-EPR；
- `H1/H2/H3` 不使用 AEC，按固定 event threshold 输出，用于隔离 alignment 增益；
- PB/Exact 不能通过 CLI 参数覆盖 `F1/F2` 的含义。

## 4. 输入张量与信息边界

### 4.1 模型输入

正式 `model.forward()` 的数据输入只有：

```text
video_content     : B x T x 2816   # [L2Norm(SlowFast) || L2Norm(CLIP)]
video_mask        : B x T
query_tokens      : B x L x 512    # contextual CLIP text features
query_mask        : B x L
```

mask 方向统一为 `True=valid`。根据 Part 1 的连续文本 mask 契约确定性派生：

```text
lexical_mask = query_mask
lexical_mask[:, 0] = False                         # SOT
lexical_mask[b, last_true(query_mask[b])] = False  # EOT
```

必须逐样本断言至少存在 SOT/EOT、valid positions 连续，并在训练数据加载时与 canonical manifest 保存的 `lexical_mask` 完全一致。manifest lexical mask 只用于该一致性检查和 criterion targets，不构成另一种推理信号。

candidate/event tensors 不是 dataset 字段。模型内部固定执行：

```text
video/query inputs
    -> frozen B0 -> candidate tensors
    -> frozen public P0 -> EventInterfaceV1
    -> Temporal HMSA
```

`EventInterfaceV1` 必须逐项符合 Part 2 第 5.6 节，固定 `M=10`，包括 `event_feat/event_span/adapter_event_logit/adapter_quality_logit/event_mask/query_global` 及 B0/P0/feature hashes。正式 train/eval 禁止加载预计算 event features。

`video_content` 表示尚未融合当前 query 的纯视觉输入；TEF、query embedding 或 cross-attended features 不得混入需要视觉隔离的分支。

### 4.2 Criterion-only targets

训练 criterion 可读取：

```text
gt_spans
count_label
is_null
action_token_indices
team_token_indices
action_label
team_label
relation_record_ids       # optional, only references the frozen relation index
```

但以下字段不得成为 `model.forward()` 参数，也不得保存进推理输入：

```text
action_token_indices
team_token_indices
action_label
team_label
phrase embedding targets
```

训练时的 token/phrase indices 只能构造 loss target、关系 mask 和诊断标签。推理表示必须由模型对全部有效 query tokens 的注意力自行产生。

## 5. Temporal HMSA 总体结构

HMSA 输入本次 forward 内生成的 `EventInterfaceV1` 和视频/query features，产生：

```text
c_word_m    : B x M x 256
c_phrase_m  : B x M x 256
c_global    : B x 256
aligned_event_m : B x M x 256
```

三层关系为：

```text
Token-Clip/Event -> Phrase-Event -> Query-Video
```

它们不是三个独立分类器。上下文必须回流到 `aligned_event_m`，随后用于 event activity、localization quality 和联合 AEC。

所有 event-wise 模块都必须接收 `event_mask`。invalid modes 在 attention 中既不能作为 query/key，也不能参与 pooling、loss 或 selection；其输出在每层后显式清零，防止 bias/residual 重新产生非零 padding 表示。

## 6. 第一层：Token-Clip/Event Alignment

### 6.1 Token-Clip attention

将纯视觉 clip features 投影为 `v_t`，query tokens 投影为 `w_l`：

```text
A_word[k,t] = masked_softmax_t((Ww*w_k)^T (Wv*v_t) / sqrt(d))
```

先从文本本身预测 factor prior，所有 softmax 都只允许 `lexical_mask=True` 的 tokens：

```text
beta_r[k] = masked_softmax_k(Head_r(query_tokens[k]), lexical_mask)
r in {full, action, team}
```

Token-Clip 的 factor response 使用文本 prior：

```text
u_r[t] = sum_k beta_r[k] * A_word[k,t]
r in {full, action, team}
```

然后为每个 valid event mode 计算事件相关注意力：

```text
alpha_mr[k] = masked_softmax_k(
    dot(Q_r(event_feat_m), K_r(query_tokens[k])) / sqrt(d)
    + log(beta_r[k] + 1e-8),
    lexical_mask
)

c_m^r = sum_k alpha_mr[k] * V_r(query_tokens[k])
```

同一个 query 的不同 modes 必须得到独立 `alpha_mr`；实现不得把 `beta_r` broadcast 后直接当作 event attention。`action_token_indices/team_token_indices` 只构造 attention loss targets，不能直接替代 `beta/alpha` 或进入 inference。

### 6.2 唯一 Full-query Temporal MIL

只有完整查询使用 GT 内外的时间 MIL。对每个 GT moment `j`：

```text
B_j+ = valid clips whose centers fall inside gt_span_j
B-   = valid clips outside union(expand(gt_spans, 2 seconds))

bag(u, B) = tau_mil * logmeanexp({u[t] / tau_mil : t in B})
tau_mil = 0.1
margin_mil = 0.2

L_full-temporal = 1/J * sum_j softplus(
    margin_mil + bag(u_full, B-) - bag(u_full, B_j+)
)
```

边界情况：

- positive query 但 `B-` 为空时，该样本不计算此项；
- null query 没有 `B_j+`，`L_full-temporal=0`；
- 多 GT 使用 GT union 定义 `B-`，不能把另一个正事件当负样本；
- action/team 不得套用该 inside-vs-outside loss。

这是主方案唯一的 Token-Clip temporal MIL 实现，不同时保留 max-pooling、hinge 或逐 clip BCE 等替代定义。

### 6.3 Token-Event factor alignment

Hungarian matched 且 `event_mask=True` 的 event modes 是监督 anchors。定义：

```text
I_full(i)   = {k | lexical_mask_i[k]}
I_action(i) = action_token_indices_i
I_team(i)   = team_token_indices_i

L_r-event = - 1/|A_r| * sum_{(i,m) in A_r}
              1/|I_r(i)| * sum_{k in I_r(i)} log(alpha_imr[k] + 1e-8)

L_token-event = L_full-event + L_action-event + L_team-event
L_token = L_full-temporal + L_token-event
```

其中 `A_r` 是 factor 可用样本的 matched-mode 集合；若 `A_r` 为空，该项精确为 0。full target 使用全部 lexical tokens，不含 SOT/EOT；Sports/unavailable team 或空 token index 使 team anchor 不进入 numerator/reduction。loss 先对 target tokens 求均值，再对 anchors 求均值，不按某条 query 的 GT 数量重复加权。

本层只监督“某个 event 应注意哪些 tokens”，不再使用未定义的 `factor_alignment(c_m,target)`。`c_m^r` 的 embedding 语义由下一层 Phrase-Event 负责。

action/team 的 temporal negatives 只允许来自可证明安全的 event regions：

1. 同视频具有明确 factor 标注；
2. 该 region 的 factor label set 与当前 factor label set 无交集；
3. 与当前 query 的任一 GT span 的 tIoU 小于 `0.1`。

不满足三项的 region 标为 unknown，不进入 denominator。不得把全部 `B-` 当 action/team negatives。

主 H1/H2/H3 配置不需要这些 factor temporal negatives；只有显式启用 relation-index 消融时才构造它们。

## 7. 第二层：Phrase-Event Alignment

### 7.1 训练 targets 与推理 contexts

训练 target：

```text
p_action* = stop_gradient(masked_mean(query_tokens[action_token_indices]))
p_team*   = stop_gradient(masked_mean(query_tokens[team_token_indices]))
p_full*   = stop_gradient(masked_mean(query_tokens[lexical_mask]))
```

模型 context：

```text
p_action_hat_m = Project(c_m^action)
p_team_hat_m   = Project(c_m^team)
p_full_hat_m   = Project(c_m^full)
```

推理只使用 `p_*_hat_m`。禁止把 `p_*` target 拼接到 event feature 或 AEC 输入。

### 7.2 Multi-positive loss

每个 projected event-specific context 是 anchor；同一句 query 的所有 Hungarian matched valid modes 都各自是正 anchor，且绝不互为负样本。定义 `KnownNeg(i,m,r)` 为 relation index 明确证明与 event `(i,m)` factor `r` 不匹配的 phrase targets：

```text
z_imr = L2Norm(p_r_hat_im)
y_jr  = L2Norm(stop_gradient(p_jr*))

ell_imr = -log (
    exp(sim(z_imr, y_ir) / tau_phrase)
    /
    (exp(sim(z_imr, y_ir) / tau_phrase)
     + sum_{j in KnownNeg(i,m,r)} exp(sim(z_imr, y_jr) / tau_phrase))
)

A_r = {(i,m) | m is matched, event_mask_im=True,
                 factor r available}

L_r-phrase = sum_{(i,m) in A_r} ell_imr / max(|A_r|,1)
L_phrase-event = L_full-phrase + L_action-phrase + L_team-phrase
```

固定 `tau_phrase=0.07`。reduction 先得到每个 event anchor 的 loss，再在有效 anchors 上求均值；有已知 negatives 时使用上述 log-softmax，denominator 只含一个对应 positive 和明确 negatives。若 `KnownNeg(i,m,r)` 为空，明确改用：

```text
ell_imr = 1 - cosine(z_imr, y_ir)
```

因此 matched anchor 始终有正对齐梯度，但不存在伪造 negative。负样本规则：

- full phrase：只使用 relation index 明确记录完整 query-event 不匹配的 targets；
- action/team：只使用 relation index 明确记录 label mismatch、label set disjoint 且 GT 不重叠的 targets；
- 其他视频或标签未知 modes 默认不是负样本；
- 同一 query 的多个 matched modes 不得互为负样本。

若主配置没有可用 relation index，则 `L_phrase-event` 只对能由同视频完整标注明确证明的 relations 计算；不得回退到 batch shuffle negatives。

## 8. 第三层：Query-Video Alignment

### 8.1 全局表示

使用 lexical text、纯视觉特征和冻结 P0 的预对齐 event activity 构造：

```text
hmsa_text_global = MaskedMean(query_tokens, lexical_mask)
v_mean = MaskedMean(Project(video_content), video_mask)
v_max  = MaskedMax(Project(video_content), video_mask)

p_adapter_m = where(event_mask_m,
                    sigmoid(adapter_event_logit_m), 0)
e_weight = sum_m p_adapter_m * event_feat_m
           / (sum_m p_adapter_m + 1e-6)

c_global = MLP([hmsa_text_global, v_mean, v_max, e_weight])
query_video_logit = Linear(c_global)
```

这里的 `adapter_event_logit` 只能来自本次 forward 的冻结 `EventInterfaceV1`。禁止用后续 `final_event_logit` 回算 `c_global`，因此计算图是单向的，不存在 logit -> global -> aligned event -> 同一 logit 的循环依赖。

### 8.2 Binary 与对比监督

```text
L_qv-bce = BCEWithLogits(query_video_logit, has_any_gt)
```

主监督优先级：

1. 显式 positive query-video pairs；
2. 数据集中显式 null pairs；
3. 同视频共享部分 action/team、但完整 query 不成立的已标注 hard negatives。

若使用 symmetric contrastive：

```text
L_qv-con = RelationMaskedSymmetricInfoNCE(hmsa_text_global, video_global, relation_mask)
```

`relation_mask[i,j]` 只有在数据标注明确证明 pair 匹配或不匹配时才有效。任意 batch-shuffled query-video pair 不默认是 negative；unknown pairs 必须从 numerator 和 denominator 同时排除。主配置固定：

```text
lambda_qv_bce = 1.0
lambda_qv_con = 0.0
```

`L_qv-con` 仅作为显式 relation mask 完整时的消融。

### 8.3 冻结 relation index

安全 action/team negatives 和 QV contrastive 只能来自离线构建、带 provenance hash 的：

```text
artifacts/relations/standard/
    factor_event_relations.jsonl
    query_video_relations.jsonl
    relation_index.json
```

每条 relation 至少记录：

```text
anchor canonical query key
candidate canonical query/video/annotated-window key
factor = full | action | team
relation = positive | negative | unknown
reason
source labels
GT overlap/tIoU
canonical manifest hash
phrase-target hash
builder version
```

`relation_index.json` 记录行数、positive/negative/unknown coverage、文件 SHA256 和生成命令。训练数据只携带 relation record IDs，由 criterion 查找已冻结关系；模型 forward 不接收 relation matrix。

离线 `event` 指 canonical annotated window，不是尚未生成的 P0 mode。训练时只有当 mode 通过 Hungarian/tIoU 明确关联到该 annotated window 时，才能继承 relation；未关联 mode 保持 unknown。relation resolver 可按 qid 从冻结的 train phrase targets 读取 `stop_gradient` target，不能把 relation label 或 target 传入 model forward，也不能在训练中读取 val/test phrase targets。

若 relation index 缺失或 hash/coverage 校验失败，主配置仍可运行，但必须固定：

```text
lambda_qv_con = 0
action/team cross-query negatives = disabled
```

此时 Query-Video 只使用原始 positive/null BCE，Phrase-Event 只使用同视频且由完整标注明确证明的 relations。实现不得根据当前 batch、文本相似度或模型分数临时猜 negative。

## 9. Alignment 必须回流到最终 Event

### 9.1 Aligned event representation

汇总第一、二层 context：

```text
c_word_m   = W_word [c_m^full || c_m^action || c_m^team]
c_phrase_m = W_phrase [p_full_hat_m || p_action_hat_m || p_team_hat_m]

aligned_event_m = event_feat_m
    + gamma_word   * LayerNorm(c_word_m)
    + gamma_phrase * LayerNorm(c_phrase_m)
    + gamma_global * LayerNorm(Broadcast(c_global))
```

每个 residual context 在相加前乘 `event_mask_m`，相加后再次将 invalid modes 清零。

`gamma_word/gamma_phrase/gamma_global` 是可训练标量或逐通道 gate，全部零初始化。初始化后必须满足：

```text
aligned_event == event_feat
```

若 public P0 原本在输出前已有 LayerNorm，则它仍保留在原位置；不要为 HMSA 新增 outer LayerNorm。所有 gamma 为零时 event/quality logits 必须与 P0 逐元素一致。

### 9.2 最终预测头

```text
aligned_event_m
  -> frozen public-P0 event head -> final_event_logit_m
  -> frozen public-P0 quality head -> final_quality_logit_m
  -> joint count summary
```

主模型继续使用固定 `event_span_m=seed_span_m`。不得从 aligned events 新增 boundary delta。P0 event/quality heads 的参数冻结，但它们对输入的梯度不能 detach，使定位监督能更新 HMSA contexts/gates。`H3-aux-only` 直接使用原 P0 logits，仅用于证明收益是否来自表示回流。

全篇命名锁定：

```text
adapter_event_logit / adapter_quality_logit = EventInterfaceV1 中的冻结 P0 输出
final_event_logit   / final_quality_logit   = aligned_event 经过冻结 P0 heads 的输出
```

不得再用无前缀的 `event_logit/quality_logit` 指代两者之一。所有 H/F 最终排序和输出使用 `final_*`；`c_global` 只使用 `adapter_*`。

## 10. 联合 AEC

### 10.1 与 C1/C2 同构的 F1/F2 输入

F1/F2 复用 Part 2 定义的同一个 `CountHeadV1` class/config。主比较只把 public P0 events 替换为 aligned events：

```text
text_mean          = MaskedMean(query_tokens, query_mask)
aligned_event_mean = MaskedMean(aligned_event, event_mask)

g_joint = CountHeadV1.encode(text_mean, aligned_event_mean)
P_joint = softmax(CountHeadV1.classifier(g_joint) / T_count)
```

F1/F2 不读取 `c_global`、event max、final logits 或 expected count 作为 count-head 额外输入。它们不继承 C1/C2 的已训练权重，但 `CountHeadV1` 使用同一 module-init key `(seed,"CountHeadV1")` 和 isolated RNG fork/local generator，因此不受 HMSA 模块创建顺序影响；C1/F1、C2/F2 的 count head 在训练开始前结构和参数逐元素一致。

`c_global` 对主 F1/F2 的影响只能先经过零初始化 HMSA residual 改变 `aligned_event`，不能绕过 event representation 直接进入 count head。额外全局计数信息只允许使用独立消融：

```text
F1-Global:
    g_global = g_joint + gamma_count_global * W_global(c_global)
    gamma_count_global = 0 at initialization
```

`F1-Global` 不得替换主 F1/F2 或用于 `C1 -> F1` 归因。

### 10.2 AEC-CE 与 count contrastive

```text
count_class = {0,1,2,3,4+}

L_count = -w[bin(n_gt)] * log P_joint[bin(n_gt)]
```

`w` 复用 Part 2 基于 Standard train split 的 effective-number weights，并裁剪到 `[0.5,2.0]`；所有 seeds 和 CE/PB variants 使用同一份带 hash 的权重。

```text
L_F1-count = L_count
L_F2-count = L_count + 0.1 * L_count-con
```

F2 的 supervised count contrastive 直接使用 `Projection(g_joint)`，与 Part 2 使用相同 `{0,1,2,3,4+}` 标签、投影定义和 class-balanced detached queue，但 queue 必须重新初始化，不得导入 C2 queue。PB/consistency 只属于 `F-PB` 或显式 enhanced 消融，不进入 F1/F2。

### 10.3 唯一空集决策

F1/F2/R1 的唯一 hard decision：

```text
pred_count = argmax P_joint
output = empty set iff pred_count == 0
```

`query_video_logit`、`1-P_joint[0]`、event activity 和任何 existence score 都不能形成第二个空集门控。`query_video_logit` 只作为训练信号和 `c_global` 的软表示来源。

非空 selection 沿用 Part 2：

- count 1/2/3：在 `event_mask=True` 的 modes 中按 `sigmoid(final_event_logit) * sigmoid(final_quality_logit)` 取 Top-N；
- count 4+：在 valid modes 中使用 validation 固定的 `tau_mode`，至少 Top-4、最多 10；
- 只在 valid event modes 上选择，不执行 NMS；
- `T_count/tau_mode` 只在 validation 拟合。

H1/H2/H3 没有 count head，其输出固定为：

```text
selected_modes = {m | event_mask_m and sigmoid(final_event_logit_m) >= 0.5}
```

阈值不按 seed 调整，不使用 legacy existence gate，也不执行 NMS。

## 11. No-Target-Aware Loss Gating

| Loss | Positive query | Null query |
| --- | ---: | ---: |
| legacy MR/HD | 是 | 按 Part 1/2 contract |
| aligned event/no-event | matched/unmatched valid modes | 所有 valid modes 为 no-event |
| aligned quality | continuous IoU | valid modes target=0 |
| span regression | 否 | 否 |
| `L_full-temporal` | 是 | 否 |
| token-event factor | matched | 否 |
| phrase-event | matched | 否 |
| query-video BCE | positive | negative |
| joint AEC | count > 0 | count = 0 |
| count contrastive | 是 | 是 |
| V-EPR | matched | 否 |

Null query 不是把所有 positive-only loss 填成零 target；不具备语义支持的 loss 必须从 reduction denominator 中移除。

## 12. 可选 V-EPR

V-EPR 是 masked text recovery 的视觉隔离时序版本，只属于 `R1`。

### 12.1 视觉隔离输入

对 matched fixed event span，从 `video_content` 做 differentiable temporal ROI pooling：

```text
visual_event_m = TemporalROIPool(
    Project(video_content),
    stop_gradient(event_span_m),
    video_mask
)
```

该分支不得读取：

```text
query_tokens
query_global
c_word/c_phrase/c_global
TEF
query-conditioned candidate/event features
```

### 12.2 Reconstruction loss

```text
action_pred_m = Head_action_rec(visual_event_m)
team_pred_m   = Head_team_rec(visual_event_m)

L_V-EPR = 1 - cosine(action_pred_m, stop_gradient(p_action*))
        + availability(team) * (
              1 - cosine(team_pred_m, stop_gradient(p_team*))
          )
```

仅 Hungarian matched positive modes 计算；null query 不计算；Sports/unavailable team 项关闭。

V-EPR 可用零初始化 residual 把纯视觉语义注入 aligned event：

```text
aligned_event_m <- aligned_event_m
                 + gamma_rec * LayerNorm(W_rec(visual_event_m))
gamma_rec = 0 at initialization
```

必须验证：

- `d visual_event / d query_tokens == 0`；
- query permutation 不改变 `visual_event`；
- video permutation 会改变 `visual_event`；
- 移除 query targets 后模型 inference 路径仍完整。

## 13. 损失与训练顺序

### 13.1 主损失

将 public P0 的同一 Hungarian matcher、event loss 和 detached quality target 应用于 aligned-event logits：

```text
L_P0-aligned = L_event(final_event_logit, event_mask)
             + L_quality(final_quality_logit, event_mask)
```

它不含 span regression。总损失为：

```text
L_H1 = L_P0-aligned
     + 1.0 * L_full-temporal
     + 1.0 * L_token-event

L_H2 = L_H1
     + 1.0 * L_phrase-event

L_H3 = L_H2
     + 1.0 * L_qv-bce

L_F1 = L_H3
     + 1.0 * L_count

L_F2 = L_F1
     + 0.1 * L_count-con

L_R1 = L_F2
     + 0.2 * L_V-EPR
```

各权重作为预注册默认值；只允许在 validation 做预先列出的 sensitivity analysis，不能根据 test 改动。

### 13.2 冻结策略

主实验：

```text
frozen: B0 feature/backbone parameters, public P0 seed selection and adapter
        including public P0 event/quality head parameters
trainable: HMSA projections/attention/gates, joint AEC head,
           optional V-EPR heads
```

冻结 head 参数不等于 detach head 输入。alignment-refined events 必须经过这些 frozen heads，并允许 loss 对 aligned events 回传梯度。public P0 checkpoint 文件和加载后的 P0 参数必须保持 hash 不变。

`F2-unfreeze` 才允许解冻 P0 relation/aggregation 参数。它必须单列训练成本、参数量并重新跑两个 seeds，不能替代 F2。

### 13.3 顺序

每个 seed 独立执行：

1. 读取对应 B0 与 public P0，验证 hash；
2. 分别训练 H1、H2、H3，不从前一个实验的最佳 checkpoint 串行继承；
3. 从相同 B0+P0 初始化分别训练 F1、F2，并用与对应 C1/C2 逐元素相同的 `CountHeadV1` 初始参数；
4. 从 F2 初始化 R1 时，仅新增 V-EPR 参数，记录这一差异；
5. 在 validation 拟合 count temperature 和 `tau_mode`；
6. 冻结 calibration 后一次性运行 test；
7. 保存完整 command、config、hashes、predictions 和 metrics。

H1/H2/H3/F1/F2 的公平主比较应从相同 B0+P0 起点训练。R1 的 staged 初始化是预先声明的增强实验。

## 14. 最终实验矩阵

### 14.1 主表

| 实验 | Public P0 | Token | Phrase | QV | Count-CE | Count-Con | V-EPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | | | | | | | |
| G0-Threshold | | | | | | | |
| G0 | | | | | ✓ | | |
| G0-Con | | | | | ✓ | ✓ | |
| P0 | ✓ | | | | | | |
| H1 | ✓ | ✓ | | | | | |
| H2 | ✓ | ✓ | ✓ | | | | |
| H3 | ✓ | ✓ | ✓ | ✓ | | | |
| C1 | ✓ | | | | ✓ | | |
| C2 | ✓ | | | | ✓ | ✓ | |
| F1 | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| F2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| R1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

`B0/G0-Threshold/G0/G0-Con/P0/C1/C2` 直接引用 Part 1/2 已锁定 predictions 和 metrics，不在本阶段重跑或改名。

### 14.2 必要机制比较

```text
B0 -> P0        : event adapter 本身
G0-Threshold -> G0 : 忠实 AGC count classifier
G0 -> G0-Con    : raw-proposal count contrastive
P0 -> H1        : token-level alignment
H1 -> H2        : phrase-level alignment
H2 -> H3        : query-video alignment
P0 -> C1        : event-level AEC-CE
C1 -> C2        : event count contrastive
C1 -> F1        : CountHeadV1 不变，只将 P0 event 换成 aligned event
H3 -> F1        : 在同一 aligned event 上增加 CountHeadV1
F1 -> F2        : joint count contrastive
F2 -> R1        : 视觉隔离 reconstruction
H3-aux-only -> H3 : alignment 回流的必要性
```

附加结果只放消融/附录：

```text
P0-R, C1-Enhanced, C-PB, C-PB-Con, C-PB-Exact, C-Exact,
F1-Global, F-PB, F2-unfreeze, QV-Con
```

## 15. 指标与唯一评估定义

### 15.1 Grounding 与 null/count

至少报告：

```text
Count-Acc-5
SetSuccess@0.5
mAP
mR+@1, mR+@5
G-mIoU@1/3/5
AUROC, Rej-F1, ECE, Brier
null false-positive rate
five-class count accuracy
exact count accuracy, count MAE
over-prediction rate, under-prediction rate
null/single/multi grouped metrics
```

所有主结果报告两个 seeds 的 mean/std，并保留逐 seed 数值。`mR+` 只在 positive queries 上计算；null 指标必须包含全部 null queries。

论文迁移的两个主指标固定为：

```text
HieA2G N-acc.             -> HieA2M Count-Acc-5
HieA2G Pr@(F1=1,IoU>=.5) -> HieA2M SetSuccess@0.5
```

其他指标用于解释 count、null、duplicate、coverage 或 localization 的具体误差，不替换这两个主指标。

### 15.2 Event duplicate

对 query `q` 的 prediction set `P_q` 和 GT set `G_q`，在：

```text
E_theta = {(p,g) | tIoU(p,g) >= theta}
```

上求最大基数、再最大总 tIoU 的一对一 matching `M_theta(q)`：

```text
P_eligible(q) = {p in P_q | exists g in G_q: tIoU(p,g)>=theta}

DuplicateCount_theta(q)
  = |P_eligible(q)| - |M_theta(q)|

DuplicateRate_theta
  = sum_q DuplicateCount_theta(q)
    / max(sum_q |P_eligible(q)|, 1)
```

只在 positive queries 上做 micro aggregation。false positives 不计为 duplicate；两个预测若分别匹配两个高度重叠 GT，也不算 duplicate。不同 GT 始终是不同标注事件，不得先合并。

### 15.3 Full coverage

复用上一节的 `M_theta(q)`：

```text
FullCoverage_theta(q) = 1[|M_theta(q)| == |G_q|]
FullCoverage_theta = mean_{q:|G_q|>=2} FullCoverage_theta(q)
```

它只在 `|G_q|>=2` 的 multi-target queries 上聚合。固定报告：

```text
Selected-FullCoverage@0.5
Oracle-Mode-FullCoverage@0.5
```

Oracle 指 public P0 的全部 10 个 modes，不经过 predicted count；Selected 指最终输出集合。多个 GT 不得共用同一个 prediction。

### 15.4 SetSuccess

复用第 15.2 节的一对一最大匹配 `M_theta(q)`：

```text
SetSuccess_theta(q) = 1
iff |P_q| = |G_q| = |M_theta(q)|

SetSuccess_theta = mean over all queries
```

对 null query，`G_q` 为空且输出也为空时为 1，否则为 0。主阈值固定 `theta=0.5`。它要求数量完全正确、没有漏检/多检/duplicate，且每个 GT 都有唯一 prediction 达到 tIoU 阈值。

### 15.5 Alignment 指标

```text
full temporal pointing accuracy
action/team token attention IoU or token-F1
phrase-to-event Recall@1 and multi-positive Recall@All
query-video AUROC
event score vs max-GT-tIoU Spearman correlation
```

token/phrase 指标只在对应 manifest status 为 `resolved` 时计算，同时报告 coverage。不能静默丢弃 ambiguous/unavailable 样本后只报告准确率。

## 16. 模板依赖诊断

训练三个不读取视频特征的 probes：

```text
Template-only -> null/non-null, five-class count
Action/team labels only -> null/non-null, five-class count
Query text embedding only -> null/non-null, five-class count
```

协议：

1. probe 只能在 train 拟合；
2. 超参数只看 validation；
3. test 一次性评估；
4. 报告 overall、action、source、template、null/single/multi 分组；
5. 若存在未见 template，单独报告 seen/unseen-template；
6. 若没有自然未见 template，使用按 template ID 的 GroupKFold，仅作为诊断；
7. 与 F2 的 count/null 指标并列，报告 F2 相对 probe 的增益。

额外做输入干预：

```text
shuffle video within label-matched groups
mask action tokens
mask team tokens
replace query with same-template query
```

若视频 shuffle 后性能几乎不变，不能把 count 提升归因于视频事件建模；需要在结果中明确标注模板依赖风险。

## 17. Phrase-Label 泄漏诊断

必须通过以下检查：

1. 删除 phrase manifest 后，eval/inference 输出逐元素不变；
2. 替换 action/team token indices 后，关闭 loss 的 eval 输出逐元素不变；
3. `model.forward` signature 不含 phrase labels/indices；
4. phrase target tensors 均为 `stop_gradient`，只在 criterion 生命周期存在；
5. train batch 保存到 checkpoint/log 时不序列化 phrase target embeddings；
6. 使用 oracle phrase context 的结果必须命名 `Oracle-Phrase`，不得进入主表；
7. V-EPR 对 query tensor 的 autograd gradient 为零；
8. inference CLI 传入 `--phrase_manifest` 必须报错。

还需运行两组对照：

```text
Predicted-Phrase : 正常模型自身 attention context
Oracle-Phrase    : criterion target 注入，仅测上界
```

两者必须在 registry、输出目录和表格中完全分开，防止 oracle label 被误报为模型能力。

## 18. CLI Execution Contract

### 18.1 Variant registry

```text
--variant {H1,H2,H3,F1,F2,R1,F1-Global,F-PB,F2-unfreeze,H3-aux-only}
```

registry 展开后保存不可变配置。例如：

```text
H1: token=true, phrase=false, qv=false, aec=none
H2: token=true, phrase=true,  qv=false, aec=none
H3: token=true, phrase=true,  qv=true,  aec=none
F1: token=true, phrase=true,  qv=true,  aec=ce, count_con=false
F2: token=true, phrase=true,  qv=true,  aec=ce, count_con=true
R1: F2 + vepr=true
F1-Global: F1 + count_global_residual=true, gamma_count_global_init=0
```

CLI 不提供 `--aec_type` 去改写主编号。需要 PB 时只能选择 `F-PB`。

### 18.2 必需参数

训练：

```text
--feature_manifest
--data_manifest_index
--baseline_index
--public_adapter_index
--phrase_manifest_index
--relation_index             # optional; required only by relation-based losses
--variant
--seed
--output_dir
```

推理：

```text
--feature_manifest
--data_manifest_index
--public_adapter_index
--checkpoint
--variant
--count_calibration       # F1/F2/R1/F-PB required
--split {val,test}
--output
```

推理 CLI 不接受 `--phrase_manifest` 或 `--phrase_manifest_index`。checkpoint 内保存训练时 phrase manifest hashes 仅供 provenance 检查，不能读取其内容参与 forward。

### 18.3 配置互斥与 Fail-Fast

```text
H1/H2/H3: aec must be none
F1/F2/R1/F1-Global: aec must be ce
F-PB: aec must be pb-bin
R1 only: vepr=true
main variants: adapter_frozen=true, boundary_residual=false
qv_con=true requires explicit relation mask coverage report
factor_cross_query_negatives=true requires the same verified relation index
missing relation index forces qv_con=false and factor_cross_query_negatives=false
F1/F2: count_head=CountHeadV1, count_global_residual=false
F1-Global only: count_global_residual=true
```

variant、checkpoint metadata、calibration variant 或 hashes 不一致时立即失败。

### 18.4 执行命令

可选 relation index 只允许离线从 train annotations/manifests 构建；构建后冻结，val/test 只做 key lookup：

```bash
python -m training.flash_vtg_gmr.build_relation_index \
  --data_manifest_index artifacts/manifests/standard/manifest_index.json \
  --phrase_manifest_index artifacts/manifests/standard/manifest_index.json \
  --output_root artifacts/relations/standard
```

不使用 relation-based 消融时可以不构建该产物，resolved config 必须记录两项 relation loss 均为 false。

```bash
for SEED in 2024 2025; do
  for VARIANT in H1 H2 H3 F1 F2; do
    bash scripts/run_hiea2m.sh train-joint \
      --variant "$VARIANT" \
      --seed "$SEED" \
      --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
      --data_manifest_index artifacts/manifests/standard/manifest_index.json \
      --baseline_index artifacts/baselines/baseline_index.json \
      --public_adapter_index artifacts/adapters/public_adapter_index.json \
      --phrase_manifest_index artifacts/manifests/standard/manifest_index.json \
      --output_dir "artifacts/hmsa_joint/$SEED/$VARIANT"
  done
done
```

Calibration：

```bash
bash scripts/run_hiea2m.sh calibrate-joint \
  --variant F2 \
  --checkpoint "$F2_CKPT" \
  --split val \
  --output "$F2_RUN/calibration.json"
```

Test inference：

```bash
bash scripts/run_hiea2m.sh infer-joint \
  --variant F2 \
  --checkpoint "$F2_CKPT" \
  --count_calibration "$F2_RUN/calibration.json" \
  --split test \
  --output "$F2_RUN/test_predictions.jsonl"
```

每次运行保存：

```text
command.txt
resolved_config.json
input_hashes.json
checkpoint.pt
checkpoint.sha256
calibration.json                # where applicable
val_predictions.jsonl
test_predictions.jsonl
metrics.json
environment.txt
```

## 19. 实现文件

修改：

- `models/flash_vtg_gmr/model.py`：接入 HMSA、aligned event heads 和 joint AEC；
- `models/flash_vtg_gmr/loss.py` 或当前实际 criterion 文件：三层 alignment、gating 和 joint count losses；
- `training/flash_vtg_gmr/dataset.py`：仅在 train target 中附加 phrase indices/relations；
- `training/flash_vtg_gmr/config.py`：注册不可变 variants；
- `training/flash_vtg_gmr/train.py`：加载并冻结 public P0；
- `training/flash_vtg_gmr/inference.py`：模型自生成 phrase contexts，唯一 AEC 空集规则；
- `scripts/run_hiea2m.sh`：增加 `train-joint/calibrate-joint/infer-joint/diagnose`。

新增：

- `models/flash_vtg_gmr/temporal_hmsa.py`；
- `models/flash_vtg_gmr/visual_event_reconstruction.py`；
- `training/flash_vtg_gmr/diagnose_template_dependence.py`；
- `training/flash_vtg_gmr/diagnose_phrase_leakage.py`；
- `training/flash_vtg_gmr/build_relation_index.py`；
- `training/flash_vtg_gmr/aggregate_final_results.py`；
- `tests/test_temporal_hmsa.py`；
- `tests/test_hmsa_loss_gating.py`；
- `tests/test_query_video_relations.py`；
- `tests/test_event_interface_v1.py`；
- `tests/test_joint_aec_contract.py`；
- `tests/test_phrase_leakage.py`；
- `tests/test_vepr_isolation.py`；
- `tests/test_joint_cli_contract.py`；
- `tests/test_final_metrics.py`。

若 criterion 实际仍位于 `models/flash_vtg_gmr/model.py`，不要为了匹配本文路径做无关重构；在现有位置实现并相应调整测试 import。

## 20. 必须通过的测试

### 20.1 Temporal HMSA

1. `lexical_mask` 可由 `query_mask` 唯一派生并与 Part 1 manifest 一致；SOT/EOT 可进入 B0 mask，但不能进入 HMSA attention/targets；
2. `L_full-temporal` 只使用 full response，action/team response 不访问全局 `B-`；
3. 多 GT 的 `B-` 使用 expanded GT union，其他 positive moment 不进入 negatives；null 或空 `B-` 时无 NaN；
4. `beta_r` 和 `alpha_mr` 在 lexical mask 外严格为 0，同 query 不同 event 的 `alpha_mr` 可不同；
5. Token-Event loss 按 target-token mean 再按 matched-anchor mean 精确 reduction；padding/unavailable factors 不进入分母；
6. Phrase-Event denominator 只含对应 positive 和 relation index 明确 negatives；所有 matched modes 都作正 anchors且互不为负；
7. action/team negative 必须同时通过 known-label、disjoint-label、non-overlap 三个条件；
8. unknown cross-video pairs 不进入 QV contrastive denominator，任意 batch shuffle 不会自动生成 negative；
9. relation index 缺失时强制关闭 QV contrastive/cross-query factor negatives，但 positive/null BCE 路径可完整训练；
10. team unavailable/Sports 时 team losses 为零且不进入 denominator。

### 20.2 表示回流与冻结

11. dataset/collate 注入 candidate/event tensors 必须失败；B0/P0 forward 产生的 `EventInterfaceV1` schema/shape/hash/mask 全部校验；
12. `event_mask` 传播到 HMSA attention、contexts、pooling、loss、PB/selection 和 JSONL，invalid outputs 每层后为零；
13. `c_global` 只依赖 `adapter_event_logit`；改变 `final_event_logit` 不改变已计算的 `c_global`，计算图无循环；
14. 全部 gamma 为零时 aligned events 及 `final_event/quality` 与 `EventInterfaceV1` 的 P0 outputs 逐元素一致；
15. gamma 非零后，event/quality losses 能穿过 frozen heads 到达 HMSA contexts/gates，而 P0 head 参数无梯度；
16. `H3-aux-only` 不改变 P0 prediction path；
17. H1/H2/H3/F1/F2 主配置中 event spans 等于 seed spans；
18. 主配置 optimizer 不含 public P0 参数，训练前后 checkpoint hash 不变；
19. `F2-unfreeze` 之外任何 variant 请求解冻 P0 都失败。

### 20.3 Joint AEC

20. F1/F2 registry 固定 `aec=ce`，override 为 PB/Exact 必须失败；
21. C1/F1、C2/F2 使用同一个 `CountHeadV1` class/config 和逐元素相同的初始参数；
22. 主 F1/F2 count head 只读取 `query_mask` text mean 和 masked aligned-event mean，不直接读取 `c_global/max/expected_count/final logits`；
23. `F1-Global` 的 global residual gate 零初始化且不改变 F1 初始 count logits；
24. F1 不含 consistency/count contrastive，F2/R1 只比对应 CE 增加固定 count contrastive；
25. AEC 只有 `argmax P_joint` 一个空集 hard decision；修改 `query_video_logit` 但保持 `P_joint` 不变时决策不变；
26. count=4+ selection 至少 4、最多 10 且只作用在 `event_mask=True` modes；
27. calibration 只能读 validation，test 路径拒绝重新拟合。

### 20.4 泄漏与 V-EPR

28. eval 删除/置乱 phrase manifest 后预测逐元素一致；
29. `model.forward` 不接受 token indices、factor labels、phrase targets 或 event cache；
30. V-EPR 的 visual event 对 query tensor 梯度严格为零；
31. query permutation 不改变 V-EPR visual event，video permutation会改变；
32. null query 和 unavailable team 不计算对应 V-EPR loss；
33. inference 传 `--phrase_manifest` 或 `--phrase_manifest_index` 均失败；
34. `Oracle-Phrase` 不能写入主结果 registry。

### 20.5 评估与 CLI

35. duplicate/full coverage 使用同一 GT-conditioned 一对一最大匹配，排除 false positives 且不合并 GT；
36. `SetSuccess@0.5` toy cases覆盖 exact set、漏检、多检、duplicate、低 tIoU 和 null success/failure；
37. 两个 GT 不能由同一 prediction 同时覆盖；
38. checkpoint/adapter/feature/manifest/relation hash mismatch 全部 fail-fast；
39. resolved variant 与 checkpoint metadata 不一致时失败；
40. H1/H2/H3/F1/F2 使用同一 validation checkpoint 词典序规则；
41. 三 seed 聚合保存 mean/std 和逐 seed 结果；
42. smoke train、resume、val calibration、test inference 全链路通过。

## 21. 结果选择与统计规则

Checkpoint 选择：

```text
H1/H2/H3/F1/F2/R1:
    primary  = validation SetSuccess@0.5 (maximize)
    tie 1    = validation positive-query mAP (maximize)
    tie 2    = validation Count-Acc-5 (maximize)
    tie 3    = earliest epoch
```

所有 H/F 主模型使用完全相同、预先固定的词典序规则，不再使用 AdapterScore 或手工加权 JointScore。H1/H2/H3 虽无 count head，仍可由最终 selected set 大小计算 `Count-Acc-5`。P0 在 Part 2 中单独按 `Selected-FullCoverage` 与 `DuplicateRate` 的 AdapterScore 选择，Oracle coverage 只作为硬约束。

统计要求：

- 两个 seeds 报告 mean/std；
- 同 seed 的关键模型使用 paired bootstrap 计算 95% CI；
- 至少比较 `P0 vs H3`、`C1 vs F1`、`C2 vs F2`、`F1 vs F2`；
- 同时报告效果、参数量、训练显存和推理时间；
- 不以 test 的单 seed 最优 checkpoint 作为最终结果。

## 22. 阶段验收与最终交付

实现完成必须满足：

- H1/H2/H3/F1/F2 三 seeds 全部完成；R1/F1-Global/F-PB 等增强消融不阻塞 Part 3 主线完成；
- 主表中的每个编号只有一个不可变配置；
- public P0 与 F-Lighthouse hashes 全程不变；
- `EventInterfaceV1` 只在 forward 内生成，`event_mask` 全路径生效；
- full-only MIL、安全 factor negatives、relation 缺失退化路径和显式 QV relations 均有测试；
- event-specific token attention 与 lexical-only HMSA mask 均有测试；
- alignment 确实回流最终 event/quality/count prediction；
- C1/F1 使用同构同初始化 `CountHeadV1`，F1 主配置不直接注入 `c_global`；
- F1/F2 使用五类 AEC-CE，且只有一个空集 hard decision；
- `Count-Acc-5/SetSuccess@0.5` 已作为两个主指标，duplicate/full coverage 使用本文唯一 evaluator；
- 模板依赖和 phrase-label 泄漏诊断完整；
- 所有命令、配置、校准、预测和指标可从索引复现；
- 全部测试通过。

最终产物：

```text
artifacts/hmsa_joint/{seed}/{H1,H2,H3,F1,F2,R1,...}/
artifacts/hmsa_joint/joint_index.json
artifacts/relations/standard/relation_index.json  # only when relation losses enabled
artifacts/diagnostics/template_dependence.json
artifacts/diagnostics/phrase_leakage.json
artifacts/results/final_table.json
artifacts/results/final_table.md
```

`joint_index.json` 至少记录：

```text
variant, seed, resolved config
feature/baseline/adapter/data/phrase hashes
EventInterfaceV1 schema hash
relation index hash or explicit "disabled"
checkpoint and calibration hashes
prediction and metric paths
git commit, environment, command path
```

## 23. 最终研究结论的判定

研究主张限定为：

> HieA2M 将 HieA2G 的层次化语义对齐与自适应目标计数，从空间对象集合迁移到时序事件集合。

只有满足以下证据链才支持该主张：

1. `P0 -> H1 -> H2 -> H3` 显示三级 alignment 的增量和失败模式；
2. `G0-Threshold -> G0 -> G0-Con` 说明忠实 AGC classifier 与 count contrastive 的各自作用；
3. `G0 -> C1` 在同构 CountHeadV1 下说明 raw proposals -> event modes 的作用，`C1 -> C2` 说明 event-level count contrastive；
4. `C1 -> F1` 在同构同初始化 CountHeadV1 下说明 aligned events 的作用，`H3 -> F1` 说明联合计数的增量；
5. selected full coverage、duplicate、null 和 count 指标与 mAP 同时改善或呈现可解释权衡；
6. template-only probe、video shuffle 和 phrase leakage 检查排除明显捷径；
7. 结果在两个 seeds 上方向基本一致。

若某一层只改善其辅助指标而不改变最终 grounding，应如实报告，并用 `H3-aux-only` 与 gradient/score-correlation 结果判断它是否真正进入决策路径。不得仅凭 alignment loss 下降宣称迁移成功。
