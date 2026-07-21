# Part 3 修订案：从 Temporal HMSA 到 GMR 约束完整性打分

最后更新：2026-07-21

## 0. 决策摘要

本计划正式放弃原来的依赖链：

```text
B0 -> P0 proposal-to-event adapter -> AEC -> Temporal HMSA
```

改为：

```text
B0 FlashVTG-GMR
  -> 直接作用于 dense clip / multi-scale proposal 的 typed constraint verification
  -> 原 FlashVTG ranking、boundary、NMS 与 GMR existence 输出
```

结论分三层：

1. **工程上可行**：HMSA 思想不需要 P0 或 AEC 才能训练；它可以直接增强
   FlashVTG 的 clip 表征、multi-scale proposal score 和 query-video existence。
2. **研究上值得做，但必须改题**：不能再声称“首次把 phrase/sentence hierarchical
   alignment 接到 FlashVTG”。DualGround 已经在 NeurIPS 2025 做了非常接近的通用
   VTG 方案。本文的独立问题应当是：**针对 GMR 的 in-domain null、single、multi
   场景，显式分解 action/team/full-query，并在不破坏 dense proposal coverage 的前提下
   做层次化组合验证。**
3. **能力边界要写清楚**：该路线不再显式预测 count，因此预期主要改善 proposal
   ranking、in-domain null rejection 和语义完整性；它不能保证解决 exact cardinality 或
   multi-moment complete-set prediction。若 multi-target recall 没有改善，不得用 existence
   或总体 G-mIoU 掩盖这一失败。

原计划 `03_temporal_hmsa_joint_experiments.md` 保留为历史预注册记录，不覆盖、不改写。
本文件是 Part 2 失败后的新研究修订案。

## 1. 为什么 Part 2 应判定为失败

Part 2 的工程执行已经完成，不是“实验尚未结束”。12 个正式记录均已生成并通过
replay，但 `part3_handoff=NOT_READY`。决定性证据来自预注册的 validation gate：

| Seed | Raw proposal oracle full coverage | P0 oracle-mode full coverage | Coverage drop | B0 val mAP | P0 val mAP |
|---|---:|---:|---:|---:|---:|
| 2024 | 0.9222 | 0.2667 | -0.6556 | 25.79 | 0.00 |
| 2025 | 0.9333 | 0.3444 | -0.5889 | 25.43 | 0.00 |

失败原因是结构性的，而不是训练不充分：P0-selection 只保留固定 greedy seeds；其 span
在训练中不可改变，因此约 0.59--0.66 的 coverage loss 无法靠继续训练 adapter 修复。
此外 C1/C2 在两个 seed 上的 multi-query `Count-Acc-5`、
`Selected-FullCoverage@0.5` 都为 0，count contrastive 也没有带来稳定收益。

因此锁定以下规则：

- 不把 P0、C1、C2 checkpoint 用作新方法初始化；
- 不修改、覆盖或事后调参挽救已有 Part 2 记录；
- 不把 Part 2 test 结果用于新方法的超参数选择；新方向的设计依据只引用已预注册的
  validation failure；
- dense proposal coverage 必须原样保留，任何新模块不得先把 proposal cloud 压成固定
  event slots。

证据文件：

- `artifacts/cardinality/part2_completion.json`
- `artifacts/cardinality/part2_report.md`
- `artifacts/cardinality/aggregate_metrics.json`

## 2. 论文调研结论与新颖性边界

### 2.1 HieA2G 真正可迁移的思想

[HieA2G](https://arxiv.org/abs/2501.01416) 的 HMSA 包含：

```text
word-object   -> masked text recovery
phrase-object -> phrase/query matching after bipartite assignment
text-image    -> global bidirectional contrastive alignment
```

重要的是，这三层 alignment 都直接改善 object-query 表征，再由原 detection heads 输出；
它们在结构上并不依赖 AGC。HieA2G 的消融也表明 HMSA 和 AGC 是可分的，但完整模型最好，
所以只保留 HMSA 是一个合理而非保证成功的假设。

对 Soccer-GMR 的合理映射为：

| HieA2G | 本修订案 | 可用监督 |
|---|---|---|
| word-object | action/team token-to-clip | phrase manifest + positive GT spans |
| phrase-object | factor/full-query-to-proposal | phrase manifest + proposal/GT tIoU |
| text-image | full-query-to-video/set | positive/null label + dense proposal statistics |

不能原样迁移的部分：

- 当前 Lighthouse 每个 2 秒 clip 只有全局 CLIP + SlowFast feature，没有 player/object token；
- Soccer-GMR 没有 phrase-box 标注；action/team 只共享同一 temporal target，不能声称获得了
  空间 object-level grounding；
- 保存的 CLIP token rows 已经上下文化。直接把 action/team rows 置零后做 text recovery，
  其余 token 仍可能携带被遮住语义，形成 masked-recovery 泄漏；
- null query 中可能存在 action 或 team 单因素，只是完整 conjunction 不成立。因此不能把
  null video 的所有 clips 标成 action-negative 或 team-negative。

### 2.2 与现有 VTG 工作的重叠

- [CG-DETR](https://arxiv.org/abs/2311.08835) 已在 clip-word correlation、dummy-token
  cross-attention 和 query-dependent saliency 上做了细粒度对齐；当前 FlashVTG backbone
  本身继承了这条数据流。因此“再加一个普通 clip-word attention”不构成贡献。
- [QD-DETR](https://arxiv.org/abs/2303.13874) 已证明 irrelevant query-video pair 对 saliency
  学习有帮助，但 Soccer-GMR 同视频/同动作查询很多，随机 batch shuffle 会制造 false
  negatives。本计划主实验只用显式 positive/null 或冻结的 known relations。
- [FlashVTG](https://arxiv.org/abs/2412.13441) 已通过 multi-scale Temporal Feature
  Layering 和 context-aware score refinement 改善 dense prediction；本计划应在其
  multi-scale points 上做语义验证，不重新发明 temporal pyramid。
- [DualGround](https://arxiv.org/abs/2510.20244) 已把 sentence-level 与 learned
  phrase-level 两条路径直接融合后送入 FlashVTG multi-scale pyramid，并报告 structured
  phrase alignment 的收益。这与“朴素 HMSA-only + FlashVTG”高度重叠，是必须包含的
  强基线和新颖性警戒线。
- [Sim-DETR](https://openaccess.thecvf.com/content/ICCV2025/html/Tang_Sim-DETR_Unlock_DETR_for_Temporal_Sentence_Grounding_ICCV_2025_paper.html)
  指出 global semantics 与 local localization 可能发生内部冲突。因此本计划使用分支隔离、
  零初始化 residual gate 和逐层 gate，而不是直接相加后全量微调。
- [Soccer-GMR](https://arxiv.org/abs/2605.02623) 的核心难点是 realistic in-domain null、
  multi-moment recall 与统一评估；这正是本计划相对普通单目标 VTG/DualGround 的差异来源。

### 2.3 最终可辩护的研究假设

> GMR 的主要假阳性来自只满足部分查询约束的片段，例如 action 正确但 team 错误。
> 在保留 FlashVTG dense proposal coverage 的条件下，学习 action、team/attribute 与
> full-query 三类 typed factors，并要求它们在同一 temporal proposal 上共同成立，再用同一
> constraint-complete score 同时驱动 ranking 与 abstention，可以提升 in-domain rejection
> 和 proposal ordering，同时不牺牲 multi-moment recall。

论文方法暂称 **Temporal Constraint-Complete Scoring (TCCS)**。HMSA 是研究动机，
constraint-complete scoring 才是可检验的核心机制；不能泛化地宣称首次提出 phrase-level
temporal grounding。

## 3. 固定输入、前置产物与禁区

### 3.1 只允许使用

```text
artifacts/features/f-lighthouse/feature_manifest.json
artifacts/manifests/standard/manifest_index.json
artifacts/baselines/baseline_index.json
artifacts/baselines/{2024,2025}/.../model_best.ckpt
artifacts/phrase_targets/standard/{train,val}.jsonl
```

数据现状：train/val 分别为 4,138/465 queries；action token alignment 覆盖 100%；team
alignment 在 WorldCup 子集覆盖 3,435/4,138 train 和 384/465 val；SportsMoments 的 team
分支必须 mask。train/val 均只有 17 个 template IDs，因此模板捷径诊断是阻塞项。

### 3.2 明确禁止

- 读取 `public_adapter_index.json`、P0/C1/C2 权重或 `EventInterfaceV1`；
- 在 main model 中构造 event modes、count head 或 count-based Top-N；
- 根据 test 选择 loss weight、阈值、phrase 数量或 unfreeze 深度；
- 把 phrase labels/team metadata 作为 inference-time model input；它们只可生成 train-time
  criterion targets；
- 把任意跨视频 query 当作负样本；
- 用 query-conditioned feature 完成“visual-only reconstruction”后声称无文本泄漏；
- 新增 hard proposal pruning。最终仍存储完整排序后的 proposal list，官方 evaluator 再取
  top-10/NMS 结果。

phrase supervision 由 criterion 持有独立、hash-pinned 的 `PhraseTargetStore`，按 canonical
query key 查询。不得把 phrase fields 加入会传给 `model.forward(targets=...)` 的 batch
metadata；model forward 只接收视频、query embeddings 和普通 masks。

### 3.3 启动时 Fail-Fast

每次 HT/HTP/HTPQ train/infer 必须检查：

```text
same-seed B0 checkpoint/hash == baseline_index
feature/data/phrase manifest hashes valid
enable_adapter == false
enable_aec == false
adapter_ckpt/public_adapter_index/count_calibration absent
video input dim == 2818 and semantic content dim == 2816
text dim == 512, runtime length == 40, clip_length == 2
eval_bsz == 1
```

variant registry、checkpoint partial-load/freeze、resume 和 `scripts/run_hiea2m.sh` 必须为 TCCS
增加独立路径。现有 Part 2 路由会要求 count calibration、绕过 baseline NMS，不能复用。
正式输出中不得出现 `event_*` 或 `pred_count_*`。

## 4. 新模型：Temporal Constraint-Complete Scoring

### 4.1 插入位置与恒等初始化

使用 B0 forward 中已有张量：

```text
raw_video_content : B x T x 2816  # [SlowFast || CLIP]，不含 TEF
query_tokens      : B x L x 512
query_mask        : B x L
video_emb         : B x T x 256   # CG-DETR/FlashVTG 融合后、pyramid 前
pyramid_points    : 各尺度 B x T_l x 256
out_class         : B x sum(T_l) x 1
out_coord         : B x sum(T_l) x 2
pred_exist_logit  : B
```

新增模块全部用 residual 方式接入：

```text
video_emb_h = video_emb + gamma_clip * Delta_clip
class_logit_h = out_class + gamma_prop * Delta_prop
exist_logit_h = pred_exist_logit + gamma_qv * Delta_qv
```

`gamma_clip/gamma_prop/gamma_qv=0` 初始化。关闭模块或全部 gate 为 0 时，B0 的 logits、
spans、existence 和最终 prediction 必须逐元素一致。不得为方便而改变 baseline 的 NMS、
坐标解码、existence threshold 或输出 schema。

实现时在 `input_vid_proj` 前保留纯视觉切片。当前 loader 会在 2816-D F-Lighthouse 内容后
拼接 2-D TEF，因此必须断言 `src_vid.shape[-1]==2818`，并令
`raw_video_content=src_vid[...,:2816]`；TCCS 的 visual branch 永远不能读取最后两维。

主 score refinement 作用于所有 valid multi-scale pyramid points，不只作用于排序后的
top-50；top-50 candidate interface 仅用于 oracle/机制诊断。eval 时必须为 pyramid 返回真实
mask。span 使用原连续坐标，主模型不做归一化坐标往返。

### 4.2 第一层：训练监督的文本 Role Router

inference 不能读取 action/team labels 或 token indices，因此用三个 learned role queries 从
运行时 query tokens 中预测 factor representations：

```text
A_action = softmax(RoleQuery_action · Key(query_tokens))
A_team   = softmax(RoleQuery_team   · Key([team_dummy; query_tokens]))
t_action_hat = sum_l A_action_l Value(query_token_l)
t_team_hat   = sum_l A_team_l   Value(query_token_l)
t_full_hat   = masked_mean(query_tokens[1:last_valid-1])
team_support = 1 - A_team[team_dummy]
```

phrase manifest 中已验证的 action/team token indices 只在 criterion 中监督 `A_action/A_team`；
SportsMoments 的 team availability target 为 0，使 router 选择 `team_dummy`。删除 phrase
manifest 后 inference 必须仍可运行且输出不变。

视觉语义支路只读取 `raw_video_content`，不读取 TEF、query_global 或 query-conditioned
`video_emb`：

```text
u_t = VisualProject(raw_video_content_t)
a_t = cosine(Wv u_t, Wa t_action_hat)
r_t = cosine(Wv u_t, Wr t_team_hat)
f_t = cosine(Wv u_t, Wf t_full_hat)
```

用 role-specific cross-attention 形成 `Delta_clip`，再通过零初始化 gate 回流到 pyramid 前的
`video_emb`。action/team/full 使用不同 projection；team contribution 乘模型自己预测的
`team_support`，而不是 inference metadata。

监督分两类：

1. **Full-query temporal map**：positive query 的 soft target 为每个 2 秒 clip 与任一 GT 的
   max-tIoU；null query 的 full-query target 全 0。使用 focal/BCE + positive ranking。
2. **Visual factor reconstruction**：只对 positive GT ROI 做 temporal pooling，用纯视觉 ROI
   分别重建 stop-gradient 的 action/team token embedding：

```text
L_factor_rec = 1 - cos(action_pred(ROI_visual), stopgrad(t_action_target))
             + team_available_target * (
                   1 - cos(team_pred(ROI_visual), stopgrad(t_team_target)))
```

这替代 HieA2G 的 masked text recovery，避免 contextual text row 泄漏。null query 不计算
factor reconstruction；未被关系索引证明的 clips 不作为 action/team negatives。这里的
`t_*_target` 由 criterion 从 phrase store 构造，不进入 model forward。

### 4.3 第二层：Typed Factor-to-Proposal Verification

对全部 valid multi-scale pyramid points 计算 proposal-level factor evidence，不先取 top-K：

```text
q_action_i = sim(P_action(p_i), t_action_hat)
q_team_i   = sim(P_team(p_i), t_team_hat)
q_full_i   = sim(P_full(p_i), t_full_hat)
```

组合分数必须能表达 conjunction 缺一不可。主配置使用固定 soft-min：

```text
q_no_team_i   = softmin([q_full_i, q_action_i], temperature=0.1)
q_with_team_i = softmin([q_full_i, q_action_i, q_team_i], temperature=0.1)
q_comp_i      = (1-team_support)*q_no_team_i + team_support*q_with_team_i
Delta_prop_i = w_full*logsigmoid(q_full_i)
             + w_action*logsigmoid(q_action_i)
             + team_support*w_team*logsigmoid(q_team_i)
             + w_comp*q_comp_i
```

主配置使用显式加性 conjunction，不用 unconstrained MLP 隐藏 factor 贡献；soft-min 温度在
所有 seed 固定，不在 test 调整。`Delta_prop` 只校准 class/ranking logit；主模型不新增
boundary residual，以便把收益归因于 semantic verification。

proposal target：

```text
y_i = max_j tIoU(span_i, GT_j)
positive: y_i >= 0.5
safe background: y_i < 0.1
unknown: 0.1 <= y_i < 0.5
```

主 `L_prop_full` 只监督 full-query relevance；unknown 不进入 denominator。action/team 的
跨事件 negatives 仅可来自第 5 节的冻结 relation index；若 index 未通过审计，主配置只做
positive pull，不伪造 negatives。

### 4.4 第三层：用同一证据统一 Ranking 与 Abstention

不再构造一个可以靠文本/视频全局先验独立解题的 QV head。把同一组 proposal-level
constraint-complete evidence 聚合成存在性证据：

```text
s_i          = class_logit_B0_i + gamma_prop * Delta_prop_i
z_support    = logsumexp_i(s_i) - log(number_of_valid_points)
support_stats = [z_support, max_i(s_i), mean(top5(s_i))]
Delta_exist  = MonotonicCalibrator(support_stats)
exist_logit_h = exist_logit_B0 + gamma_qv * Delta_exist
```

`MonotonicCalibrator` 对 support evidence 的权重约束为非负，避免高 proposal support 反而
降低 existence。训练目标为 `has_any_gt` 的 BCE；它不形成第二个 hard gate，最终仍只有
B0/GMR adapter 的 `pred_exist_score` 和固定 operating point。主实验不做 batch-wise
InfoNCE，因为同视频、同 action 或同 team 的 false negative 风险高。

### 4.5 总损失

保留 B0 的 MR、saliency、existence losses，新增：

```text
L_HT   = L_B0 + 0.2 L_role + 0.5 L_full_clip + 0.2 L_factor_rec
L_HTP  = L_HT + 0.5 L_prop_full + 0.2 L_relation_factor
L_HTPQ = L_HTP + 1.0 L_support_bce
```

`L_relation_factor=0` 是 relation index 不完整时的 fail-safe 主配置。上述权重先预注册；只允许
在 train-internal dev protocol 中做 `{0.5x,1x,2x}` sensitivity，选择后冻结，再看正式 val。

### 4.6 推理契约

```text
完整 valid pyramid points -> refined score 排序 -> B0 NMS@0.7
-> 保存连续 windows（evaluator 使用前 10）
pred_exist_score = sigmoid(exist_logit_h)
```

不存在 count、event modes、score threshold proposal deletion 或第二个 empty gate。主表的
existence operating point 仍为 0.4，eval batch size 固定 1。HMSA/TCCS variants 必须走单独的
baseline-NMS inference route；不能复用当前 Part 2 variant 的 hard-set selector，也不能要求
`adapter_ckpt` 或 `count_calibration`。

## 5. 冻结关系索引：防止 false negatives

仅从 train canonical manifest 和 train phrase targets 构建：

```text
artifacts/relations/hmsa_only/standard/
  factor_event_relations.jsonl
  query_video_relations.jsonl
  relation_index.json
```

每条 relation 必须记录 anchor/candidate query key、video、annotated window、action/team labels、
relation=`positive|negative|unknown`、理由、GT tIoU、输入 hash 和 builder version。

安全 negative 同时满足：

1. 来自同一视频内的 canonical annotated event；
2. 对当前 factor 的 label sets 明确无交集；
3. 与 anchor 任一 GT 的 tIoU `<0.1`。

不满足三项均为 unknown。不得从 val/test 构建训练关系，不得按文本相似度或当前模型分数
在线猜 negative。关系覆盖率不足不阻塞 HT/HTPQ，但必须自动关闭 factor-negative loss。

关系索引还必须显式构造四类机制审计对：

```text
correct query
same action / wrong team
same team / wrong action
both action and team wrong
```

主 counterfactual loss 不只要求 full-query score 降低，还要求共享因素保持、冲突因素选择性
降低：例如 action-swap 只应显著降低 `q_action/q_full/q_comp`，不应无差别压低 team score。
正确 query 的同一 proposal 上 `q_comp` 必须比 partial-match query 至少高固定 margin。若安全
pairs 数量或 action/team 覆盖不足，不进入正式训练；team 的 GT-ROI linear probe 若接近
chance，则放弃 team visual factor，改以 `HTPQ-no-team` 为主并把 team 失败写入结论。

## 6. 数据捷径和泄漏 Gate（训练前必做）

### 6.1 三个 text-only probes

只在 train 拟合、只在 train-internal dev 选超参、最终在正式 val 报告：

```text
template_id only -> null/non-null
action/team labels only -> null/non-null
query embedding only -> null/non-null
```

报告 AUROC、accuracy、F1，以及按 source 的结果。若 template-only AUROC 很高，不直接停止，
但所有主结果必须增加 template/source-stratified 指标，且模型必须通过 video-shuffle Gate。

额外对新增 TCCS head 做 train-only 5-fold `GroupKFold(template_id)`，并报告 macro-template
delta。由于 frozen B0 已经看过完整 train，这个检查只能证明“新增 head 的增益”不是简单模板
记忆，不能声称整个系统具有 template-OOD 泛化。若后续有固定 provenance 的 paraphrase、
keyword、verbose queries，再作为只读 stress set；不得根据 stress 结果调模型。

### 6.2 因果 sanity checks

- 同 query 随机替换 video 后，full alignment 与 existence 应显著下降；
- 同 video 交换 team/action phrase 后，factor score 应按关系方向改变；
- 固定 video、置换 query 时，visual-only ROI feature 必须逐元素不变；
- 固定 query、置换 video 时，visual-only ROI feature 必须改变；
- 删除 phrase targets 后 inference 仍可完整运行，证明 labels 未进入 forward；
- TEF 清零不应改变 factor reconstruction feature；
- null query 不得进入 positive-only factor reconstruction denominator。

对 normal 与 video-shuffle 条件计算 difference-in-differences：

```text
[(TCCS - B0)_normal - (TCCS - B0)_shuffle]
```

其 video-clustered bootstrap 95% CI 下界必须大于 0，否则新增收益很可能来自语言/模板先验。

任一检查失败则停止正式训练。

## 7. 训练协议

### 7.1 公平初始化

每个实验、每个 seed 均从对应 B0 checkpoint 独立初始化，不串行继承 HT/HTP/HTPQ：

```text
seed 2024 HT/HTP/HTPQ <- B0 seed 2024
seed 2025 HT/HTP/HTPQ <- B0 seed 2025
```

新增参数量控制组 `HR0`：使用与 TCCS 近似参数量和同一 score-feedback 接口，但不使用
role/alignment supervision，以排除收益仅来自额外容量。

### 7.2 主训练协议

默认协议：

1. 主实验始终冻结完整 B0，只训练 role router、visual/factor projections、residual gates、
   proposal score 和 monotonic support calibrator；B0 始终保持 eval mode，但不 detach TCCS
   输入所需计算图；训练前后 B0 tensor-state hash 必须不变；
2. 新模块 LR `3e-5`，训练 10--15 epochs；effective batch 固定 200；显存不足时用
   gradient accumulation 保持 effective batch，不改变
   optimizer step 语义；
3. 只在 `train-fit/train-dev` 上选择固定训练 epoch `E*`、loss sensitivity 和 learning rate；
4. 锁定 `E*` 后在完整 train 上从同一 B0 重新训练固定 `E*` epochs，不再用正式 val 做
   early stopping；existence operating point 主表固定为 0.4；
5. `HTPQ-unfreeze` 只能在 frozen-B0 主模型已通过全部 gate 后作为独立消融，不能替代主表。

若 gate 始终接近 0 或 branch gradient 为 0，先判实现失败，不用解冻 B0 掩盖问题。

### 7.3 数据切分纪律

正式 val 已参与 B0 checkpoint selection，且 Part 2 test 已被查看。为减少继续适配风险：

1. 从 train 按 `video_id` 做固定 85/15 `train-fit/train-dev`，同时保持 source、null/single/multi
   和 action 分布；同视频 queries 不得跨子集；
2. 所有 architecture choice、loss sensitivity、relation coverage threshold 只看 train-dev；
3. 正式 val 只作一次外层确认，不参与 checkpoint/epoch/结构选择；
4. Standard test 只运行最终锁定的一个 TCCS 配置；失败 ablation 不运行 test；
5. 因现有 test 已用于过去阶段诊断，论文中透明披露这一事实。新方法的所有选择必须能由
   train-dev + val 记录证明与 test 无关。

## 8. 实验矩阵

### 8.1 必要主实验

| ID | 说明 | Clip factor | Proposal constraint | Set support | 回流预测 |
|---|---|---:|---:|---:|---:|
| B0 | locked FlashVTG-GMR | | | | |
| HR0 | 参数量匹配 residual control | | | | ✓ |
| DG-lite | DualGround-style sentence/learned phrase 强基线 | ✓ | ✓ | | ✓ |
| HT | learned role-to-clip | ✓ | | | ✓ |
| HTP | HT + factor/full-query-to-proposal | ✓ | ✓ | | ✓ |
| HTPQ | HTP + shared-evidence abstention | ✓ | ✓ | ✓ | ✓ |
| HTPQ-aux | 与 HTPQ 相同 losses，但 residual gates 固定 0 | ✓ | ✓ | ✓ | 否 |
| HTPQ-no-team | 去掉 team factor | action/full | ✓ | ✓ | ✓ |

`DG-lite` 必须使用同一 F-Lighthouse、同一 B0、相近新增参数量和训练预算。若无法忠实复现
DualGround，则明确称为 DualGround-style control，不报告成官方 DualGround 数字。

### 8.2 必要消融

```text
learned phrase slots vs explicit action/team masks
full-query only vs +action vs +team vs action+team conjunction
soft-min conjunction vs simple mean/sum
query-conditioned visual reconstruction vs visual-only reconstruction
safe relation negatives vs positive-only
auxiliary-only vs prediction feedback
fixed B0 vs partial unfreeze
```

为复刻 HieA2G Table 6 的机制证据，主模型通过后再运行 `W-only/P-only/Q-only` 与
`Full-W/Full-P/Full-Q` leave-one-level-out；三项比较用 Holm 校正。只有 HTPQ 通过双 seed
promotion gate 后才运行全套消融；否则止损。

## 9. 指标、分组与统计

### 9.1 主指标

同时报告，禁止只选好看的指标：

```text
positive-query localization: mAP, mR+@1/5, positive mIoU@1/3/5
null rejection: AUROC, Rej-F1@0.4, Null-FPR, ECE, Brier
end-to-end: G-mIoU@1/3/5
multi-target: Multi-G-mIoU@1/3/5, FullCoverage@0.5, per-query recall
ranking mechanism: candidate score vs max-GT-tIoU Spearman
alignment: full temporal pointing, action/team phrase-to-region retrieval
```

分组至少包含 null/single/multi、source、action type、team available、duration、moment count、
inter-moment gap、template ID 和 query length。

正式 val 的 null/single/multi 为 210/165/90；总体指标很容易被 null 主导，因此 Gate D 要求
三组逐项报告，null 提升不能抵消 multi 明显退化。没有 count head 时，Count-Acc、MAE、
SetSuccess 只作为解码诊断，不作为 TCCS 学会 cardinality 的证据。

### 9.2 统计方法

- 每个 delta 以同一 query/video 的 paired comparison 计算；
- 用 video-clustered paired bootstrap 10,000 次，避免同一视频内 queries 被当作独立样本；
- 报告 95% CI、逐 seed 值、mean/std；
- inner-dev 用 seed 2024 作低成本 discovery；正式 val 使用两个 B0 seeds × 三个独立 TCCS
  module initializations，共 6 runs。它们是嵌套重复，不得伪装成 6 个独立 backbone seeds；
- 若准备论文主张，在成功后补 seed 2026。两个 seed 可以做方向判断，不足以支撑“稳定提升”
  的强结论。

## 10. 分阶段 Go/No-Go 与止损条件

### Gate A：实现恒等性

必须全部通过：

- gates=0 时 B0 tensors/predictions 逐元素一致；
- 完整 proposal 数量、raw oracle coverage 与 B0 一致；
- phrase labels 不进入 inference forward；
- masks/padding/nonfinite tests 通过；
- 新增 loss 在合法样本有非零有限梯度，在被 mask 样本严格为 0。
- TEF 隔离、SOT/EOT attention=0、Sports team denominator skip、multi-GT safe-negative、null
  batch、resume/replay 单测全部通过；
- B0 参数无梯度且训练前后 hash 不变；`HTPQ-aux` 的 prediction 与 B0 一致；完整 proposal
  list 与同一 NMS 路径可精确 replay。

失败：修实现，不训练。

### Gate B：机制 canary（train-dev，seed 2024）

至少满足：

- video shuffle 后 full-query score/AUROC 显著下降；
- action/team permutation 对 hard-negative 子集产生正确方向变化；
- factor/full score 与 max-GT-tIoU 的 Spearman 比 B0 candidate score 提高 `>=0.05`；
- positive temporal pointing 优于 uniform/random 基线；
- raw oracle multi coverage drop `<=0.01`。

失败：停止，不启动双 seed 正式训练。

### Gate C：内层 promotion（train-dev，seed 2024）

最多比较 6 个预注册配置。HTP/HTPQ 至少满足其中一个主要收益，并满足所有安全约束：

```text
主要收益：mAP +1.0，或 AUROC +1.5，或 G-mIoU@3 +2.0
安全约束：mR+@5 drop <=0.5；Multi-G-mIoU@3 drop <=0.5；
          raw oracle coverage drop <=0.01；Null-FPR 不恶化 >2.0
```

主要收益的 video-clustered paired bootstrap 95% CI 下界需大于 0。失败：停止正式 val、
第二个 B0 seed 和大规模消融。

### Gate D：外层双 seed confirmation（正式 val 只看一次）

最终 HTPQ 必须：

1. 两个 B0 seed 的关键 delta 同方向，且 6 个嵌套 runs 至少 5 个方向为正；
2. mean mAP 至少 `+1.0`；
3. mean G-mIoU@3 至少 `+2.0`；
4. mR+@5 和 multi coverage 不低于安全约束；
5. 明确优于 HR0；
6. 在 in-domain null/hard-factor-mismatch 子集优于 DG-lite，否则只能称为 DualGround transfer，
   不能称为 GMR-specific 方法贡献；
7. null 与 multi 分组都不退化，counterfactual score 阶梯和 token-deletion sensitivity 均按
   预期成立；若收益只来自 null，判定 grounding 假设失败；
8. mAP 与 G-mIoU@3 的 video-clustered paired-bootstrap 95% CI 下界均大于 0。

失败：不运行 test，结论写为 constraint-complete alignment 在当前 frozen features/数据规模下
没有可靠收益。

### Gate E：一次性 test

只有 Gate D 通过后：

- 冻结 code commit、config、checkpoint hash、threshold 和 experiment ID；
- 只运行最终 HTPQ（及已锁定 B0 reference）一次；
- 不因 test 结果返回修改模型；
- 保存 prediction replay、环境、manifest hashes 和完整分组指标。

## 11. 推荐执行顺序与成本控制

```text
P0. 冻结本修订案与 hashes
P1. 构建 train-fit/train-dev、relation index、text-only probes
P2. 实现 TCCS 接口、恒等/泄漏/梯度单测
P3. seed-2024 小规模 HT/HTP/HTPQ train-dev canary
P4. 过 Gate C 后锁定配置/E*，完成两 B0 seeds × 三个 module inits
P5. 冻结所有 artifacts，正式 val 只评估一次并执行 Gate D
P6. 过 Gate D 后补必要机制消融；准备投稿则补 seed 2026
P7. 锁定最终配置，一次性 Standard test
```

最大止损点设在 P3：在任何第二个 seed、完整消融或 test 之前，先证明 alignment 不只是
学模板，且不损害 dense proposal coverage/multi recall。

## 12. 最终论文叙事（仅在 Gate D 通过后）

建议题目方向：

```text
Constraint-Complete Temporal Grounding for Generalized Moment Retrieval
```

三项可主张贡献：

1. 针对 GMR 将 action/team/full-query 语义角色映射到 clip/proposal/video-set 三层验证；
2. 设计 null-aware、relation-safe 的训练规则，避免同域同动作 false negatives；
3. 在不压缩 dense proposals、不做显式 count 的条件下，联合改善 localization、rejection，
   并保留 multi-moment recall。

不得主张：

- 首次提出 phrase-level/structured temporal alignment；
- 完整复现 HieA2G object-level path；
- 解决 event cardinality；
- 仅凭总体 G-mIoU 提升就宣称 multi-moment retrieval 已解决。

## 13. 一句话执行准则

**从 B0 直接做 TCCS，保留全部 dense proposals；先证明 typed conjunction 比
DualGround-style 对齐更能处理 in-domain null/action-team mismatch，且不损害 multi recall，
再投入第二个 seed 和 test。**
