# RL-Style On-Policy Distillation with Early-Decayed Forward KL

本文档说明当前 `on_policy_distill/train_on_policy_rl_local.py` 这条方法线的完整逻辑。

我们的实验已经表明，在数学推理蒸馏任务上：

1. 纯 `rKL` 的 on-policy distillation 更有利于得到尖锐、低熵、模式更集中的学生分布；
2. 这种 mode-seeking 特性更适合数学任务中“找到一条确定而稳定的正确推理路径”；
3. 但在训练初期，学生与教师分布差异过大时，单纯 `rKL` 容易让学生在错误区域内自我强化，从而训练不稳定，甚至进入局部最优；
4. 因此更合理的策略不是始终保留 `fKL`，而是在训练初期利用 `fKL` 把学生快速拉到教师分布附近，随后逐步去掉 `fKL`，回归纯 `rKL` 的标准 OPD。

基于这个假设，我们提出当前的训练方案：

- 训练初期：`rKL + fKL`
- 训练中期：`fKL` 线性衰减
- 训练后期：只保留 `rKL`

这可以看成一种“前期 distribution locating，后期 mode seeking”的蒸馏策略。

## 1. 问题背景

标准 on-policy distillation（OPD）的核心思想是：

- 学生先按自己的当前策略生成轨迹；
- 教师只对学生实际生成到的 token 给出逐 token 的密集监督；
- 用学生路径上的 `reverse KL` 更新学生，而不是像传统 KD 那样始终让学生直接拟合教师分布。

设：

- 学生旧策略为 $ \pi_{old} $
- 当前训练中的学生策略为 $ \pi_\theta $
- 教师策略为 `q`
- 输入 prompt 为 `x`
- 学生生成响应为 `y = (y_1, y_2, ..., y_T)`

则标准 OPD 使用的是学生路径上的 `reverse KL` 信号：

$$
\Delta_t^{(r)} = \log \pi_{\text{old}}(y_t \mid x, y_{<t}) - \log q(y_t \mid x, y_{<t})
$$

它的直观意义是：

- 如果学生在自己采到的 token 上比教师更自信，则 $ \Delta_t^{(r)} > 0 $
- 这说明学生可能在错误模式上过度集中
- 那么更新应压低这类 token 的概率

这类训练天然带有 mode-seeking 倾向，因此在数学任务中，常常比持续保留更“覆盖式”的 `fKL` 更有利。

## 2. 为什么不能始终保留 fKL

`forward KL` 的优势在于：

- 它能把学生快速拉向教师分布的支持集；
- 当学生还很弱、分布还很散时，这很重要。

但如果整个训练过程都持续保留强 `fKL`，就会带来两个问题：

1. 学生会持续被迫覆盖教师的更宽分布，而不是收缩到最有用的窄分布。
2. 对数学推理这类“正确路径通常很尖锐”的任务，持续的 `fKL` 可能抑制学生在后期形成低熵、确定性的正确推理模式。

因此，“始终 `rKL + fKL`”并不是数学蒸馏任务上的最优方案。更合理的方案是：

- 初期利用 `fKL` 稳定训练；
- 后期让 `rKL` 独占训练目标。

## 3. 方法概述

我们的方法由三部分组成：

1. `rKL` 主分支
   使用 RL / PPO 风格的 on-policy 训练框架，在学生 rollout 上构造 token-level advantage。

2. `fKL` 辅助分支
   在相同上下文下，从教师分布逐位置采样 token，并在这些 token 上给学生增加额外的 NLL 约束。

3. `fKL` 线性衰减调度
   初始时 `fKL` 权重较大，随后在训练前一段时间内线性减小到 0；在权重归零后，不再计算 `fKL` 对应的教师采样分支。

## 4. 训练流程

每个训练 step 的流程如下：

1. 从数据集中取一批 prompt。
2. 学生按当前策略 $ \pi_\theta $ 采样生成若干条回答。
3. 在这些学生轨迹上，计算：
   - 学生旧 logprob
   - 教师 logprob
4. 构造 `rKL` 的 token-level advantage。
5. 如果当前 step 的 `fKL` 权重仍大于 0：
   - 从教师分布逐位置采样 token；
   - 在这些 token 上计算学生的 NLL 辅助损失。
6. 用 PPO 或 importance sampling 目标更新学生。

因此，整个训练是一个标准 RL-style OPD 框架，只是额外挂接了一个可衰减的 `fKL` 辅助分支。

## 5. 公式

### 5.1 学生路径上的 reverse KL

学生先生成：

$$
y \sim \pi_{\text{old}}(\cdot \mid x)
$$

对每个位置 `t`，定义学生路径上的 `reverse KL` Monte Carlo 差值：

$$
\Delta_t^{(r)} =
\log \pi_{\text{old}}(y_t \mid x, y_{<t})
- \log q(y_t \mid x, y_{<t})
$$

## 5.2 rKL advantage

我们将其转成 RL 里的 token-level advantage：

$$
A_t^{(r)} = -\lambda_{\text{coef}} \, \Delta_t^{(r)}
$$

其中：

- $$ \lambda_{\text{coef}} $$ 对应代码中的 `kl_coef`
- 默认建议取 `1`

如果开启未来折扣，则进一步构造：

$$
\tilde A_t^{(r)} =
\sum_{u=t}^{T}\gamma^{u-t} A_u^{(r)}
$$

其中 $ \gamma $ 对应 `kl_discount`。

在标准 baseline 里，我们建议：

$$
\lambda_{\text{coef}} = 1, \quad \gamma = 0
$$

即不额外放大，也不做未来折扣。

### 5.3 PPO / IS 形式的 rKL 主分支

令：

$$
\rho_t(\theta) =
\exp\Big(
\log \pi_\theta(y_t \mid x, y_{<t})
- \log \pi_{\text{old}}(y_t \mid x, y_{<t})
\Big)
$$

如果使用 PPO，则 rKL 主分支为：

$$
L_t^{(r)}(\theta)
=
-\min\Big(
\rho_t(\theta)\tilde A_t^{(r)},
\text{clip}(\rho_t(\theta), 1-\epsilon_l, 1+\epsilon_h)\tilde A_t^{(r)}
\Big)
$$

如果使用 importance sampling，则为：

$$
L_t^{(r)}(\theta)
=
- \rho_t(\theta)\tilde A_t^{(r)}
$$

再乘上 rKL 主分支的权重 $ \lambda_r $ ：

$$
L_{t,\text{weighted}}^{(r)} = \lambda_r L_t^{(r)}
$$

通常默认：

$$
\lambda_r = 1
$$

### 5.4 教师路径采样的 fKL 辅助项

当 `fKL` 开启时，我们在相同上下文 $ (x, y_{<t}) $ 下，从教师分布采样：

$$
z_t \sim q(\cdot \mid x, y_{<t})
$$

并构造学生在教师采样 token 上的负对数似然：

$$
L_t^{(f)}(\theta)
=
- \log \pi_\theta(z_t \mid x, y_{<t})
$$

这项损失的作用不是让学生永远覆盖教师分布，而是在训练初期给学生提供“定位教师支持集”的能力。

### 5.5 逐步衰减的 fKL 权重

令总训练步数为 `T`，当前步数为 `s`，设 `p` 为 `fKL` 衰减结束所占的训练进度比例，例如 `p = 0.3`。

则我们定义：

$$
\lambda_f^{(s)} =
\begin{cases}
\lambda_f \left(1 - \frac{s/T}{p}\right), & s/T < p \\
0, & s/T \ge p
\end{cases}
$$

含义是：

- 第 0 步时，`fKL` 权重为初始值 $ \lambda_f $  
- 在训练前 `p` 比例的 step 中，线性下降
- 当训练进度达到 `p` 后，`fKL` 权重变为 0
- 之后训练完全退化为纯 `rKL` OPD

这正是本方法最核心的设计。

### 5.6 总损失

最终每个位置的总损失写为：

$$
L_t(\theta)
=
\lambda_r L_t^{(r)}(\theta) + 
 \lambda_f^{(s)} L_t^{(f)}(\theta)
$$

只在 continuation 且非 pad 的位置上累积：

$$
L(\theta)
=
\frac{1}{|\mathcal V|}
\sum_{t \in \mathcal V} L_t(\theta)
$$

其中 $ \mathcal V $ 是有效 token 位置集合。

## 6. 为什么这样做

这套设计的逻辑可以概括为两句话：

1. 前期先找对地方
2. 后期再往正确模式上收缩

更具体地说：

### 6.1 前期需要 fKL

训练前期，学生分布和教师分布往往差异很大。

如果此时只使用 `rKL`：

- 学生只会在自己已经采到的 token 上获得信号；
- 如果学生一开始就走偏了，`rKL` 可能只能在错误模态附近继续局部修补；
- 这样容易训练不稳定，也更容易进入局部最优。

而 `fKL` 的作用是：

- 让学生看到教师更可能采到哪些 token；
- 把学生快速拉回教师分布附近；
- 起到 warm start 和稳定训练的作用。

### 6.2 后期不能继续依赖 fKL

一旦学生已经进入教师支持集附近，继续保留强 `fKL` 会带来副作用：

- 学生持续被迫维持更宽的分布；
- 分布熵下降变慢；
- 在数学任务中，不利于形成尖锐、稳定、确定的推理路径。

我们的实验结果表明，后期更合理的目标是：

- 去掉 `fKL`
- 仅保留 `rKL`
- 让学生在正确区域内进一步 mode-seeking

这正是“early-decayed fKL”的核心动机。

## 7. 与其他方法的关系

### 7.1 与标准 OPD 的关系

当：

$$
\lambda_r = 1,\quad \lambda_f = 0
$$

时，本方法退化为标准的 RL-style OPD。

### 7.2 与持续 dual-KL 的关系

当：

$$
\lambda_f^{(s)} = \lambda_f \quad \text{for all } s
$$

即 `fkl_decay_until = 0` 时，本方法退化为“始终保留 fKL”的 dual-KL 训练。

因此当前方法可以看成是下面两者之间的一种更细化、也更有效的插值：

- 标准 OPD：只用 `rKL`
- 持续 dual-KL：始终 `rKL + fKL`

而我们主张的是：

- 只在前期使用 `fKL`
- 后期回归纯 `rKL`

## 8. 关键实验设计

当前推荐至少做以下对照：

1. `standard_opd`
   - 纯 `rKL`
   - 作为标准 baseline

2. `dual_kl_equal`
   - 初始 `fKL` 权重为 `1`
   - 在前 `30%` 训练内衰减到 `0`
   - 这是当前主方案

3. `dual_kl_always_on`
   - 初始 `fKL` 权重为 `1`
   - 始终不衰减
   - 用于验证“持续 fKL 是否反而不利于数学蒸馏”

4. `dual_kl_mild_f`
   - 初始 `fKL` 权重较小
   - 用于分析前期辅助强度的影响

5. `dual_kl_mid_f`
   - 中等初始 `fKL`
   - 用于观察从弱到强的趋势

实验结果表明，应观察到：

- `dual_kl_equal` 优于 `standard_opd`
  原因：训练前期更稳定，更容易找到教师分布

- `dual_kl_equal` 优于 `dual_kl_always_on`
  原因：后期去掉 `fKL` 后，学生可以更充分地在纯 `rKL` 下收缩到尖锐模式

## 9. 实现上的工程选择

当前实现还采用了以下工程策略：

- 学生 rollout 先生成，再落到 CPU，降低 GPU 峰值
- logprob 前向按 `lp_micro_batch` 分片
- 教师模型支持 ZeRO-3 分片推理
- `fKL` 权重一旦衰减到 0，直接跳过教师采样分支，不再浪费计算
- 支持本地模型路径和本地数据集路径，适配无网训练环境
- 支持 swanlab 的 `offline / online / disabled` 三种模式

## 10. 总结

这项工作的核心不是简单地把 `fKL` 叠加到 OPD 上，而是提出了一个更符合数学蒸馏任务特性的训练策略：

- 训练前期，用 `fKL` 找到教师分布；
- 训练后期，用纯 `rKL` 收缩到更尖锐、更稳定的正确推理模式。

实验已经表明，相比：

- 纯 `rKL`：本方法训练更稳定；
- 持续 `rKL + fKL`：本方法后期更适合数学推理蒸馏。

因此，本方法的本质可以概括为：

> `fKL` 用来“找对地方”，`rKL` 用来“收得更准”。
