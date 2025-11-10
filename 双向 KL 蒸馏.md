# 双向 KL 通用蒸馏框架

------

## 1. Background

### 1.1 经典蒸馏：Forward KL vs Reverse KL

- 传统 KD 主流是 **forward KL / 交叉熵**：用老师分布 (p_T) 当 “真实分布”，学生 ($p_\theta$)拟合它：
   $$
   L_{\text{FKL}} = D_{\mathrm{KL}}(p_T\Vert p_\theta)
   $$
   这类目标被很多博客和论文归纳为 **“zero-avoiding / mode-covering”**：在 teacher 高概率区域，student 必须给出比较高的概率，整体分布变得**更宽、更能覆盖所有峰**。

- 最近一批工作（ on-policy distillation 博客）强调 **reverse KL / on-policy distillation**：学生根据自己的策略生成轨迹，老师对这些轨迹给出 token-level 指导，用 **反向 KL $(D_{\mathrm{KL}}(p_\theta\Vert p_T))$** 或等价形式更新。([Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/?utm_source=chatgpt.com))
  - 优点：**on-policy（学生自己分布）+ dense teacher signal**，能缓解 off-policy KD 的分布漂移问题；
  - 缺点：反向 KL 是 **“zero-forcing / mode-seeking”**，容易把学生分布收得很窄，探索能力下降。([mlpod.com](https://www.mlpod.com/732.html?utm_source=chatgpt.com))

### 1.2 现有方法的共性问题

1. **只用 forward KL（off-policy KD）：**
   - 学生在 teacher 采样的状态上学得很好，但在“自己实际生成的状态”上可能行为不佳 → 分布 shift。
   - 分布虽然宽，但有时会带来更多 “奇怪但 teacher 稍微给过一点概率” 的输出。
2. **只用 reverse KL（on-policy distillation）：**
   - 学生在自己的分布上对齐 teacher，有利于实际使用时的稳定性。
   - 但长期看会变成强烈的模式聚焦：分布塌缩、多样性降低，对没见过的输入泛化能力弱。

**Motivation：**

> 有没有可能 **按难度/分布差异在 per-step 级别切换**：
>
> - 能搞定的步 → 用 on-policy + reverse KL 微调，保持稳定；
> - 明显搞不定的步 → 请老师直接写出来，再用 forward KL 抄答案、补盲区？

------

## 2. 核心 insight & 理论依据

### 2.1 正向 / 反向 KL 的“行为模式”差异

根据 KL 分析博客和一些蒸馏综述：([mlpod.com](https://www.mlpod.com/732.html?utm_source=chatgpt.com))

- **正向 KL    ($D_{\mathrm{KL}}(p_T\Vert p_\theta)$)**
  - 在 (p_T(z)) 大的地方，如果 (p_\theta(z)) 太小会被强烈惩罚；
  - 所以它驱动 student 去**覆盖 teacher 的所有高概率区域** → **mode-covering，zero-avoiding**。
- **反向 KL    ($D_{\mathrm{KL}}(p_\theta\Vert p_T)$)**
  - 在 (p_T(z)=0) 且 ($p_\theta(z)$>0) 时，散度趋向无穷大 → student 不敢把概率放在 teacher 认为“不可能”的区域；
  - 所以它鼓励 student 找一个“**单峰或少数模态**”去贴 teacher，避免在 teacher 的低概率区放 mass → **mode-seeking，zero-forcing**。

直观图就是：真实分布双峰，用单高斯拟合时，

- forward KL 会学一个“中间较宽的高斯”；
- reverse KL 会学其中一个峰。

### 2.2 用 “KL 大小” 做 per-step 难度 / 置信度指示

我们的状态是某个 step 的上下文 (s_t)，两边分布：

$$
 p_\theta(\cdot|s_t),\quad p_T(\cdot|s_t)
$$
**关键 insight：**
 “**反向 KL 大小**”可以作为一个**“学生在该状态下是否还在 teacher 支持集附近”**的指标：

- 若 ($D_{\mathrm{KL}}(p_\theta\Vert p_T)$) 很小：
   表示学生当前策略已经把概率质量主要放在 teacher 支持集上，**大概率已经在“合理模态附近”**——
   → 适合 on-policy 训练，用 reverse KL 在学生采样的轨迹上做微调即可。
- 若 ($D_{\mathrm{KL}}(p_\theta\Vert p_T)$) 很大：
   表示学生大量概率被放在 teacher 几乎不认可的区域，**已经跑偏到奇怪模态上**——
   → 这时候继续在学生的分布上做 reverse KL 只会强行“把这个错误模态收窄”；
   → 更合理是：**让 teacher 重写/重采样 step**，再用正向 KL off-policy 蒸馏，强制 student 覆盖老师正确模态。

从变分推断/信息论角度看，这等价于在每个 step 上做一个局部的：

> **“I-projection（reverse KL） vs M-projection（forward KL）” 切换**

- I-projection：给定 student 支持集，往 teacher 投影 → 模式聚焦；
- M-projection：给定 teacher 支持集，往 student 投影 → 覆盖全部模式。([HsinJhao's Blogs](https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/?utm_source=chatgpt.com))

### 2.3 Piecewise 目标的合理性

定义：
$$
 D_t = D_{\mathrm{KL}}\big(p_\theta(\cdot|s_t)\Vert p_T(\cdot|s_t)\big),
 \quad G_t = \mathbf{1}[D_t < \tau]
$$
在 step 级别上：

- 当 (G_t=1)（**easy / on-policy 区域**）：
  - 使用 **reverse KL**：
     $$
     L_{\text{on}}(t)=D_{\mathrm{KL}}(p_\theta\Vert p_T)
     $$
  - 稳定 & 不破坏已有探索模式。
- 当 (G_t=0)（**hard / off-policy 区域**）：
  - teacher 重写该 step：($y^T\sim p_T(\cdot|s_t)$)；
  - 使用 **forward KL**：
     $$
     L_{\text{off}}(t)=D_{\mathrm{KL}}(p_T\Vert p_\theta)
     $$
  - 强制 student 在 teacher 支持集上“补课”，避免继续沿着错误模态狂奔。

整体目标：
$$
L = \mathbb{E}_t\big[ G_t L*{\text{on}}(t) + (1-G_t)L_{\text{off}}(t)\big]
$$
**理论上它做了两件事：**

1. 在 student 已经不太偏的时候，用 on-policy reverse KL 减少 distribution shift，兼顾稳定性（和 On-Policy Distillation 的动机一致）。([Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/?utm_source=chatgpt.com))
2. 在 student 明显偏的时候，用 forward KL 强制把它的质量“拉回 teacher 支持集”，防止模式坍缩在坏模态上。

------

## 3. 方案细化：训练算法（Method）

### 3.1 数据与状态粒度

这篇是“通用蒸馏”，不一定要讲 step-level reasoning，但可以：

- 要么按 **token** 粒度（语言建模）；
- 要么按 **step** 粒度（例如数学推理中的每一步推导）。

记通用为 “step” 即可，写论文时可根据任务具体化。

### 3.2 单个 step 的训练流程

对每个训练样本 & step (t)：

1. 学生前向：得到 logits → softmax → ($p_\theta(\cdot|s_t)$)；

2. 老师前向：得到 ($p_T(\cdot|s_t)$)（可以只算本 step 的 logits）；

3. 计算 **反向 KL 差异：**
    $$
    D_t = D_{\mathrm{KL}}(p_\theta\Vert p_T)
    $$
    
4. 若 ($D_t < \tau$)：
   - **on-policy 模式（Regime 1）**：
      $$
      L_t = D_{\mathrm{KL}}(p_\theta\Vert p_T)
      $$
   - 可以理解为 on-policy distillation 框架的一个实例。([Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/?utm_source=chatgpt.com))
   
5. 若 ($D_t \ge \tau$)：
   - **off-policy 模式（Regime 2）**：
     
     - 可选：
       - 用 teacher 在同一状态下做一次 **argmax / nucleus** 重采样得到 (y^T)；
       
     - loss 用 **forward KL / cross-entropy**：
        $$
        L_t = D_{\mathrm{KL}}(p_T\Vert p_\theta)
        $$

### 3.3 软 gating 版本（optional）

为了避免硬阈值带来的不连续，可以用：

$$
 \lambda_t = \sigma\big(\alpha(\tau - D_t)\big)
$$
然后：
$$
L_t = \lambda_t * D_{\mathrm{KL}}(p_\theta\Vert p_T)
 + (1-\lambda_t)* D_{\mathrm{KL}}(p_T\Vert p_\theta)
$$
这里的区别是：

- 采样还是按一开始的设想分 student / teacher 来拿样本
- 但 loss 权重可以平滑从 reverse 过渡到 forward

------

## 4. 实验设计

### 4.1 实验目标

重点不在 “绝对分数有多高”，而在于：

- 相比单用 forward / reverse KL：
  - 我们的**组合策略在**：
    - 任务表现（accuracy / pass@k）
    - 多样性（entropy / distinct-n）
    - 稳定性（训练是否容易发散 / mode collapse）
       上取得更好的综合折中。

### 4.2 任务和数据集选择

1. **语言建模 / general LLM 基准：**
   - 一个开放领域数据集（如 open-web text 子集）做 perplexity；
   - 一个 instruction-following 数据集（如分享类指令集合）做 win-rate / GPT-judge。
2. **推理类任务：**
   - GSM8K / MATH/ AIME，小规模即可；
   - 评估 chain-of-thought 的质量（正确率 + 步数）。

这样可以展示：

- 在简单语言建模上也有收益（说明它不是特例）；
- 在 reasoning 上更明显（因为 student 容易跑到奇怪模态，gating 训练更有用）。

### 4.3 模型与基线

- 老师：一个 7B / 32B 级别 model；
- 学生：1.5B / 3B 级别。

对比方法：

1. **FKL-KD**：纯 forward KL 蒸馏（标准 KD）；
2. **RKL-OPD**：纯 reverse KL 的 on-policy distillation（参考 Thinking Machines 的设定）；([Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/?utm_source=chatgpt.com))
3. **Ours-DualKL-Gated**：你提出的 per-step gated dual KL。
4. （可选）SFT 基线：直接在 teacher 生成的数据上做监督微调。

### 4.4 指标 & 分析

- **主指标：**
  - 各任务 test accuracy / pass@1；
  - PPL / NLL；
  - 不同 difficulty bucket 下的表现（容易 vs 困难题）。
- **分布相关指标：**
  - 学生输出分布熵（平均 entropy）；
  - 多样性指标：distinct-n、不同样本间的 lexical variety；
  - 对 out-of-domain 测试的性能 drop。
- **训练行为 & 稳定性：**
  - loss 曲线（有无震荡）；
  - 模式塌缩现象（某些 prompt 下总是同一种回答的比例）。

**预期现象：**

- FKL-KD：
  - 表现不错，但在真正 on-policy 使用时可能略逊 / 有奇怪答案；
  - 分布偏宽，多样性好。
- RKL-OPD：
  - 表现稳定，但 diversity / OOD 能力显著下降，有模式塌缩倾向。
- Ours-DualKL-Gated：
  - 在主指标（accuracy/pass@1）上 ≥ 两端单独方案；
  - 熵和多样性介于两者之间（不过分收窄也不过分发散）；
  - 训练更稳，对高难样本有更小的性能 drop。

### 4.5 消融

- 不同 ($\tau$) / 不同 gating function（硬阈值 或 soft sigmoid 或 step score）；
- 只用 forward / 只用 reverse / 双向 gated；
- 对困难样本子集（例如高难数学题）的行为：
  - 看 gating 频率（多少 step 判为 hard）；
  - 看 teacher 重写次数与性能提升关系。

------

## 5. 其他需要提前想好的要素

1. **计算成本**：

   - On-policy 部分要算双边分布，但只在学生轨迹上；
   - Off-policy 部分要 teacher 重新采样 step（但只在 KL 大的部分 step 上）；
   - 可以给出 “teacher 调用次数 / step 比例” 的统计，说明成本可控。

2. **实现细节**：

   - KL 计算时只在 top-k / top-p 截断后的分布上算，减少噪声；
   - 对 (D_t) 做 clip 或 smooth，防止极端 outlier。

3. **安全和有害样本的处理（如果想到 alignment 向）：**

   - 可以讨论：在明显高风险输出区域，KL 会非常大，框架会更倾向 teacher 接管，这对安全性有好处（但这可能是第二篇/后续工作扩展）。

4. **写作上的 framing：**

   > “我们从 on-policy distillation 与 forward/reverse KL 的差异出发，提出了一种 **per-step difficulty-aware dual KL 框架**，实践证明在蒸馏 LLM 时兼顾了稳定性、多样性与任务性能。”

5. **后续可以在此基础上增加RPO，做 step-level speculative reasoning。**