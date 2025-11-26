之前的代码rkl和fkl的计算方式和门控机制都有问题，使得我们的梯度优化目标并不是指向双向kl的最低值。这样的结果是loss虽然降低，但是rkl和fkl并不是一直降低的，而且lam的选择完全依赖于rkl和硬阈值，实则非常不合理。

以下内容描述一种对称的、基于 Monte Carlo 采样的“双向 KL 蒸馏损失”，作为我们的一种新的改进方案，包含：

1. rKL-MC（Reverse KL，学生采样）
2. fKL-MC（Forward KL，老师采样）
3. 对称 gating（偏差越大权重越大）
4. 最终 loss 汇总

具体想法是做出如下改进，以下提供一些公式和伪代码。

1. 代码的基本结构和原先维持不变
2. 在 rKL 部分：保留现有的mc采样版本：

    rkl_adv = (s_g_s - t_g_s).detach()
    rkl_loss_pos = - rkl_adv * s_g_s

3. 删除原来的 FKL（full/argmax），提出fkl的mc版本：

   * 在教师分布中进行采样得出 `y_t`，而不是用 argmax 选择
   * 计算 `fkl_loss_pos = -(t_g_t - s_g_t).detach() * s_g_t`

4. 增加 gating 标量：

   * `g_R = relu((s_g_s - t_g_s)*s_g_s)`
   * `g_F = relu((t_g_t - s_g_t)*t_g_t)`    
   这里的fkl公式和rkl是完全对称的，只是这样写没有梯度，仅仅是标量，但是正好可以用作归一化权重
   * 归一化成 lam_R, lam_F
   [
   \lambda_R = \frac{g_R}{g_R + g_F + \epsilon}
   ]
   [
   \lambda_F = \frac{g_F}{g_R + g_F + \epsilon}
   ]

5. 最终 loss：

   * `loss_pos_mb = lam_R * rkl_loss_pos + lam_F * fkl_loss_pos`

---