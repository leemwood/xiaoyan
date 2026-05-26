# xiaoyan：一个带情景记忆与巩固重放的微型 Transformer 语言模型

**摘要：** 本文提出 **xiaoyan**，一个完全基于 NumPy 手动实现的微型 Transformer 语言模型。该模型在标准 Transformer 架构之上集成了情景记忆库与巩固重放机制，模拟生物大脑中"记忆编码—衰减遗忘—睡眠巩固"的认知过程。模型支持逐字符在线学习，每见一个字符即执行一次梯度更新。实验表明，模型能够在交替学习两条不同句子后同时保留两种模式，并在单侧强化训练时展现出符合预期的"遗忘—保留"不对称性。

**关键词：** Transformer，情景记忆，巩固重放，灾难性遗忘，在线学习

---

## 1. 引言

以 GPT 系列为代表的大语言模型通常在静态数据集上离线训练，一旦训练完成便难以持续适应新知识。当模型按顺序学习多个任务时，往往会出现**灾难性遗忘**——新知识覆盖旧知识，导致在早期任务上的性能急剧下降 [1]。

生物大脑则采用截然不同的策略：海马体负责快速编码情景记忆（episodic memory），而新皮层通过慢速的离线巩固（consolidation）将这些记忆整合为长时知识 [2]。受此启发，本文设计并实现了 xiaoyan 模型，在微型 Transformer 架构上引入：

1. **情景记忆库**：以「上下文序列 → 下一字符」三元组的形式存储每次学习经历，包含强度维度；
2. **记忆衰减与遗忘**：每次学习后全局衰减强度值，低于阈值的弱记忆被清除；
3. **巩固重放**：定期从记忆库加权采样进行重放训练，对抗灾难性遗忘。

整个系统仅依赖 NumPy，所有前向/反向传播均为手动推导实现，旨在以最简洁的代码验证上述机制的可行性。

## 2. 方法

### 2.1 模型架构

xiaoyan 采用简化版 Transformer 解码器架构，各层均为 NumPy 手动实现：

- **词嵌入 + 位置编码**：可学习词嵌入矩阵加上正弦位置编码；
- **单头自注意力**：缩放点积注意力（Scaled Dot-Product Attention），Q/K/V 线性投影后计算 softmax 权重，经输出投影得注意力输出；
- **前馈网络**：两层全连接（Linear → ReLU → Linear），隐藏层维度为 d_model 的 2 倍；
- **层归一化 + 残差连接**：每个子层前后均设残差连接与层归一化。

模型参数量约 9000（d_model=32, d_ff=64），输出端取序列末位隐状态投影至词汇表，经 softmax 得下一字符的概率分布。

### 2.2 反向传播

所有层的反向传播均从损失函数出发手动推导。以自注意力为例，梯度流如下：

$$
\begin{aligned}
\delta_{\text{context}} &= \delta_{\text{out}} \cdot W_O^T \\
\delta_V &= A^T \cdot \delta_{\text{context}} \\
\delta_A &= \delta_{\text{context}} \cdot V^T \\
\delta_{\text{scores}} &= A \odot (\delta_A - \sum(A \odot \delta_A)) \\
\delta_Q &= \delta_{\text{scores}} \cdot K / \sqrt{d_k} \\
\delta_K &= \delta_{\text{scores}}^T \cdot Q / \sqrt{d_k} \\
\delta_X &= \delta_Q \cdot W_Q^T + \delta_K \cdot W_K^T + \delta_V \cdot W_V^T
\end{aligned}
$$

其中 $A$ 为注意力权重矩阵，$\odot$ 为逐元素乘法。层归一化的反向传播使用了紧凑形式的化简公式。

### 2.3 情景记忆与遗忘

**记忆条目**定义为一个三元组 $m = (\mathbf{c}, t, s)$：
- $\mathbf{c}$：上下文字符索引序列（元组）
- $t$：真实下一个字符索引
- $s$：记忆强度（标量）

**插入与更新**：当模型遇到 $(\mathbf{c}, t)$ 时，若记忆中已有相同条目，则 $s \leftarrow s + 1.0$（强化重复模式）；否则新建条目，$s=1.0$。

**衰减**：每次学习后，所有记忆的强度乘以衰减因子 $\gamma \in (0,1)$：
$$s_i \leftarrow s_i \cdot \gamma$$

**遗忘**：衰减后，$s_i < \theta$ 的条目被永久删除。$\gamma=0.995$，$\theta=0.1$ 时，一条未被重新激活的记忆约在 460 步后被遗忘。

### 2.4 巩固重放

每隔 $K$ 步（本文取 $K=30$），从记忆库中按强度加权采样 $B$ 条记忆进行重放训练。采样概率正比于 $s_i$，这意味着高频激活的重要模式获得更多重放机会。重放时使用较小的学习率，避免干扰在线学习的主导梯度方向。

## 3. 实验

### 3.1 实验设置

- **数据集**：两条小写字母句子，句子 A = "hello world"，句子 B = "the cat sat"，共用 12 个字符
- **超参数**：d_model=32，d_ff=64，context_len=5，lr=0.03
- **训练**：交替训练 A 和 B 共 40 轮，随后仅训练 A 额外 30 轮
- **度量**：交叉熵损失，自回归生成质量，记忆库统计

### 3.2 结果与分析

**损失收敛**：训练初期平均损失约 3.3（接近随机猜测的 $\ln 13 \approx 2.56$），400 步后降至约 0.02，表明模型已近乎完美地记住了字符转移模式。

**交替训练生成**：经过 40 轮交替训练后，模型能够根据起始字符串生成混合了两条句子特征的文本：

| 起始 | 生成文本 |
|------|---------|
| `"hel"` | `"helo worldthe cat sathello worldt"` |
| `"the"` | `"thelo worldthe cat sathello world"` |
| `"ca"`  | `"cathello worldthe cat sathello w"` |

可见模型同时保留了"hello world"和"the cat sat"的循环模式。

**遗忘实验**：额外仅训练句子 A 30 轮后，生成结果明显偏向 A：

| 起始 | 生成文本（遗忘后） |
|------|-------------------|
| `"th"` | `"thello worldhello worldhello wor"` |
| `"ca"` | `"cathello worldhello worldhello w"` |

以 `"th"` 起始本应生成"the cat sat"的关联被"hello world"覆盖，而 `"ca"` 后仍保留了部分"cat"的残余——这与记忆强度分布一致：被持续激活的 A 模式强度高，B 模式虽衰减但未完全消失。

## 4. 结论

本文提出了 xiaoyan——一个在 NumPy 上从零实现、集成情景记忆与巩固重放的微型 Transformer 语言模型。实验表明：

1. 情景记忆衰减机制有效模拟了"遗忘曲线"；
2. 巩固重放能减缓灾难性遗忘，使模型在多任务交替训练中维持对旧知识的记忆；
3. 模型在 Termux（Android）CPU 环境下即可完整运行，验证了轻量级持续学习系统的可行性。

未来工作可考虑引入弹性权重巩固（EWC）[1] 或动态扩展网络结构，以在更大规模任务上验证这些机制。

## 参考文献

[1] Kirkpatrick, J., et al. "Overcoming catastrophic forgetting in neural networks." *PNAS*, 2017.

[2] McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. "Why there are complementary learning systems in the hippocampus and neocortex." *Psychological Review*, 1995.

[3] Vaswani, A., et al. "Attention is all you need." *NeurIPS*, 2017.

---

*项目地址：https://github.com/leemwood/xiaoyan*
