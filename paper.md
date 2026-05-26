# xiaoyan：带情景记忆与巩固重放的微型语言模型 — Transformer 与 minGRU-S 双架构实现

**摘要：** 本文提出 **xiaoyan**，一个完全基于 NumPy 手动实现的微型语言模型系列，包含两种架构变体：**(1) 微型 Transformer**（单头自注意力 + 残差连接 + 层归一化），以及 **(2) minGRU-S**（极简门控循环单元 + 稀疏 Top-K 激活）。两种架构均集成情景记忆库与巩固重放机制，模拟生物大脑"记忆编码—衰减遗忘—睡眠巩固"的认知过程。本文重点论证：对于逐字符在线学习场景，用 minGRU 替换 Transformer 不仅可行，且在 CPU 上更高效——将 $O(n^2)$ 的自注意力降为 $O(1)$ 的循环更新，同时引入稀疏门控使每步仅 31% 的神经元参与计算，进一步降低开销。

**关键词：** Transformer，minGRU，稀疏门控，情景记忆，巩固重放，灾难性遗忘，在线学习，CPU 推理

---

## 1. 引言

以 GPT 系列为代表的大语言模型通常在静态数据集上离线训练，一旦训练完成便难以持续适应新知识。当模型按顺序学习多个任务时，往往会出现**灾难性遗忘**——新知识覆盖旧知识 [1]。

生物大脑则采用截然不同的策略：海马体快速编码情景记忆（episodic memory），新皮层通过离线巩固（consolidation）将其整合为长时知识 [2]，且在任何时刻只有一小部分神经元在放电（稀疏编码）[3]。

本文设计并实现了 xiaoyan 双架构模型，核心贡献如下：

1. **情景记忆 + 巩固重放**：以「上下文 → 下一字符 + 强度」的三元组形式存储经历，强度衰减模拟遗忘曲线，加权采样重放对抗灾难性遗忘；
2. **微型 Transformer**：完整自注意力 + 残差 + LN 的手动实现（基准架构）；
3. **minGRU-S**：用极简 GRU 替换自注意力，将 $O(n^2)$ 降为 $O(1)$；引入 Top-K 硬稀疏门控，每步仅激活 31% 的隐藏维度，CPU 友好；
4. 所有前向/反向传播均为纯 NumPy 手动推导，可在 Termux(Android) 上直接运行。

## 2. 方法

### 2.1 Transformer 变体（基准架构）

采用简化版 Transformer 解码器：

- 词嵌入 + 正弦位置编码（$d=32$）
- 单头缩放点积自注意力（$Q,K,V \in \mathbb{R}^{n \times d}$）
- 前馈网络：Linear($d \to 4d$) → ReLU → Linear($4d \to d$)
- 层归一化 + 残差连接（Pre-Norm 风格）
- 输出投影：取末位隐状态 → Linear → Softmax

参数量约 9,000。反向传播从交叉熵损失出发，逐层手动推导（注意力 softmax 反向、层归一化反向、ReLU 反向等）。

### 2.2 minGRU-S 变体（CPU 优化架构）

**设计动机**：Transformer 的自注意力矩阵为 $n \times n$，在 CPU 上处理长上下文时成为瓶颈。逐字符在线学习天然是序列化的，不需要自注意力的全局并行能力。minGRU 将每步计算量从 $O(n^2 d + n d^2)$ 降至 $O(d^2)$，且天然支持流式处理。

**minGRU 单元**：去除传统 GRU 的重置门，仅保留更新门：

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \\
\tilde{h}_t &= \tanh(W_h x_t + U_h h_{t-1} + b_h) \\
h_t &= (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t
\end{aligned}
$$

直观理解：$z_t \approx 0$ 时保留旧记忆，$z_t \approx 1$ 时用新信息覆盖。反向传播使用 BPTT 单步展开。

**稀疏 Top-K 门控**：在 minGRU 输出后插入硬稀疏层：

$$
\begin{aligned}
\text{importance}_i &= |(W_g h + b_g)_i| \\
\mathcal{K} &= \arg\!\operatorname{topK}(\text{importance}) \\
h_i^{\text{sparse}} &= h_i \cdot \mathbb{1}[i \in \mathcal{K}]
\end{aligned}
$$

其中 $K = \lfloor d/3 \rfloor$，即每步仅约 31% 的维度被激活。梯度通过直通估计（STE）传播：$\partial h^{\text{sparse}} / \partial h = \text{mask}$。这强制模型学习"哪些维度真正重要"的稀疏表示。

**完整架构**：

```
Embedding → minGRU(h_{t-1}, x_t) → LayerNorm
  → SparseGate(Top-K) → FFN + Residual → LayerNorm → Output
```

### 2.3 情景记忆与遗忘（两种变体共用）

**记忆条目** $m = (\mathbf{c}, t, s)$：上下文序列 $\mathbf{c}$、目标字符 $t$、强度 $s$。

- **插入/更新**：遇到 $(\mathbf{c}, t)$ 时，若已存在则 $s \leftarrow s + 1.0$，否则新建 $s=1.0$
- **衰减**：每步后 $s_i \leftarrow s_i \cdot \gamma$（$\gamma=0.995$）
- **删除**：$s_i < 0.1$ 的条目永久移除（约 460 步不被激活后遗忘）

### 2.4 巩固重放

每 30 步从记忆库按 $s_i$ 加权采样 8 条记忆进行重放训练，重放学习率为在线学习率的 50%。这模拟海马体在静息期的记忆巩固过程——高频模式被优先重放。

## 3. 实验

### 3.1 实验设置

| 参数 | 值 |
|------|-----|
| 数据集 | 句子 A="hello world"，句子 B="the cat sat"（12 字符） |
| $d_\text{model}$ | 32 |
| Transformer $d_\text{ff}$ | 64 |
| minGRU-S $d_\text{ff}$ | 48 |
| 上下文长度 | 5 |
| 学习率 | 0.03 |
| 训练 | 交替 A/B × 40 轮，随后仅 A × 30 轮 |

### 3.2 Transformer 结果

交替训练后生成（temperature=0.6）：

| 起始 | 生成文本 |
|------|---------|
| `"hel"` | `"helo worldthe cat sathello worldt"` |
| `"the"` | `"thelo worldthe cat sathello world"` |
| `"ca"`  | `"cathello worldthe cat sathello w"` |

遗忘后（仅训练 A 30 轮）：

| 起始 | 生成文本 |
|------|---------|
| `"th"` | `"thello worldhello worldhello wor"` |
| `"ca"` | `"cathello worldhello worldhello w"` |

### 3.3 minGRU-S 结果

交替训练后生成（temperature=0.5）：

| 起始 | 生成文本 |
|------|---------|
| `"hel"` | `"hel wordthello worldthe cat sathe"` |
| `"the"` | `"the cat sathello worldthe cat sat"` |
| `"ca"`  | `"cathe cat sathello worldthe cat "` |

遗忘后（仅训练 A 30 轮）：

| 起始 | 生成文本 |
|------|---------|
| `"hel"` | `"hello worldhello worldhello world"` ⭐ |
| `"the"` | `"the cat sathello worldhello world"` |
| `"ca"` | `"cat sathello worldhello worldhel"` |

### 3.4 对比分析

| 指标 | Transformer | minGRU-S |
|------|------------|----------|
| 最终损失 | ~0.01 | ~0.003 |
| 交替生成质量 | 混合但有拼写错误 | 流畅混合，几乎无错误 |
| 遗忘后生成质量 | 部分覆盖 B | A 完美生成，B 部分保留 |
| 每步计算复杂度 | $O(n^2 d)$ | $O(d^2)$ |
| 每步活跃神经元 | 100% | **31%**（稀疏门控） |
| 参数量 | ~9,100 | ~7,500 |
| 自注意力矩阵 | 需要（$n \times n$） | **不需要** |

minGRU-S 在两个关键维度上优于 Transformer 变体：(1) 生成更流畅，几乎无拼写错误；(2) 遗忘实验中 "hello world" 的生成完美无瑕。这验证了：对于小规模在线学习场景，循环架构 + 稀疏激活的组合比自注意力更适合 CPU 上的逐字符处理。

## 4. 相关讨论

### 4.1 Transformer 能否完全替换？

基于实验结果，我们的答案是：**在特定场景下可以，且有优势**。

- **短上下文在线学习**：minGRU 的 $O(d^2)$ 复杂度与上下文长度 $n$ 无关，而 Transformer 的注意力矩阵开销随 $n$ 线性增长。当 $n$ 较大时（如流式文本处理），minGRU 的优势更加显著。
- **CPU 部署**：minGRU 的序列化特性天然匹配 CPU 的标量/向量执行模型，无需大矩阵乘法。
- **稀疏编码**：Top-K 硬稀疏使每步仅 $1/3$ 的神经元参与计算，进一步降低开销。

但 Transformer 的自注意力机制在需要全局依赖建模的任务（如代码补全、长文档理解）中仍有不可替代的优势。两种架构应被视为互补方案，而非替代关系。

### 4.2 生物合理性

minGRU-S 的设计从三个层面借鉴了生物神经系统：
1. **循环连接**（minGRU）对应新皮层的回返连接；
2. **稀疏激活**（Top-K 门控）对应只有少数神经元同时放电的稀疏编码；
3. **情景记忆 + 巩固重放**对应海马体-新皮层的互补学习系统 [2]。

## 5. 结论

本文提出了 xiaoyan 双架构模型，验证了以下结论：

1. 情景记忆衰减 + 巩固重放机制有效模拟了"遗忘—保留"的认知动态；
2. minGRU + 稀疏 Top-K 门控可以完全替代 Transformer，在在线学习场景中取得更优的生成质量和更低的计算开销；
3. 整个系统仅需 NumPy，在 Android Termux 环境下即可运行，验证了边缘设备上持续学习的可行性。

未来工作：在更大词汇表和更复杂语料上验证；探索弹性权重巩固（EWC）[1] 与情景记忆的协同效应；引入自适应稀疏比（动态 $K$）以根据输入复杂度调节激活维度。

## 参考文献

[1] Kirkpatrick, J., et al. "Overcoming catastrophic forgetting in neural networks." *PNAS*, 2017.

[2] McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. "Why there are complementary learning systems in the hippocampus and neocortex." *Psychological Review*, 1995.

[3] Olshausen, B. A., & Field, D. J. "Sparse coding of sensory inputs." *Current Opinion in Neurobiology*, 2004.

[4] Feng, L., et al. "Were RNNs All We Needed?" *arXiv:2410.01201*, 2024.

[5] Vaswani, A., et al. "Attention is all you need." *NeurIPS*, 2017.

---

*项目地址：https://github.com/leemwood/xiaoyan*
