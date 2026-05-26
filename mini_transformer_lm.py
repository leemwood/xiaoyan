# -*- coding: utf-8 -*-
"""
微型 Transformer 语言模型 —— 纯 NumPy 手动实现
=================================================
功能：
  1. 微型 Transformer（嵌入 + 位置编码 + 单头自注意力 + 前馈网络 + 层归一化）
  2. 逐字符在线学习（每见一个字符就执行一次梯度下降）
  3. 情景记忆库（记忆插入、强度增减、全局衰减、低强度遗忘）
  4. 巩固重放机制（加权采样一批记忆进行重放训练）
  5. 自回归文本生成（温度采样）

依赖：仅需 numpy，可在 Termux(Android) CPU 上直接运行

运行方式：
  python mini_transformer_lm.py
"""

import numpy as np
from collections import deque

# ============================================================
# 第一部分：基础工具函数
# ============================================================

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def relu(x):
    """ReLU 激活"""
    return np.maximum(0, x)


def cross_entropy(probs, target_idx):
    """计算交叉熵损失（单样本），返回 loss 值"""
    return -np.log(probs[target_idx] + 1e-12)


# ============================================================
# 第二部分：层归一化（Layer Normalization）
# ============================================================

class LayerNorm:
    """
    层归一化：对每个 token 的特征维度做归一化。
    公式：y = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)          # 缩放参数 (d_model,)
        self.beta = np.zeros(d_model)          # 平移参数 (d_model,)
        self.eps = eps
        # 累积梯度
        self.dgamma = np.zeros(d_model)
        self.dbeta = np.zeros(d_model)

    def forward(self, x):
        """
        x: (seq_len, d_model) 或 (d_model,) 的单个向量
        返回归一化后的输出
        """
        self.x = x                              # 保存输入，backward 时会用到
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) * self.inv_std  # 标准化后的值
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, d_out):
        """
        层归一化的反向传播。
        d_out: 上游梯度，形状与 forward 输出一致
        返回 dx: 对输入 x 的梯度
        内部累加 dgamma 和 dbeta
        """
        # 累加 gamma 和 beta 的梯度（在序列维度上求和）
        self.dgamma += np.sum(d_out * self.x_hat, axis=0)
        self.dbeta  += np.sum(d_out, axis=0)

        dx_hat = d_out * self.gamma                     # 对 x_hat 的梯度
        N = d_out.shape[-1]                              # 特征维度

        # LayerNorm 反向传播核心公式：
        # dx = (1/σ) * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        dx_hat_mean = np.mean(dx_hat, axis=-1, keepdims=True)
        dx_hat_x_hat_mean = np.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)
        dx = self.inv_std * (dx_hat - dx_hat_mean -
                             self.x_hat * dx_hat_x_hat_mean)
        return dx

    def zero_grad(self):
        """清零累积梯度"""
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)

    def apply_gradients(self, lr):
        """用 SGD 更新参数"""
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta


# ============================================================
# 第三部分：嵌入层 + 位置编码
# ============================================================

class EmbeddingWithPosition:
    """
    词嵌入 + 正弦位置编码（不可学习）。
    - W_emb: (vocab_size, d_model)  词嵌入矩阵
    - pos_enc: (max_len, d_model)  固定正弦位置编码
    """

    def __init__(self, vocab_size, d_model, max_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Xavier 初始化词嵌入矩阵
        scale = np.sqrt(2.0 / (vocab_size + d_model))
        self.W_emb = np.random.randn(vocab_size, d_model) * scale
        self.dW_emb = np.zeros_like(self.W_emb)

        # 预计算正弦位置编码（不可学习，不需梯度）
        self.pos_enc = self._sinusoidal_encoding(max_len, d_model)

    def _sinusoidal_encoding(self, max_len, d_model):
        """生成标准 Transformer 正弦位置编码"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]           # (max_len, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )                                                        # (d_model/2,)
        pe[:, 0::2] = np.sin(position * div_term)                # 偶数位 sin
        pe[:, 1::2] = np.cos(position * div_term)                # 奇数位 cos
        return pe

    def forward(self, token_ids):
        """
        token_ids: (seq_len,) 整数序列
        返回: (seq_len, d_model) 嵌入 + 位置编码
        """
        self.token_ids = token_ids
        seq_len = len(token_ids)
        # 词嵌入查找
        x = self.W_emb[token_ids]                                # (seq_len, d_model)
        # 加上位置编码（固定，不可学习）
        x = x + self.pos_enc[:seq_len]
        return x

    def backward(self, d_out):
        """
        反向传播：将梯度传播回词嵌入矩阵。
        d_out: (seq_len, d_model) 对嵌入输出的梯度
        由于位置编码不可学习，所有梯度直接流向 W_emb。
        使用 np.add.at 正确处理序列中重复 token 的梯度累加。
        """
        np.add.at(self.dW_emb, self.token_ids, d_out)
        # 嵌入层不需要向更早的层传播梯度（它是最底层）
        return None

    def zero_grad(self):
        self.dW_emb = np.zeros_like(self.W_emb)

    def apply_gradients(self, lr):
        self.W_emb -= lr * self.dW_emb


# ============================================================
# 第四部分：单头自注意力（Single-Head Self-Attention）
# ============================================================

class SingleHeadAttention:
    """
    单头缩放点积自注意力.
    - Q = X @ W_Q
    - K = X @ W_K
    - V = X @ W_V
    - Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    - Output = Attention @ W_O
    """

    def __init__(self, d_model):
        self.d_model = d_model
        self.d_k = d_model   # 单头所以 d_k = d_model
        scale = np.sqrt(2.0 / d_model)

        self.W_Q = np.random.randn(d_model, d_model) * scale * 0.5
        self.W_K = np.random.randn(d_model, d_model) * scale * 0.5
        self.W_V = np.random.randn(d_model, d_model) * scale * 0.5
        self.W_O = np.random.randn(d_model, d_model) * scale * 0.5

        # 梯度
        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)

    def forward(self, x):
        """
        x: (seq_len, d_model)
        返回: (seq_len, d_model)
        """
        self.x = x   # 保存输入供 backward 使用
        L = x.shape[0]

        # 线性投影
        self.Q = x @ self.W_Q                                   # (L, d_model)
        self.K = x @ self.W_K                                   # (L, d_model)
        self.V = x @ self.W_V                                   # (L, d_model)

        # 缩放点积注意力
        self.scores = self.Q @ self.K.T / np.sqrt(self.d_k)     # (L, L)
        self.attn_weights = softmax(self.scores, axis=-1)       # (L, L)

        # 加权求和
        self.context = self.attn_weights @ self.V                # (L, d_model)

        # 输出投影
        out = self.context @ self.W_O                            # (L, d_model)
        return out

    def backward(self, d_out):
        """
        自注意力的完整反向传播。
        d_out: (L, d_model) 对注意力输出的梯度
        返回 dx: (L, d_model) 对输入 x 的梯度
        内部累加 dW_Q, dW_K, dW_V, dW_O
        """
        L = d_out.shape[0]

        # --- 输出投影的梯度 ---
        # d_out = d(context @ W_O) → 链式法则
        d_context = d_out @ self.W_O.T                           # (L, d_model)
        self.dW_O += self.context.T @ d_out                      # (d_model, d_model)

        # --- 注意力输出 context = attn_weights @ V ---
        d_attn_weights = d_context @ self.V.T                     # (L, L)
        d_V = self.attn_weights.T @ d_context                     # (L, d_model)

        # --- softmax 的梯度 ---
        # 若 y = softmax(scores)，则 dscores = y * (d_attn - sum(y * d_attn, axis=-1))
        d_scores = self.attn_weights * (
            d_attn_weights -
            np.sum(self.attn_weights * d_attn_weights, axis=-1, keepdims=True)
        )                                                          # (L, L)

        # --- scores = Q @ K^T / sqrt(d_k) ---
        inv_sqrt_dk = 1.0 / np.sqrt(self.d_k)
        d_Q = d_scores @ self.K * inv_sqrt_dk                     # (L, d_model)
        d_K = d_scores.T @ self.Q * inv_sqrt_dk                   # (L, d_model)

        # --- Q = x @ W_Q, K = x @ W_K, V = x @ W_V ---
        self.dW_Q += self.x.T @ d_Q                               # (d_model, d_model)
        self.dW_K += self.x.T @ d_K
        self.dW_V += self.x.T @ d_V

        # --- 对输入 x 的梯度（三个投影路径之和）---
        dx = (d_Q @ self.W_Q.T + d_K @ self.W_K.T +
              d_V @ self.W_V.T)                                    # (L, d_model)
        return dx

    def zero_grad(self):
        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)

    def apply_gradients(self, lr):
        self.W_Q -= lr * self.dW_Q
        self.W_K -= lr * self.dW_K
        self.W_V -= lr * self.dW_V
        self.W_O -= lr * self.dW_O


# ============================================================
# 第五部分：前馈网络（Feed-Forward Network）
# ============================================================

class FeedForward:
    """
    两层全连接前馈网络：Linear → ReLU → Linear
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    """

    def __init__(self, d_model, d_ff):
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)

        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

    def forward(self, x):
        """
        x: (seq_len, d_model)
        返回: (seq_len, d_model)
        """
        self.x = x
        self.h1 = x @ self.W1 + self.b1                           # (seq_len, d_ff)
        self.a1 = relu(self.h1)                                    # (seq_len, d_ff)  ReLU 激活
        out = self.a1 @ self.W2 + self.b2                          # (seq_len, d_model)
        return out

    def backward(self, d_out):
        """
        FFN 反向传播。
        d_out: (seq_len, d_model) 对 FFN 输出的梯度
        返回 dx: (seq_len, d_model)
        """
        # --- 第二层：Linear2 ---
        d_a1 = d_out @ self.W2.T                                   # (seq_len, d_ff)
        self.dW2 += self.a1.T @ d_out                              # (d_ff, d_model)
        self.db2 += np.sum(d_out, axis=0)                          # (d_model,)

        # --- ReLU 反向 ---
        d_h1 = d_a1 * (self.h1 > 0)                                 # (seq_len, d_ff)

        # --- 第一层：Linear1 ---
        dx = d_h1 @ self.W1.T                                       # (seq_len, d_model)
        self.dW1 += self.x.T @ d_h1                                 # (d_model, d_ff)
        self.db1 += np.sum(d_h1, axis=0)                            # (d_ff,)

        return dx

    def zero_grad(self):
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

    def apply_gradients(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


# ============================================================
# 第六部分：微型 Transformer（组装所有层）
# ============================================================

class MiniTransformer:
    """
    微型 Transformer 模型结构：
      Embedding + PositionEncoding
      → Self-Attention → Add & LayerNorm
      → FeedForward    → Add & LayerNorm
      → Linear(output projection) → Softmax

    输入: (seq_len,) token 索引序列
    输出: (vocab_size,) 下一个 token 的概率分布
    """

    def __init__(self, vocab_size, d_model=32, d_ff=64, max_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 各子层
        self.embed = EmbeddingWithPosition(vocab_size, d_model, max_len)
        self.attn = SingleHeadAttention(d_model)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)

        # 输出投影层 (d_model → vocab_size)
        scale_out = np.sqrt(2.0 / d_model)
        self.W_out = np.random.randn(d_model, vocab_size) * scale_out
        self.b_out = np.zeros(vocab_size)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)

    def forward(self, token_ids):
        """
        前向传播。
        token_ids: (seq_len,) 上下文 token 索引
        返回: (vocab_size,) 预测下一个 token 的概率分布
        """
        # 嵌入 + 位置编码
        x = self.embed.forward(token_ids)                          # (seq_len, d_model)

        # --- 自注意力子层 + 残差连接 + 层归一化 ---
        attn_out = self.attn.forward(x)                            # (seq_len, d_model)
        self.res1 = x + attn_out                                   # 残差连接（缓存供 backward 用）
        x = self.ln1.forward(self.res1)                            # 层归一化

        # --- 前馈网络子层 + 残差连接 + 层归一化 ---
        ffn_out = self.ffn.forward(x)                              # (seq_len, d_model)
        self.res2 = x + ffn_out                                    # 残差连接
        x = self.ln2.forward(self.res2)                            # (seq_len, d_model)

        # --- 输出投影（仅用最后一个位置的隐状态预测下一个 token）---
        self.last_hidden = x[-1]                                   # (d_model,)
        logits = self.last_hidden @ self.W_out + self.b_out        # (vocab_size,)
        self.probs = softmax(logits)                                # (vocab_size,)
        return self.probs

    def backward(self, target_idx):
        """
        反向传播（完整手动实现）。
        target_idx: 真实下一个 token 的索引（整数）
        遍历所有层，从输出端向输入端累积梯度。
        """
        # --- 输出端：softmax + 交叉熵的梯度 ---
        # dL/d(logits) = probs - one_hot(target)
        d_logits = self.probs.copy()
        d_logits[target_idx] -= 1.0                                # (vocab_size,)

        # --- 输出投影层梯度 ---
        self.dW_out += np.outer(self.last_hidden, d_logits)        # (d_model, vocab_size)
        self.db_out += d_logits                                     # (vocab_size,)

        # 回传到最后一个隐状态的梯度
        d_last_hidden = d_logits @ self.W_out.T                     # (d_model,)

        # 构造整个序列的梯度矩阵（只有最后一个位置有非零梯度）
        L = self.res2.shape[0]
        d_x = np.zeros((L, self.d_model))
        d_x[-1] = d_last_hidden

        # --- 第二层层归一化反向 ---
        d_res2 = self.ln2.backward(d_x)                             # (L, d_model)

        # 残差连接梯度分流：一份流向 FFN 输入，一份跳过 FFN
        d_x_ln1_out = d_res2.copy()                                 # 残差路径梯度

        # --- FFN 反向 ---
        d_ffn_in = self.ffn.backward(d_res2)                        # (L, d_model)

        # 两条路径的梯度相加（残差连接的链式法则）
        d_x_ln1_out += d_ffn_in                                     # (L, d_model)

        # --- 第一层层归一化反向 ---
        d_res1 = self.ln1.backward(d_x_ln1_out)                     # (L, d_model)

        # 残差连接梯度分流
        d_embed_out = d_res1.copy()                                 # 残差路径梯度

        # --- 自注意力反向 ---
        d_attn_in = self.attn.backward(d_res1)                      # (L, d_model)

        d_embed_out += d_attn_in                                     # 总梯度

        # --- 嵌入层反向 ---
        self.embed.backward(d_embed_out)

    def zero_grad(self):
        """清零所有层梯度"""
        self.embed.zero_grad()
        self.attn.zero_grad()
        self.ln1.zero_grad()
        self.ffn.zero_grad()
        self.ln2.zero_grad()
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)

    def apply_gradients(self, lr, clip_val=5.0):
        """对所有层执行 SGD 更新，带梯度裁剪"""
        # 对所有可学习参数的梯度做裁剪（防止梯度爆炸）
        for grad in [self.dW_out, self.db_out]:
            np.clip(grad, -clip_val, clip_val, out=grad)

        self.embed.apply_gradients(lr)
        self.attn.apply_gradients(lr)
        self.ln1.apply_gradients(lr)
        self.ffn.apply_gradients(lr)
        self.ln2.apply_gradients(lr)
        self.W_out -= lr * self.dW_out
        self.b_out -= lr * self.db_out

    def train_step(self, token_ids, target_idx, lr):
        """
        单步训练：前向 → 计算损失 → 反向 → 更新参数
        token_ids: (seq_len,) 上下文
        target_idx: 目标 token 索引
        返回: loss 值
        """
        self.zero_grad()
        probs = self.forward(token_ids)
        loss = cross_entropy(probs, target_idx)
        self.backward(target_idx)
        self.apply_gradients(lr)
        return loss


# ============================================================
# 第七部分：情景记忆库（Episodic Memory）
# ============================================================

class EpisodicMemory:
    """
    情景记忆库 —— 存储 (上下文, 下一个字符, 记忆强度) 三元组。

    核心机制：
      1. **记忆插入** — 每次学习时，将 (context, target) 存入记忆库。
         若已存在相同条目，强度增加；否则新建条目，初始强度 = 1.0。
      2. **全局衰减** — 每学习一步后，所有记忆强度乘以衰减因子 (decay)。
         这模拟了"遗忘曲线"——不常被激活的记忆会逐渐消失。
      3. **遗忘删除** — 强度低于阈值 (threshold) 的记忆被永久删除，
         实现"记住重要信息、遗忘次要信息"。
      4. **加权采样** — 巩固重放时，按记忆强度加权随机采样一批条目。
         强度越高的记忆被重放的概率越大。
    """

    def __init__(self, decay=0.995, threshold=0.05, initial_strength=1.0):
        """
        decay: 每次学习后的强度衰减因子（越接近 1 遗忘越慢）
        threshold: 强度低于此值的记忆将被删除
        initial_strength: 新记忆或重复激活时增加的强度
        """
        self.memories = []          # 列表: [{'context': (tuple of ints), 'target': int, 'strength': float}, ...]
        self.decay = decay
        self.threshold = threshold
        self.initial_strength = initial_strength
        self.stats = {'added': 0, 'merged': 0, 'removed': 0}

    def add_or_update(self, context, target):
        """
        插入或更新一条情景记忆。
        context: tuple of ints — 上下文 token 序列
        target: int — 真实下一个 token 索引
        若 (context, target) 已存在，强度 += initial_strength（模式巩固）；
        否则，新增条目，初始强度 = initial_strength。
        """
        # 查找是否已有完全相同的条目
        for mem in self.memories:
            if mem['context'] == context and mem['target'] == target:
                mem['strength'] += self.initial_strength
                self.stats['merged'] += 1
                return
        # 新增记忆
        self.memories.append({
            'context': context,
            'target': target,
            'strength': self.initial_strength
        })
        self.stats['added'] += 1

    def decay_all(self):
        """
        全局强度衰减：
        所有记忆的强度 *= decay（衰减因子）。
        衰减后删除强度 < threshold 的弱记忆。
        这实现了"遗忘次要信息"——不被重复激活的模式逐渐消失。
        """
        for mem in self.memories:
            mem['strength'] *= self.decay

        before = len(self.memories)
        self.memories = [m for m in self.memories
                         if m['strength'] >= self.threshold]
        self.stats['removed'] += (before - len(self.memories))

    def sample_batch(self, batch_size):
        """
        从记忆库中加权随机采样一批记忆。
        采样权重 = 记忆强度（强度越高越可能被选中）。
        返回: (contexts, targets)
          contexts: list of tuples
          targets: list of ints
        用于巩固重放——重要记忆被重放的概率更高。
        """
        n = len(self.memories)
        if n == 0:
            return [], []

        strengths = np.array([m['strength'] for m in self.memories],
                             dtype=np.float64)
        probs = strengths / strengths.sum()
        k = min(batch_size, n)
        indices = np.random.choice(n, size=k, p=probs, replace=False)

        contexts = [self.memories[i]['context'] for i in indices]
        targets = [self.memories[i]['target'] for i in indices]
        return contexts, targets

    def size(self):
        return len(self.memories)

    def avg_strength(self):
        """平均记忆强度（用于监控）"""
        if not self.memories:
            return 0.0
        return np.mean([m['strength'] for m in self.memories])


# ============================================================
# 第八部分：在线训练器
# ============================================================

class OnlineTrainer:
    """
    在线训练器 —— 逐字符训练，集成情景记忆与巩固重放。

    训练流程（每见一个字符执行一次）：
      1. 用前 context_len 个字符作为上下文，前向传播预测下一个字符。
      2. 计算交叉熵损失，反向传播，SGD 更新参数。
      3. 将 (context, target) 写入情景记忆库。
      4. 对所有记忆执行全局衰减，删除弱记忆。
      5. 每隔 replay_interval 步，从记忆库加权采样一批记忆进行巩固重放。
         重放时对各记忆分别执行一次训练步骤（梯度累加后更新），
         防止模型遗忘之前学到的模式（对抗灾难性遗忘）。
    """

    def __init__(self, model, vocab, context_len=10,
                 lr=0.01, replay_interval=50, replay_batch=8,
                 mem_decay=0.995, mem_threshold=0.05):
        """
        model: MiniTransformer 实例
        vocab: dict {char → idx}, 必须包含 '<PAD>' → 0
        context_len: 上下文窗口大小
        lr: 学习率
        replay_interval: 每多少步执行一次巩固重放
        replay_batch: 每次重放采样的记忆数量
        mem_decay: 记忆衰减因子
        mem_threshold: 记忆遗忘阈值
        """
        self.model = model
        self.vocab = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.context_len = context_len
        self.lr = lr
        self.replay_interval = replay_interval
        self.replay_batch = replay_batch

        self.memory = EpisodicMemory(decay=mem_decay,
                                     threshold=mem_threshold)
        # 滑动窗口：维护最近见过的字符
        self.buffer = deque(maxlen=context_len)

        # 训练统计
        self.step = 0
        self.total_loss = 0.0
        self.loss_history = []

    def _ids_to_tuple(self, ids):
        """将 token id 列表转为 tuple（用作记忆库的 key）"""
        return tuple(int(i) for i in ids)

    def train_on_char(self, char):
        """
        逐字符在线训练核心函数。
        char: 当前见到的字符
        返回: loss（若有效），否则 None（上下文不足时跳过）
        """
        self.step += 1
        target_idx = self.vocab[char]

        # 如果上下文还不够长，先积累
        if len(self.buffer) < self.context_len:
            self.buffer.append(target_idx)
            return None

        # 构造上下文（最近的 context_len 个字符）
        context = np.array(list(self.buffer), dtype=np.int32)

        # === 第一步：在线学习（前向 + 反向 + 更新）===
        loss = self.model.train_step(context, target_idx, self.lr)
        self.total_loss += loss

        # === 第二步：更新情景记忆 ===
        # 将 (上下文序列, 下一个字符) 存入记忆库
        context_tuple = self._ids_to_tuple(context)
        self.memory.add_or_update(context_tuple, target_idx)

        # === 第三步：全局记忆衰减 ===
        self.memory.decay_all()

        # === 第四步：巩固重放（每隔 replay_interval 步） ===
        if self.step % self.replay_interval == 0 and self.memory.size() > 0:
            self._consolidation_replay()

        # 滑动窗口前进
        self.buffer.append(target_idx)

        return loss

    def _consolidation_replay(self):
        """
        巩固重放：
        从记忆库中按强度加权采样一批记忆，
        对每条记忆分别执行一次训练步骤（前向+反向），
        将梯度累加后统一更新参数。
        这模拟了海马体的记忆巩固过程——在"休息"时重放重要经历。
        """
        contexts, targets = self.memory.sample_batch(self.replay_batch)
        if not contexts:
            return

        self.model.zero_grad()

        for ctx, tgt in zip(contexts, targets):
            ctx_arr = np.array(list(ctx), dtype=np.int32)
            # 前向传播
            _ = self.model.forward(ctx_arr)
            # 反向传播（梯度累加到 model 内部各层的 d* 缓冲区）
            self.model.backward(tgt)

        # 统一应用累加梯度（用稍小的重放学习率，防止覆盖在线学习效果）
        replay_lr = self.lr * 0.5
        self.model.apply_gradients(replay_lr)

    def train_on_text(self, text):
        """
        在一段文本上逐字符训练。
        text: 字符串
        返回: losses 列表
        """
        losses = []
        for ch in text:
            if ch not in self.vocab:
                continue
            loss = self.train_on_char(ch)
            if loss is not None:
                losses.append(loss)
            # 每 100 步打印一次进度
            if self.step % 100 == 0 and losses:
                recent_avg = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
                print(f"  [步骤 {self.step:5d}] 近期平均损失: {recent_avg:.4f}  "
                      f"| 记忆数: {self.memory.size():4d}  "
                      f"| 平均强度: {self.memory.avg_strength():.3f}")
        return losses


# ============================================================
# 第九部分：文本生成
# ============================================================

def generate(model, start_text, vocab, idx_to_char,
             context_len=10, max_new_chars=50, temperature=0.8):
    """
    自回归文本生成（温度采样）。

    model: MiniTransformer 实例
    start_text: 起始字符串
    vocab: {char → idx} 映射
    idx_to_char: {idx → char} 映射
    context_len: 上下文窗口长度
    max_new_chars: 最大生成字符数
    temperature: 温度参数（<1: 更保守, >1: 更随机, ≈0: 贪婪）
    """
    pad_idx = vocab.get('<PAD>', 0)
    result = list(start_text)

    for _ in range(max_new_chars):
        # 取最后 context_len 个字符作为上下文
        recent = result[-context_len:]
        ids = [vocab.get(c, pad_idx) for c in recent]

        # 如果上下文不足 context_len，在左侧填充 PAD
        if len(ids) < context_len:
            ids = [pad_idx] * (context_len - len(ids)) + ids

        context = np.array(ids, dtype=np.int32)

        # 前向传播获取概率分布
        probs = model.forward(context)

        # 温度采样
        if temperature <= 0:
            # 贪婪解码
            next_idx = np.argmax(probs)
        else:
            # 温度缩放
            logits = np.log(probs + 1e-12)
            logits = logits / temperature
            scaled_probs = softmax(logits)
            next_idx = np.random.choice(len(probs), p=scaled_probs)

        next_char = idx_to_char.get(next_idx, '?')
        result.append(next_char)

    return ''.join(result)


# ============================================================
# 第十部分：演示示例
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  微型 Transformer 语言模型 —— 演示")
    print("  (纯 NumPy，手动反向传播，情景记忆，巩固重放)")
    print("=" * 60)

    # ------------------- 1. 准备数据与词汇表 -------------------
    # 使用两条小写字母句子交替训练，观察模型如何在在线学习中
    # 同时记住两条句子，以及遗忘机制如何工作。

    sentence_a = "hello world"      # 句子 A
    sentence_b = "the cat sat"      # 句子 B

    # 构建字符集（去重后排序）
    all_chars = sorted(set(sentence_a + sentence_b))
    print(f"\n词汇表大小: {len(all_chars)}")
    print(f"字符集: {all_chars}")

    # 构建映射（0 预留给 PAD）
    vocab = {'<PAD>': 0}
    for i, ch in enumerate(all_chars):
        vocab[ch] = i + 1
    idx_to_char = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # ------------------- 2. 初始化模型 -------------------
    print("\n初始化微型 Transformer (d_model=32, d_ff=64)...")
    model = MiniTransformer(vocab_size=vocab_size, d_model=32, d_ff=64, max_len=64)

    trainer = OnlineTrainer(
        model=model,
        vocab=vocab,
        context_len=5,           # 上下文窗口（较小以增加训练样本数）
        lr=0.03,                 # 学习率
        replay_interval=30,      # 每 30 步巩固重放一次
        replay_batch=8,          # 每次重放 8 条记忆
        mem_decay=0.995,         # 衰减因子（越接近1遗忘越慢）
        mem_threshold=0.1,       # 低于此强度的记忆被删除
    )

    # ------------------- 3. 交替训练 -------------------
    print("\n" + "=" * 60)
    print("  开始交替训练两条句子...")
    print(f"  句子 A: \"{sentence_a}\"")
    print(f"  句子 B: \"{sentence_b}\"")
    print("=" * 60)

    epochs = 40  # 交替轮数
    all_losses = []

    for epoch in range(epochs):
        print(f"\n--- 第 {epoch + 1} 轮 ---")

        # 训练句子 A
        print(f"  学习句子 A: \"{sentence_a}\"")
        losses_a = trainer.train_on_text(sentence_a)
        all_losses.extend(losses_a)

        # 查看记忆库状态
        print(f"  → 记忆数: {trainer.memory.size()}, "
              f"平均强度: {trainer.memory.avg_strength():.3f}")

        # 训练句子 B
        print(f"  学习句子 B: \"{sentence_b}\"")
        losses_b = trainer.train_on_text(sentence_b)
        all_losses.extend(losses_b)

        print(f"  → 记忆数: {trainer.memory.size()}, "
              f"平均强度: {trainer.memory.avg_strength():.3f}")

    # ------------------- 4. 生成展示 -------------------
    print("\n" + "=" * 60)
    print("  生成测试：用不同起始字符串生成文本")
    print("=" * 60)

    seed_texts = ["hel", "the", "he", "th", "ca"]

    for seed in seed_texts:
        print(f"\n起始 \"{seed}\" → ", end="")
        generated = generate(model, seed, vocab, idx_to_char,
                             context_len=5, max_new_chars=30, temperature=0.6)
        print(f"\"{generated}\"")

    # ------------------- 5. 记忆库状态 -------------------
    print("\n" + "=" * 60)
    print("  情景记忆库状态")
    print("=" * 60)
    print(f"  总记忆条目: {trainer.memory.size()}")
    print(f"  平均强度: {trainer.memory.avg_strength():.4f}")
    print(f"  新增: {trainer.memory.stats['added']}, "
          f"合并: {trainer.memory.stats['merged']}, "
          f"删除: {trainer.memory.stats['removed']}")

    # 展示前几条记忆
    print("\n  最强记忆 Top 5:")
    sorted_mems = sorted(trainer.memory.memories,
                         key=lambda m: m['strength'], reverse=True)
    for i, mem in enumerate(sorted_mems[:5]):
        ctx_str = ''.join(idx_to_char.get(c, '?') for c in mem['context'])
        tgt_str = idx_to_char.get(mem['target'], '?')
        print(f"    {i+1}. \"{ctx_str}\" → '{tgt_str}'  强度: {mem['strength']:.3f}")

    # ------------------- 6. 遗忘演示 -------------------
    print("\n" + "=" * 60)
    print("  遗忘与保留演示")
    print("=" * 60)
    print("  观察：如果某条句子的模式长期不被激活，")
    print("  其记忆强度会衰减直到被删除；而频繁出现的")
    print("  模式强度会增长。")

    # 只反复训练句子 A 多次（"冷落"句子 B）
    print(f"\n  现在只反复训练句子 A (\"{sentence_a}\") 30 轮...")
    for i in range(30):
        trainer.train_on_text(sentence_a)

    print(f"\n  遗忘后记忆数: {trainer.memory.size()}")
    print(f"  平均强度: {trainer.memory.avg_strength():.4f}")

    # 再次生成：句子 B 的模式应该减弱
    print("\n  遗忘后生成测试:")
    for seed in seed_texts:
        generated = generate(model, seed, vocab, idx_to_char,
                             context_len=5, max_new_chars=30, temperature=0.6)
        print(f"  \"{seed}\" → \"{generated}\"")

    print("\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)
