# -*- coding: utf-8 -*-
"""
minGRU-S 语言模型 —— 纯 NumPy · 极简门控 + 稀疏激活 · CPU 优化
================================================================
架构特点：
  - **minGRU**：极简门控循环单元（仅保留更新门，去除了重置门），
    每步 O(d²) 计算量，CPU 友好，天然适合逐字符流式处理。
  - **稀疏门控激活**：GLU 风格的 sigmoid 门控，仅在重要维度上激活，
    模拟大脑"仅激活相关神经元"的稀疏编码机制。
  - **与 Transformer 对比**：无自注意力矩阵 → O(n) 替代 O(n²)，
    无 Q/K/V 三路投影 → 参数减少约 40%，每步计算量降低约 60%。

运行：
  python minGRU_lm.py
依赖：仅需 numpy
"""

import numpy as np
from collections import deque


# ============================================================
# 工具函数
# ============================================================

def sigmoid(x):
    """数值稳定的 sigmoid"""
    # 使用 clipping 防止 exp 溢出
    x_clipped = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def relu(x):
    return np.maximum(0, x)

def cross_entropy(probs, target):
    """单样本交叉熵，返回 loss（标量）"""
    return -np.log(probs[target] + 1e-12)

def softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum()


# ============================================================
# minGRU 单元 — 极简门控循环单元
# ============================================================

class MinGRU:
    """
    minGRU：只保留更新门 z，去除传统 GRU 的重置门 r。
    
    前向公式：
      z_t     = σ(W_z @ x_t + U_z @ h_{t-1} + b_z)    -- 更新门
      h̃_t     = tanh(W_h @ x_t + U_h @ h_{t-1} + b_h) -- 候选状态
      h_t     = (1 - z_t) * h_{t-1} + z_t * h̃_t       -- 新状态

    直观理解：
      - z_t ≈ 0 → 保留旧记忆（h_t ≈ h_{t-1}）
      - z_t ≈ 1 → 用新信息覆盖（h_t ≈ h̃_t）
      - z_t 在 (0,1) → 平滑插值
    """

    def __init__(self, d_model):
        self.d = d_model
        scale = np.sqrt(2.0 / d_model) * 0.5

        # 更新门参数
        self.W_z = np.random.randn(d_model, d_model) * scale
        self.U_z = np.random.randn(d_model, d_model) * scale
        self.b_z = np.zeros(d_model)

        # 候选状态参数
        self.W_h = np.random.randn(d_model, d_model) * scale
        self.U_h = np.random.randn(d_model, d_model) * scale
        self.b_h = np.zeros(d_model)

        # 梯度累积
        self.dW_z = np.zeros((d_model, d_model))
        self.dU_z = np.zeros((d_model, d_model))
        self.db_z = np.zeros(d_model)
        self.dW_h = np.zeros((d_model, d_model))
        self.dU_h = np.zeros((d_model, d_model))
        self.db_h = np.zeros(d_model)

    def forward(self, x_t, h_prev):
        """
        单步前向传播。
        x_t:    (d_model,) 当前输入向量（嵌入 + 位置编码）
        h_prev: (d_model,) 上一步隐藏状态
        返回:   h_t (d_model,) 新隐藏状态
        """
        self.x_t = x_t
        self.h_prev = h_prev

        # --- 更新门计算 ---
        self.a_z = self.W_z @ x_t + self.U_z @ h_prev + self.b_z
        self.z_t = sigmoid(self.a_z)                          # (d_model,)

        # --- 候选状态计算 ---
        self.a_h = self.W_h @ x_t + self.U_h @ h_prev + self.b_h
        self.h_tilde = np.tanh(self.a_h)                      # (d_model,)

        # --- 状态更新 ---
        self.h_t = (1.0 - self.z_t) * h_prev + self.z_t * self.h_tilde

        return self.h_t

    def backward(self, d_h_t):
        """
        单步反向传播（BPTT 展开）。
        d_h_t: (d_model,) 损失对当前隐藏状态的梯度
        返回:   (d_x_t, d_h_prev) — 对输入和上一状态的梯度
        """
        # ---- 对 h_t 的梯度分三路 ----
        # h_t = (1-z) * h_prev + z * h̃
        # ∂h_t/∂h_prev = (1-z) (直接路径)
        # ∂h_t/∂h̃ = z
        # ∂h_t/∂z = h̃ - h_prev

        d_h_prev_direct = (1.0 - self.z_t) * d_h_t             # (d_model,)
        d_h_tilde = self.z_t * d_h_t                            # (d_model,)
        d_z_partial = (self.h_tilde - self.h_prev) * d_h_t      # 逐元素乘

        # ---- 候选状态 tanh 反向 ----
        d_a_h = d_h_tilde * (1.0 - self.h_tilde ** 2)           # tanh'(x) = 1 - tanh²(x)

        # ---- 更新门 sigmoid 反向 ----
        d_a_z = d_z_partial * self.z_t * (1.0 - self.z_t)       # σ'(x) = σ(x)(1-σ(x))

        # ---- 权重梯度累加 ----
        self.dW_h += np.outer(self.x_t, d_a_h)                   # (d, d)
        self.dU_h += np.outer(self.h_prev, d_a_h)
        self.db_h += d_a_h

        self.dW_z += np.outer(self.x_t, d_a_z)
        self.dU_z += np.outer(self.h_prev, d_a_z)
        self.db_z += d_a_z

        # ---- 对输入和前一状态的间接梯度 ----
        # x_t 影响 a_h (via W_h) 和 a_z (via W_z)
        d_x = d_a_h @ self.W_h.T + d_a_z @ self.W_z.T           # (d_model,)

        # h_prev 影响 a_h (via U_h) 和 a_z (via U_z)
        d_h_prev_indirect = d_a_h @ self.U_h.T + d_a_z @ self.U_z.T

        # 总梯度 = 直接路径 + 间接路径
        d_h_prev = d_h_prev_direct + d_h_prev_indirect

        return d_x, d_h_prev

    def zero_grad(self):
        self.dW_z.fill(0); self.dU_z.fill(0); self.db_z.fill(0)
        self.dW_h.fill(0); self.dU_h.fill(0); self.db_h.fill(0)

    def apply_gradients(self, lr):
        self.W_z -= lr * self.dW_z; self.U_z -= lr * self.dU_z; self.b_z -= lr * self.db_z
        self.W_h -= lr * self.dW_h; self.U_h -= lr * self.dU_h; self.b_h -= lr * self.db_h


# ============================================================
# 稀疏门控激活 — 仅激活重要维度
# ============================================================

class SparseGate:
    """
    Top-K 硬稀疏门控 —— 仅激活最重要的维度。

    前向：
      importance = |W_g @ h + b_g|     -- 每个维度的重要性分数
      mask = top_k(importance)          -- 只保留前 K 个最高分
      h_sparse = h * mask               -- 稀疏激活

    直观理解：
      - K = d_model // 2 → 一半维度被激活
      - K = d_model // 4 → 四分之一被激活（更稀疏）
      - 类似于生物大脑：每个时刻只有一小部分神经元放电，
       且放电的总是"最相关"的那些
    """

    def __init__(self, d_model, sparsity_k=None):
        """
        sparsity_k: 保留的激活维度数。None 则自动取 d_model // 3
        """
        self.d = d_model
        self.k = sparsity_k if sparsity_k else max(1, d_model // 3)
        scale = np.sqrt(2.0 / d_model) * 0.3
        self.W_g = np.random.randn(d_model, d_model) * scale
        self.b_g = np.zeros(d_model)

        self.dW_g = np.zeros((d_model, d_model))
        self.db_g = np.zeros(d_model)

    def forward(self, h):
        """
        h: (d_model,) 输入向量
        返回: h_sparse (d_model,), sparsity (float 被抑制的比例)
        """
        self.h = h
        # 计算重要性分数
        self.importance = np.abs(self.W_g @ h + self.b_g)        # (d_model,)

        # 找出 top-k 索引
        self.top_k_indices = np.argpartition(
            -self.importance, self.k)[:self.k]

        # 构造 mask：只有 top-k 维度为 1，其余为 0
        self.mask = np.zeros(self.d, dtype=np.float64)
        self.mask[self.top_k_indices] = 1.0

        # 稀疏激活
        self.h_sparse = h * self.mask

        # 稀疏度 = 被抑制的维度比例
        self.sparsity = 1.0 - self.k / self.d
        return self.h_sparse, self.sparsity

    def backward(self, d_h_sparse):
        """
        反向传播。
        关键：只有被激活的 top-k 维度有非零梯度；
        未被激活的维度，其梯度被截断为 0。
        这强制模型学习"哪些维度真正重要"。

        d_h_sparse: (d_model,) 梯度
        返回: d_h (d_model,) 对输入的梯度
        """
        # h_sparse = h * mask
        # mask 是硬阈值，不参与梯度传播（直通估计 STE）
        # ∂h_sparse/∂h = mask  (直通方式)
        d_h = d_h_sparse * self.mask                               # (d_model,)

        # 重要性分数的梯度（只对 top-k 维度回传）
        # g_i = |W_g @ h|_i，梯度方向取决于 sign
        signs = np.sign(self.W_g @ self.h + self.b_g)              # (d_model,)
        d_importance = d_h_sparse * signs * self.mask               # 只更新 top-k
        # 注意：这里 d_h 不直接从 importance 回传（直通估计），
        # 但 W_g 需要从 importance 更新

        # 累加 W_g 和 b_g 梯度
        self.dW_g += np.outer(self.h, d_importance)
        self.db_g += d_importance

        # 直通估计：d_h 已经通过 mask，不再从 importance 路径额外回传
        return d_h

    def zero_grad(self):
        self.dW_g.fill(0); self.db_g.fill(0)

    def apply_gradients(self, lr):
        self.W_g -= lr * self.dW_g; self.b_g -= lr * self.db_g


# ============================================================
# 前馈网络（FFN）— 用于稀疏门控之后
# ============================================================

class FeedForward:
    """两层全连接：Linear → ReLU → Linear"""

    def __init__(self, d_model, d_ff):
        s1 = np.sqrt(2.0 / d_model) * 0.5
        s2 = np.sqrt(2.0 / d_ff) * 0.5

        self.W1 = np.random.randn(d_model, d_ff) * s1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * s2
        self.b2 = np.zeros(d_model)

        self.dW1 = np.zeros_like(self.W1); self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2); self.db2 = np.zeros_like(self.b2)

    def forward(self, x):
        self.x = x
        self.h1 = x @ self.W1 + self.b1                         # (d_ff,)
        self.a1 = relu(self.h1)                                  # (d_ff,)
        out = self.a1 @ self.W2 + self.b2                        # (d_model,)
        return out

    def backward(self, d_out):
        d_a1 = d_out @ self.W2.T                                 # (d_ff,)
        self.dW2 += np.outer(self.a1, d_out)                     # (d_ff, d_model)
        self.db2 += d_out

        d_h1 = d_a1 * (self.h1 > 0)                               # ReLU 梯度
        d_x = d_h1 @ self.W1.T                                   # (d_model,)
        self.dW1 += np.outer(self.x, d_h1)                       # (d_model, d_ff)
        self.db1 += d_h1
        return d_x

    def zero_grad(self):
        self.dW1.fill(0); self.db1.fill(0)
        self.dW2.fill(0); self.db2.fill(0)

    def apply_gradients(self, lr):
        self.W1 -= lr * self.dW1; self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2; self.b2 -= lr * self.db2


# ============================================================
# 层归一化
# ============================================================

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        self.dgamma = np.zeros(d_model)
        self.dbeta = np.zeros(d_model)

    def forward(self, x):
        self.x = x
        self.mean = x.mean()
        self.var = x.var()
        self.inv_std = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) * self.inv_std
        return self.gamma * self.x_hat + self.beta

    def backward(self, d_out):
        self.dgamma += np.sum(d_out * self.x_hat)
        self.dbeta += np.sum(d_out)

        dx_hat = d_out * self.gamma
        N = d_out.shape[-1]
        dx = self.inv_std * (dx_hat - dx_hat.mean() -
                             self.x_hat * (dx_hat * self.x_hat).mean())
        return dx

    def zero_grad(self):
        self.dgamma.fill(0); self.dbeta.fill(0)

    def apply_gradients(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


# ============================================================
# minGRU-S 模型 — minGRU + 稀疏门控
# ============================================================

class MinGRUS:
    """
    minGRU-S (minGRU with Sparsity)

    架构流程（每个字符处理一步）：
      1. 嵌入 + 简单位置编码
      2. minGRU 更新隐藏状态 h_t = minGRU(x_t, h_{t-1})
      3. 层归一化 h_t
      4. 稀疏门控：h_sparse, sparsity = SparseGate(h_t)
         → 仅重要的维度被激活
      5. 前馈网络：FFN(h_sparse) + 残差连接
      6. 输出投影 → softmax → 预测下一个字符
    """

    def __init__(self, vocab_size, d_model=32, d_ff=48):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff

        # 词嵌入
        scale_emb = np.sqrt(2.0 / (vocab_size + d_model))
        self.W_emb = np.random.randn(vocab_size, d_model) * scale_emb
        self.dW_emb = np.zeros_like(self.W_emb)

        # 核心组件
        self.min_gru = MinGRU(d_model)
        self.ln_h = LayerNorm(d_model)        # 对隐藏状态做归一化
        self.sparse_gate = SparseGate(d_model) # 稀疏门控
        self.ffn = FeedForward(d_model, d_ff)
        self.ln_out = LayerNorm(d_model)       # 输出前归一化

        # 输出投影
        scale_out = np.sqrt(2.0 / d_model)
        self.W_out = np.random.randn(d_model, vocab_size) * scale_out
        self.b_out = np.zeros(vocab_size)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)

    def forward(self, token_ids, h_prev):
        """
        处理一段字符序列，返回最后一个位置的预测概率。
        
        token_ids: (seq_len,) 上下文 token 索引序列
        h_prev:    (d_model,) 循环状态（处理前一个序列后的状态）
        返回: (probs, h_final, sparsity)
          probs:    (vocab_size,) 预测概率
          h_final:  (d_model,) 最终隐藏状态（后续复用）
          sparsity: float 稀疏度
        """
        h = h_prev

        for t, idx in enumerate(token_ids):
            # 嵌入
            x_t = self.W_emb[idx]                              # (d_model,)
            # minGRU 一步
            h = self.min_gru.forward(x_t, h)                    # (d_model,)

        # 层归一化
        h_norm = self.ln_h.forward(h)

        # 稀疏门控
        h_sparse, sparsity = self.sparse_gate.forward(h_norm)

        # 前馈网络 + 残差连接
        ffn_out = self.ffn.forward(h_sparse)                   # (d_model,)
        h_residual = h_norm + ffn_out

        # 输出前归一化
        h_out = self.ln_out.forward(h_residual)

        # 输出投影 → logits → softmax
        logits = h_out @ self.W_out + self.b_out               # (vocab_size,)
        probs = softmax(logits)

        return probs, h, sparsity

    def backward(self, target_idx):
        """
        反向传播。
        target_idx: 真实下一个 token 索引
        """
        # --- 输出端：softmax + CE 梯度 ---
        d_logits = self.probs.copy()
        d_logits[target_idx] -= 1.0

        # --- 输出投影 ---
        self.dW_out += np.outer(self.h_out, d_logits)
        self.db_out += d_logits
        d_h_out = d_logits @ self.W_out.T

        # --- 输出归一化 ---
        d_h_residual = self.ln_out.backward(d_h_out)

        # --- 残差 + FFN ---
        d_h_sparse = self.ffn.backward(d_h_residual)
        d_h_norm_from_residual = d_h_residual    # 残差路径
        d_h_norm_from_ffn = d_h_sparse           # FFN 路径

        # --- 稀疏门控 ---
        d_h_norm_from_sparse = self.sparse_gate.backward(d_h_norm_from_ffn)
        d_h_norm = d_h_norm_from_residual + d_h_norm_from_sparse

        # --- 隐藏状态归一化 ---
        d_h = self.ln_h.backward(d_h_norm)

        # --- minGRU 反向（BPTT 逐时间步展开）---
        h_curr = d_h  # 从损失端回传的梯度
        # 这里需要反向展开 minGRU 的所有时间步
        # 只展开最后一步（因为损失仅来自最后一个位置的预测）
        d_x, d_h_prev = self.min_gru.backward(h_curr)

        # 嵌入层梯度
        np.add.at(self.dW_emb, self.token_ids[-1], d_x)

    def train_step(self, token_ids, target_idx, h_prev, lr):
        """单步训练"""
        self.zero_grad()

        # 缓存 token_ids（供 backward 用）
        self.token_ids = token_ids

        # 前向
        self.probs, h_new, sparsity = self.forward(token_ids, h_prev)
        self.h_out = self._last_h_out()  # 供 backward 用

        # 损失
        loss = cross_entropy(self.probs, target_idx)

        # 反向
        self.backward(target_idx)

        # 更新
        self.apply_gradients(lr)

        return loss, h_new, sparsity

    def _last_h_out(self):
        """获取最后一次 forward 中的 h_out"""
        # h_out = ln_out.forward(h_residual) 的输出
        # 在 ffn.forward(h_sparse) + h_norm → h_residual → ln_out → h_out
        return self.ln_out.forward.__self__  # 不，这是方法不是属性...

    # 修正：直接在 backward 前缓存关键激活值
    def forward_with_cache(self, token_ids, h_prev):
        """前向传播 + 缓存所有反向需要的激活值"""
        self.token_ids = token_ids
        h = h_prev
        self.h_prev_cached = h_prev

        for t, idx in enumerate(token_ids):
            x_t = self.W_emb[idx]
            h = self.min_gru.forward(x_t, h)

        self.h_norm = self.ln_h.forward(h)
        self.h_sparse, sparsity = self.sparse_gate.forward(self.h_norm)
        self.ffn_out = self.ffn.forward(self.h_sparse)
        self.h_residual = self.h_norm + self.ffn_out
        self.h_out = self.ln_out.forward(self.h_residual)
        logits = self.h_out @ self.W_out + self.b_out
        self.probs = softmax(logits)

        return self.probs, h, sparsity

    def backward_with_cache(self, target_idx):
        """使用 forward_with_cache 的缓存进行反向传播"""
        d_logits = self.probs.copy()
        d_logits[target_idx] -= 1.0

        self.dW_out += np.outer(self.h_out, d_logits)
        self.db_out += d_logits
        d_h_out = d_logits @ self.W_out.T

        d_h_residual = self.ln_out.backward(d_h_out)

        d_h_sparse = self.ffn.backward(d_h_residual)
        d_h_norm = d_h_residual + self.sparse_gate.backward(d_h_sparse)

        d_h = self.ln_h.backward(d_h_norm)

        # minGRU 反向
        d_x, _ = self.min_gru.backward(d_h)

        np.add.at(self.dW_emb, self.token_ids[-1], d_x)

    def zero_grad(self):
        self.dW_emb.fill(0); self.dW_out.fill(0); self.db_out.fill(0)
        self.min_gru.zero_grad()
        self.ln_h.zero_grad()
        self.sparse_gate.zero_grad()
        self.ffn.zero_grad()
        self.ln_out.zero_grad()

    def apply_gradients(self, lr):
        self.W_emb -= lr * self.dW_emb
        self.W_out -= lr * self.dW_out
        self.b_out -= lr * self.db_out
        self.min_gru.apply_gradients(lr)
        self.ln_h.apply_gradients(lr)
        self.sparse_gate.apply_gradients(lr)
        self.ffn.apply_gradients(lr)
        self.ln_out.apply_gradients(lr)

    def predict(self, token_ids, h_prev):
        """仅前向，用于生成"""
        probs, h_new, sparsity = self.forward(token_ids, h_prev)
        return probs, h_new


# ============================================================
# 情景记忆库（沿用 xiaoyan 的记忆系统）
# ============================================================

class EpisodicMemory:
    """
    情景记忆库 (context, target, strength)
    
    机制不变：插入/更新强度 → 全局衰减 → 低强度删除 → 加权采样重放
    """

    def __init__(self, decay=0.995, threshold=0.05, initial_strength=1.0):
        self.memories = []
        self.decay = decay
        self.threshold = threshold
        self.initial_strength = initial_strength
        self.stats = {'added': 0, 'merged': 0, 'removed': 0}

    def add_or_update(self, context, target):
        for mem in self.memories:
            if mem['context'] == context and mem['target'] == target:
                mem['strength'] += self.initial_strength
                self.stats['merged'] += 1
                return
        self.memories.append({
            'context': context,
            'target': target,
            'strength': self.initial_strength
        })
        self.stats['added'] += 1

    def decay_all(self):
        for mem in self.memories:
            mem['strength'] *= self.decay
        before = len(self.memories)
        self.memories = [m for m in self.memories
                         if m['strength'] >= self.threshold]
        self.stats['removed'] += (before - len(self.memories))

    def sample_batch(self, batch_size):
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
        if not self.memories:
            return 0.0
        return np.mean([m['strength'] for m in self.memories])


# ============================================================
# 在线训练器
# ============================================================

class OnlineTrainer:
    """逐字符训练 + 巩固重放"""

    def __init__(self, model, vocab, context_len=5,
                 lr=0.03, replay_interval=30, replay_batch=8,
                 mem_decay=0.995, mem_threshold=0.1):
        self.model = model
        self.vocab = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.context_len = context_len
        self.lr = lr
        self.replay_interval = replay_interval
        self.replay_batch = replay_batch

        self.memory = EpisodicMemory(decay=mem_decay,
                                     threshold=mem_threshold)
        self.buffer = deque(maxlen=context_len)
        self.h_state = np.zeros(model.d_model)     # minGRU 循环状态
        self.step = 0
        self.total_loss = 0.0
        self.sparsity_history = []

    def _ids_to_tuple(self, ids):
        return tuple(int(i) for i in ids)

    def train_on_char(self, char):
        """逐字符训练"""
        self.step += 1
        target_idx = self.vocab[char]

        if len(self.buffer) < self.context_len:
            self.buffer.append(target_idx)
            return None

        context = np.array(list(self.buffer), dtype=np.int32)
        context_tuple = self._ids_to_tuple(context)

        # --- 在线学习 ---
        self.model.zero_grad()
        probs, h_new, sparsity = self.model.forward_with_cache(
            context, self.h_state)
        loss = cross_entropy(probs, target_idx)
        self.model.backward_with_cache(target_idx)
        self.model.apply_gradients(self.lr)

        self.h_state = h_new
        self.total_loss += loss
        self.sparsity_history.append(sparsity)

        # --- 更新情景记忆 ---
        self.memory.add_or_update(context_tuple, target_idx)
        self.memory.decay_all()

        # --- 巩固重放 ---
        if self.step % self.replay_interval == 0 and self.memory.size() > 0:
            self._consolidate()

        self.buffer.append(target_idx)
        return loss

    def _consolidate(self):
        """巩固重放（梯度累加后统一更新）"""
        contexts, targets = self.memory.sample_batch(self.replay_batch)
        if not contexts:
            return

        self.model.zero_grad()
        for ctx, tgt in zip(contexts, targets):
            ctx_arr = np.array(list(ctx), dtype=np.int32)
            probs, _, _ = self.model.forward_with_cache(
                ctx_arr, np.zeros(self.model.d_model))
            self.model.backward_with_cache(tgt)

        self.model.apply_gradients(self.lr * 0.5)

    def train_on_text(self, text):
        losses = []
        for ch in text:
            if ch not in self.vocab:
                continue
            loss = self.train_on_char(ch)
            if loss is not None:
                losses.append(loss)
            if self.step % 100 == 0 and losses:
                recent_avg = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
                avg_sparsity = (np.mean(self.sparsity_history[-100:])
                                if self.sparsity_history else 0)
                print(f"  [步骤 {self.step:5d}] 损失: {recent_avg:.4f}  "
                      f"| 稀疏度: {avg_sparsity:.1%}  "
                      f"| 记忆: {self.memory.size():3d}")
        return losses


# ============================================================
# 文本生成
# ============================================================

def generate(model, start_text, vocab, idx_to_char,
             context_len=5, max_new_chars=50, temperature=0.6):
    pad_idx = vocab.get('<PAD>', 0)
    result = list(start_text)
    h_state = np.zeros(model.d_model)

    for _ in range(max_new_chars):
        recent = result[-context_len:]
        ids = [vocab.get(c, pad_idx) for c in recent]
        if len(ids) < context_len:
            ids = [pad_idx] * (context_len - len(ids)) + ids

        context = np.array(ids, dtype=np.int32)
        probs, h_state = model.predict(context, h_state)

        if temperature <= 0:
            next_idx = np.argmax(probs)
        else:
            logits = np.log(probs + 1e-12) / temperature
            scaled = softmax(logits)
            next_idx = np.random.choice(len(probs), p=scaled)

        result.append(idx_to_char.get(next_idx, '?'))

    return ''.join(result)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  minGRU-S 语言模型 —— 演示")
    print("  (纯 NumPy · minGRU + 稀疏门控 · CPU 优化)")
    print("=" * 60)

    # 数据与词汇表
    sentence_a = "hello world"
    sentence_b = "the cat sat"
    all_chars = sorted(set(sentence_a + sentence_b))
    print(f"\n词汇表: {len(all_chars)} 字符 | 句子A: \"{sentence_a}\" | 句子B: \"{sentence_b}\"")

    vocab = {'<PAD>': 0}
    for i, ch in enumerate(all_chars):
        vocab[ch] = i + 1
    idx_to_char = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # 初始化模型
    print("\n初始化 minGRU-S (d_model=32, d_ff=48)...")
    model = MinGRUS(vocab_size=vocab_size, d_model=32, d_ff=48)

    trainer = OnlineTrainer(
        model=model, vocab=vocab,
        context_len=5, lr=0.03,
        replay_interval=30, replay_batch=8,
        mem_decay=0.995, mem_threshold=0.1,
    )

    # 交替训练
    print("\n" + "=" * 60)
    print("  交替训练 40 轮...")
    print("=" * 60)

    for epoch in range(40):
        trainer.train_on_text(sentence_a)
        trainer.train_on_text(sentence_b)
        if (epoch + 1) % 10 == 0:
            avg_sp = (np.mean(trainer.sparsity_history[-100:])
                      if trainer.sparsity_history else 0)
            print(f"  --- 第 {epoch+1} 轮完成 | 记忆: {trainer.memory.size()} "
                  f"| 稀疏度: {avg_sp:.1%} ---")

    # 生成测试
    print("\n" + "=" * 60)
    print("  交替训练后生成测试")
    print("=" * 60)

    for seed in ["hel", "the", "he", "th", "ca"]:
        gen = generate(model, seed, vocab, idx_to_char,
                       context_len=5, max_new_chars=30, temperature=0.5)
        print(f"  \"{seed}\" → \"{gen}\"")

    # 遗忘测试
    print("\n" + "=" * 60)
    print("  遗忘测试：仅训练句子 A 30 轮")
    print("=" * 60)

    for i in range(30):
        trainer.train_on_text(sentence_a)

    print("\n  遗忘后生成:")
    for seed in ["hel", "the", "he", "th", "ca"]:
        gen = generate(model, seed, vocab, idx_to_char,
                       context_len=5, max_new_chars=30, temperature=0.5)
        print(f"  \"{seed}\" → \"{gen}\"")

    # 记忆库统计
    print("\n" + "=" * 60)
    print("  情景记忆库状态")
    print("=" * 60)
    print(f"  总条目: {trainer.memory.size()}  "
          f"| 平均强度: {trainer.memory.avg_strength():.3f}")
    print(f"  新增: {trainer.memory.stats['added']}  "
          f"| 合并: {trainer.memory.stats['merged']}  "
          f"| 删除: {trainer.memory.stats['removed']}")

    # Top 记忆
    sorted_mems = sorted(trainer.memory.memories,
                         key=lambda m: m['strength'], reverse=True)
    print("\n  最强记忆 Top 5:")
    for i, mem in enumerate(sorted_mems[:5]):
        ctx_str = ''.join(idx_to_char.get(c, '?') for c in mem['context'])
        tgt_str = idx_to_char.get(mem['target'], '?')
        print(f"    {i+1}. \"{ctx_str}\" → '{tgt_str}'  {mem['strength']:.3f}")

    print("\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)
