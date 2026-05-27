# -*- coding: utf-8 -*-
"""
小炎对话 —— 加载训练好的 minGRU-S 模型进行对话生成
====================================================
用法：
  python chat_xiaoyan.py

模型文件：xiaoyan_model.pkl（由 train_xiaoyan.py 生成）
"""

import numpy as np
import pickle
import sys

# ============================================================
# 复制模型类（与训练脚本一致）
# ============================================================

def sigmoid(x):
    x_clipped = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum()

class MinGRU:
    def __init__(self, d):
        self.d = d
        self.W_z = np.random.randn(d, d) * 0.01
        self.U_z = np.random.randn(d, d) * 0.01
        self.b_z = np.zeros(d)
        self.W_h = np.random.randn(d, d) * 0.01
        self.U_h = np.random.randn(d, d) * 0.01
        self.b_h = np.zeros(d)

    def forward(self, x_t, h_prev):
        self.a_z = self.W_z @ x_t + self.U_z @ h_prev + self.b_z
        self.z_t = sigmoid(self.a_z)
        self.a_h = self.W_h @ x_t + self.U_h @ h_prev + self.b_h
        self.h_tilde = np.tanh(self.a_h)
        self.h_t = (1.0 - self.z_t) * h_prev + self.z_t * self.h_tilde
        return self.h_t

class SparseGate:
    def __init__(self, d, k=10):
        self.d = d; self.k = k
        self.W_g = np.random.randn(d, d) * 0.01
        self.b_g = np.zeros(d)

    def forward(self, h):
        importance = np.abs(self.W_g @ h + self.b_g)
        top_k = np.argpartition(-importance, self.k)[:self.k]
        mask = np.zeros(self.d, dtype=np.float64)
        mask[top_k] = 1.0
        return h * mask

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return relu(x @ self.W1 + self.b1) @ self.W2 + self.b2

class LayerNorm:
    def __init__(self, d):
        self.gamma = np.ones(d); self.beta = np.zeros(d)

    def forward(self, x):
        mean = x.mean(); var = x.var()
        inv_std = 1.0 / np.sqrt(var + 1e-5)
        return self.gamma * (x - mean) * inv_std + self.beta

class MinGRUS:
    """仅推理用的迷你版"""
    def __init__(self, vocab_size, d_model=32, d_ff=48):
        self.d = d_model; self.vocab_size = vocab_size
        scale = np.sqrt(2.0 / (vocab_size + d_model))
        self.W_emb = np.random.randn(vocab_size, d_model) * scale
        self.min_gru = MinGRU(d_model)
        self.ln_h = LayerNorm(d_model)
        self.sparse_gate = SparseGate(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln_out = LayerNorm(d_model)
        self.W_out = np.random.randn(d_model, vocab_size) * 0.01
        self.b_out = np.zeros(vocab_size)

    def predict(self, token_ids, h_prev):
        h = h_prev
        for idx in token_ids:
            x_t = self.W_emb[idx]
            h = self.min_gru.forward(x_t, h)
        h_norm = self.ln_h.forward(h)
        h_sparse = self.sparse_gate.forward(h_norm)
        h_res = h_norm + self.ffn.forward(h_sparse)
        h_out = self.ln_out.forward(h_res)
        logits = h_out @ self.W_out + self.b_out
        return softmax(logits), h


# ============================================================
# 加载模型
# ============================================================

def load_model(path):
    """从 pickle 文件加载模型"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    model = MinGRUS(
        vocab_size=len(data['vocab']),
        d_model=data['d_model'],
        d_ff=data['d_ff'],
    )

    # 恢复权重
    model.W_emb = data['W_emb']
    model.min_gru.W_z = data['W_z']; model.min_gru.U_z = data['U_z']; model.min_gru.b_z = data['b_z']
    model.min_gru.W_h = data['W_h']; model.min_gru.U_h = data['U_h']; model.min_gru.b_h = data['b_h']
    model.sparse_gate.W_g = data['W_g']; model.sparse_gate.b_g = data['b_g']
    model.ffn.W1 = data['W1']; model.ffn.b1 = data['b1']
    model.ffn.W2 = data['W2']; model.ffn.b2 = data['b2']
    model.ln_h.gamma = data['gamma_h']; model.ln_h.beta = data['beta_h']
    model.ln_out.gamma = data['gamma_out']; model.ln_out.beta = data['beta_out']
    model.W_out = data['W_out']; model.b_out = data['b_out']

    return model, data['vocab'], data['idx_to_char'], data['context_len']


# ============================================================
# 生成对话
# ============================================================

def chat_generate(model, vocab, idx_to_char, prompt, context_len=5,
                  max_new=60, temperature=0.6):
    """给定提示文本，自回归生成回复"""
    pad = vocab.get('<PAD>', 0)
    result = list(prompt)
    h_state = np.zeros(model.d)

    for _ in range(max_new):
        recent = result[-context_len:]
        ids = [vocab.get(c, pad) for c in recent]
        if len(ids) < context_len:
            ids = [pad] * (context_len - len(ids)) + ids

        probs, h_state = model.predict(np.array(ids, dtype=np.int32), h_state)

        if temperature <= 0:
            nxt = np.argmax(probs)
        else:
            logits = np.log(probs + 1e-12) / temperature
            nxt = np.random.choice(len(probs), p=softmax(logits))

        result.append(idx_to_char.get(nxt, '?'))

    return ''.join(result)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    import os
    model_path = '/data/data/com.termux/files/home/xiaoyan/xiaoyan_model.pkl'

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行 train_xiaoyan.py 生成模型。")
        sys.exit(1)

    print("加载小炎模型...")
    model, vocab, idx_to_char, ctx_len = load_model(model_path)
    print(f"模型已加载 (d_model={model.d}, vocab={len(vocab)}, ctx_len={ctx_len})")

    print("\n" + "=" * 50)
    print("  小炎 对话模式")
    print("  输入 'quit' 退出")
    print("=" * 50)

    # 预设测试场景
    test_prompts = [
        ("嘿嘿", "笑声开头"),
        ("主人", "称呼主人"),
        ("我推测", "分析推理"),
        ("哼", "傲娇语气"),
        ("很简单啊", "轻松分析"),
        ("敢动", "战斗模式"),
    ]

    print("\n--- 预设测试 ---")
    for prompt, desc in test_prompts:
        gen = chat_generate(model, vocab, idx_to_char, prompt, ctx_len,
                           max_new=50, temperature=0.5)
        print(f"\n[{desc}] 输入: {prompt!r}")
        print(f"小炎: {gen}")

    print("\n--- 交互模式 ---")
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n小炎: 主人要走了吗？那我回去休息啦～")
            break

        if user_input.lower() == 'quit':
            print("小炎: 主人再见！")
            break
        if not user_input:
            continue

        # 取用户输入的后几个字作为种子
        seed = user_input[-4:] if len(user_input) >= 4 else user_input
        reply = chat_generate(model, vocab, idx_to_char, seed, ctx_len,
                             max_new=50, temperature=0.55)
        print(f"小炎: {reply}")
