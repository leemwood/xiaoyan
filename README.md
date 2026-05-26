# Mini Transformer LM

微型 Transformer 语言模型 —— 纯 NumPy 手动实现。

## 项目性质

本项目为**想法型 / 概念验证项目**，旨在探索以下机制的可行性：

- 纯 NumPy 手工实现 Transformer（嵌入、位置编码、单头自注意力、FFN、层归一化）
- 逐字符在线学习 + 手动反向传播
- 情景记忆库（记忆强度 + 衰减 + 阈值遗忘 + 加权采样）
- 巩固重放机制（仿海马体记忆巩固，防止灾难性遗忘）

**代码实现由 DeepSeek V4 Pro 完成。**

## 依赖

仅需 `numpy`：

```bash
# Termux (Android)
pkg install python-numpy

# 通用
pip install numpy
```

## 运行

```bash
python mini_transformer_lm.py
```

## 许可

MIT License — 详见 [LICENSE](LICENSE)
