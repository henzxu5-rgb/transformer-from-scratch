# Transformer from Scratch

这是一个以理解 Transformer 为目标的学习项目。项目从经典的 Encoder–Decoder 结构开始，在复用基础模块的基础上进一步实现了 Decoder-only Transformer。

代码使用 PyTorch 的基础组件编写，不调用现成的 Transformer 模型实现。项目关注各模块的内部结构、连接方式，以及 next-token prediction 完整流程。

## 实现内容

- 词嵌入与正弦位置编码
- 多头注意力、前馈网络、残差连接和 LayerNorm
- Encoder–Decoder Transformer
- 带因果掩码的 Decoder-only Transformer
- 字符级数据处理、训练和自回归生成

## 文件说明

| 文件 | 内容 |
|---|---|
| `model.py` | Encoder–Decoder 和 Decoder-only Transformer |
| `data.py` | 字符级编码与训练 batch |
| `train.py` | 模型训练、保存和文本生成 |
| `data/input.txt` | Tiny Shakespeare 训练文本 |

## 训练结果

用 `train.py` 的默认模型配置（0.81M 参数，4 层，上下文 64 个字符），把 `STEPS` 改成 5000，在 CPU 上训练约 10 分钟：

| step | 0 | 500 | 1000 | 2000 | 3000 | 4000 | 4999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss | 4.98 | 2.46 | 2.24 | 2.05 | 2.01 | 1.85 | 1.75 |

表里是单个 batch 的训练 loss，会有波动。作为参照，在 65 个字符里完全随机猜的 loss 是 ln(65) ≈ 4.17。

以 `ROMEO:` 开头生成的一段（temperature 0.8）：

```
ROMEO:
I comprater you hears brother, he's do the like, what of shale
Who law some, which and they him mether.

YORK:
Hen thou gild and the preoces father.

SICINIUS:
The const that slike of Wither,
Ay not you grath, like of the a fancion of you spolise's
```

模型学会了剧本的格式（角色名、冒号、换行）和不少常见英文单词，但拼写和语义还不通顺。目前没有划分验证集，只能看训练 loss。
