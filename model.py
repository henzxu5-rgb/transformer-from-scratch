import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.shape[1] > self.seq_len:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds maximum {self.seq_len}"
            )
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return self.alpha * (x - mean) * torch.rsqrt(variance + self.eps) + self.bias


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        if d_model % h != 0:
            raise ValueError("d_model must be divisible by h")
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_scores = None

    @staticmethod
    def attention(q, k, v, mask, dropout):
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=attention_scores.device)
            attention_scores = attention_scores.masked_fill(
                ~mask, torch.finfo(attention_scores.dtype).min
            )
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ v, attention_scores

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # (batch, seq, d_model) -> (batch, h, seq, d_k)
        q = q.view(batch_size, q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, k.shape[1], self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, v.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = self.attention(q, k, v, mask, self.dropout)

        # (batch, h, seq, d_k) -> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, d_model, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda value: self.self_attention_block(value, value, value, src_mask)
        )
        return self.residual_connections[1](x, self.feed_forward_block)


class Encoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, d_model, self_attention_block, cross_attention_block,
                 feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda value: self.self_attention_block(value, value, value, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda value: self.cross_attention_block(
                value, encoder_output, encoder_output, src_mask
            ),
        )
        return self.residual_connections[2](x, self.feed_forward_block)


class Decoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class LinearLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


class Transformer(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed,
                 src_pos, tgt_pos, linear_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear_layer = linear_layer

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_pos(self.tgt_embed(tgt))
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def linear(self, x):
        return self.linear_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int,
                      tgt_seq_len: int, d_model: int = 512, N: int = 6,
                      h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    # ① 词嵌入 + 位置编码（输入线和输出线各一套）
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # ② N 层编码器
    encoder_blocks = []
    for _ in range(N):
        encoder_blocks.append(EncoderBlock(
            d_model,
            MultiHeadAttention(d_model, h, dropout),
            FeedForward(d_model, d_ff, dropout),
            dropout,
        ))
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # ③ N 层解码器（比编码器多一个交叉注意力）
    decoder_blocks = []
    for _ in range(N):
        decoder_blocks.append(DecoderBlock(
            d_model,
            MultiHeadAttention(d_model, h, dropout),
            MultiHeadAttention(d_model, h, dropout),
            FeedForward(d_model, d_ff, dropout),
            dropout,
        ))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # ④ 输出层
    linear_layer = LinearLayer(d_model, tgt_vocab_size)

    # ⑤ 把所有零件接进 Transformer
    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos,
        tgt_pos,
        linear_layer,
    )

    # ⑥ 参数初始化
    _initialize_parameters(transformer)
    return transformer


# Decoder-only 架构，从 Encoder-Decoder 改造而来，复用上面的零件
class GPTBlock(nn.Module):

    def __init__(self, d_model, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])

    def forward(self, x, mask):
        x = self.residual_connections[0](
            x, lambda value: self.self_attention_block(value, value, value, mask)
        )
        return self.residual_connections[1](x, self.feed_forward_block)


class GPT(nn.Module):

    def __init__(self, vocab_size: int, seq_len: int, N: int, d_model: int,
                 h: int, dropout: float, d_ff: int):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = InputEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(
                d_model,
                MultiHeadAttention(d_model, h, dropout),
                FeedForward(d_model, d_ff, dropout),
                dropout,
            )
            for _ in range(N)
        ])
        self.norm = LayerNormalization(d_model)
        self.lm_head = LinearLayer(d_model, vocab_size)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, seq_len, seq_len))

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("input must have shape (batch, sequence)")
        if x.shape[1] > self.seq_len:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds maximum {self.seq_len}"
            )

        x = self.positional_encoding(self.token_embedding(x))
        mask = self.causal_mask[:, :, :x.shape[1], :x.shape[1]]
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.norm(x))


def build_gpt(vocab_size: int, seq_len: int = 64, d_model: int = 128,
              N: int = 4, h: int = 4, dropout: float = 0.1,
              d_ff: int = 512) -> GPT:
    gpt = GPT(vocab_size, seq_len, N, d_model, h, dropout, d_ff)
    _initialize_parameters(gpt)
    return gpt


def _initialize_parameters(module):
    for parameter in module.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
