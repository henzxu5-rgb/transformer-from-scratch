import time

import torch
import torch.nn.functional as F

from data import DATA_DIR, decode, encode, get_batch, load_data
from model import build_gpt


# 超参数都集中在这里，方便逐个调整
SEQ_LEN = 64
BATCH_SIZE = 16
STEPS = 300
LOG_EVERY = 50
LR = 3e-4
SEED = 0

D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
D_FF = 512
DROPOUT = 0.1

PROMPT = "ROMEO:"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8


def generate(model, prompt_ids, max_new_tokens, temperature, device):
    ids = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = torch.tensor(
                [ids[-SEQ_LEN:]], dtype=torch.long, device=device
            )
            logits = model(window)[0, -1]
            probabilities = torch.softmax(logits / temperature, dim=-1)
            ids.append(torch.multinomial(probabilities, 1).item())
    return ids


def main():
    start_time = time.time()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 数据
    data, stoi, itos = load_data()
    vocab_size = len(stoi)
    print(f"[1] 设备 {device} | 语料 {data.numel():,} 字符 | 词表 {vocab_size}")

    # 2) Decoder-only Transformer
    model = build_gpt(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        N=N_LAYERS,
        h=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"[2] 模型参数 {parameter_count / 1e6:.2f}M")

    # 3) 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 4) 训练循环：取批 -> 前向 -> loss -> 回传 -> 更新
    model.train()
    for step in range(STEPS):
        x, y = get_batch(data, BATCH_SIZE, SEQ_LEN, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            print(f"[3] step {step:>4} | loss {loss.item():.3f}")

    # 5) 保存模型（以后的实验可以加载它）
    checkpoint_path = DATA_DIR / "gpt_final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[4] 模型已保存到 {checkpoint_path}")

    # 6) 接龙采样：给一个开头，让模型自己往下写
    prompt_ids = encode(PROMPT, stoi)
    generated_ids = generate(
        model, prompt_ids, MAX_NEW_TOKENS, TEMPERATURE, device
    )
    print(f"[5] 总耗时 {time.time() - start_time:.0f} 秒")
    print("\n" + decode(generated_ids, itos))


if __name__ == "__main__":
    main()
