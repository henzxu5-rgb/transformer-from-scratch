from pathlib import Path

import torch


DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "input.txt"


def load_data(data_path: Path = DATA_PATH):
    if not data_path.exists():
        raise FileNotFoundError(
            f"cannot find {data_path}; put the Tiny Shakespeare text at this path"
        )

    text = data_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for index, char in enumerate(chars)}
    token_ids = torch.tensor(encode(text, stoi), dtype=torch.long)
    return token_ids, stoi, itos


def encode(text, stoi):
    unknown = sorted(set(text) - set(stoi))
    if unknown:
        raise ValueError(f"characters not in vocabulary: {unknown}")
    return [stoi[char] for char in text]


def decode(token_ids, itos):
    return "".join(itos[int(index)] for index in token_ids)


def get_batch(data, batch_size, seq_len, device="cpu"):
    if data.numel() <= seq_len:
        raise ValueError("data must contain more tokens than seq_len")

    # y 比 x 向右移动一个字符
    starts = torch.randint(0, data.numel() - seq_len, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in starts])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


def main():
    data, stoi, itos = load_data()
    print(f"[data] 字符数 = {data.numel():,}")
    print(f"[data] 词表大小 = {len(stoi)}")

    x, y = get_batch(data, batch_size=1, seq_len=20)
    print(f"[data] x 形状 {tuple(x.shape)} | y 形状 {tuple(y.shape)}")
    print(f"[data] 输入: {decode(x[0], itos)!r}")
    print(f"[data] 答案: {decode(y[0], itos)!r}")
    print(f"[data] 右移检查: {torch.equal(x[:, 1:], y[:, :-1])}")


if __name__ == "__main__":
    main()
