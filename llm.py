#!/usr/bin/env python3
"""
llm.py — Minimal GPT-style language model from scratch.

Quick start:
    # 1. Get training data (tiny Shakespeare, ~1MB)
    python llm.py download

    # 2. Train
    python llm.py train --data input.txt

    # 3. Generate
    python llm.py generate "HAMLET:"

Tune the model size:
    python llm.py train --data input.txt --embd 256 --layers 6 --heads 8 --context 512 --iters 10000
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    # Architecture
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 256
    dropout: float = 0.1
    vocab_size: int = 0  # set after tokenizer is built

    # Training
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    lr: float = 3e-4
    lr_min: float = 1e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # I/O
    checkpoint: str = "model.pt"
    tokenizer_path: str = "tokenizer.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Tokenizer (character-level)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Maps every unique character in the training corpus to an integer."""

    def __init__(self):
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(enumerate(chars))
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tok.stoi = d["stoi"]
        tok.itos = {int(k): v for k, v in d["itos"].items()}
        tok.vocab_size = len(tok.stoi)
        return tok


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # project to Q, K, V in one shot
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # causal (lower-triangular) mask — tokens can only attend to earlier positions
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        nh, hd = self.n_head, self.head_dim

        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, nh, hd).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer layer: pre-norm attention + pre-norm feed-forward."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying: output projection shares weights with token embeddings
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual stream projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Sequence length {T} > block_size {self.cfg.block_size}"

        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT, train: torch.Tensor, val: torch.Tensor, cfg: Config
) -> dict[str, float]:
    model.eval()
    results = {}
    for split, data in [("train", train), ("val", val)]:
        losses = [
            model(*get_batch(data, cfg.block_size, cfg.batch_size, cfg.device))[1].item()
            for _ in range(cfg.eval_iters)
        ]
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def cosine_lr(step: int, cfg: Config) -> float:
    """Linear warmup then cosine decay."""
    if step < cfg.warmup_iters:
        return cfg.lr * step / max(1, cfg.warmup_iters)
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_download(_args: argparse.Namespace) -> None:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dest = "input.txt"
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping download.")
        return
    print(f"Downloading tiny Shakespeare (~1 MB) …")
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"Saved to {dest} ({size:,} bytes)")
    print(f"\nNext: python llm.py train --data {dest}")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.update(
        n_embd=args.embd,
        n_head=args.heads,
        n_layer=args.layers,
        block_size=args.context,
        max_iters=args.iters,
        batch_size=args.batch,
        lr=args.lr,
        checkpoint=args.checkpoint,
    )

    print(f"Loading {args.data} …")
    text = open(args.data, encoding="utf-8").read()
    print(f"  {len(text):,} characters")

    tok = CharTokenizer()
    tok.fit(text)
    tok.save(cfg.tokenizer_path)
    cfg.vocab_size = tok.vocab_size
    print(f"  Vocab size : {cfg.vocab_size}")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"  Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    model = GPT(cfg).to(cfg.device)
    n = model.num_params()
    print(f"  Parameters : {n:,}  ({n / 1e6:.2f}M)")
    print(f"  Device     : {cfg.device}")
    print(f"  Architecture: {cfg.n_layer}L × {cfg.n_head}H × {cfg.n_embd}D, ctx={cfg.block_size}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    t0 = time.time()
    for step in range(cfg.max_iters + 1):
        lr = cosine_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg)
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{cfg.max_iters} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )
            torch.save({"model": model.state_dict(), "config": cfg.__dict__}, cfg.checkpoint)

        if step == cfg.max_iters:
            break

        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    print(f"\nTraining complete. Checkpoint: {cfg.checkpoint}")


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.checkpoint = args.checkpoint

    if not os.path.exists(cfg.checkpoint):
        sys.exit(f"Checkpoint not found: {cfg.checkpoint}\n  Train first: python llm.py train --data input.txt")
    if not os.path.exists(cfg.tokenizer_path):
        sys.exit(f"Tokenizer not found: {cfg.tokenizer_path}")

    tok = CharTokenizer.load(cfg.tokenizer_path)

    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    for k in ("n_embd", "n_head", "n_layer", "block_size", "vocab_size", "dropout"):
        setattr(cfg, k, saved[k])

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(cfg.device)

    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        sys.exit("Prompt contains no characters in the vocabulary.")
    ctx = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)
    out = model.generate(ctx, args.tokens, temperature=args.temp, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


def cmd_info(args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.checkpoint = args.checkpoint
    if not os.path.exists(cfg.checkpoint):
        sys.exit(f"Checkpoint not found: {cfg.checkpoint}")
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    for k in ("n_embd", "n_head", "n_layer", "block_size", "vocab_size", "dropout"):
        setattr(cfg, k, saved[k])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    n = model.num_params()
    print("Model info")
    print(f"  Parameters : {n:,}  ({n / 1e6:.2f}M)")
    print(f"  Layers     : {cfg.n_layer}")
    print(f"  Heads      : {cfg.n_head}")
    print(f"  Embed dim  : {cfg.n_embd}")
    print(f"  Context    : {cfg.block_size} tokens")
    print(f"  Vocab size : {cfg.vocab_size}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal GPT from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # download
    sub.add_parser("download", help="Download tiny Shakespeare training data")

    # train
    tr = sub.add_parser("train", help="Train a model on a text file")
    tr.add_argument("--data", required=True, help="Path to training text (.txt)")
    tr.add_argument("--embd",    type=int,   default=128,    help="Embedding dim        (default 128)")
    tr.add_argument("--heads",   type=int,   default=4,      help="Attention heads      (default 4)")
    tr.add_argument("--layers",  type=int,   default=4,      help="Transformer layers   (default 4)")
    tr.add_argument("--context", type=int,   default=256,    help="Context window       (default 256)")
    tr.add_argument("--iters",   type=int,   default=5000,   help="Training iterations  (default 5000)")
    tr.add_argument("--batch",   type=int,   default=32,     help="Batch size           (default 32)")
    tr.add_argument("--lr",      type=float, default=3e-4,   help="Peak learning rate   (default 3e-4)")
    tr.add_argument("--checkpoint", default="model.pt", help="Output checkpoint file")

    # generate
    ge = sub.add_parser("generate", help="Generate text from a trained model")
    ge.add_argument("prompt",         help="Seed text for generation")
    ge.add_argument("--tokens", type=int,   default=200,  help="Tokens to generate  (default 200)")
    ge.add_argument("--temp",   type=float, default=0.8,  help="Temperature         (default 0.8)")
    ge.add_argument("--top-k",  type=int,   default=40,   help="Top-k sampling      (default 40, 0=off)")
    ge.add_argument("--checkpoint", default="model.pt", help="Checkpoint to load")

    # info
    inf = sub.add_parser("info", help="Print info about a saved checkpoint")
    inf.add_argument("--checkpoint", default="model.pt")

    args = parser.parse_args()
    {"download": cmd_download, "train": cmd_train, "generate": cmd_generate, "info": cmd_info}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
