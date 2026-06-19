from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LoraInitBank = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name or ""
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rank = 0
        self.alpha = 1.0
        self.lora_A: nn.Parameter | None = None
        self.lora_B: nn.Parameter | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = fan_in ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def set_lora(
        self,
        rank: int,
        alpha: float = 1.0,
        init_scale: float = 0.01,
        *,
        a_init: torch.Tensor | None = None,
        b_init: torch.Tensor | None = None,
    ) -> None:
        self.rank = int(rank)
        self.alpha = float(alpha)
        if self.rank <= 0:
            self.lora_A = None
            self.lora_B = None
            return

        if a_init is None:
            a = init_scale * torch.randn(
                self.rank,
                self.in_features,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        else:
            if a_init.ndim != 2 or a_init.shape[0] < self.rank or a_init.shape[1] != self.in_features:
                raise ValueError(
                    f"invalid A initialization {tuple(a_init.shape)} for rank={self.rank}, in={self.in_features}"
                )
            a = a_init[: self.rank].detach().to(device=self.weight.device, dtype=self.weight.dtype).clone()

        if b_init is None:
            b = torch.zeros(
                self.out_features,
                self.rank,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        else:
            if b_init.ndim != 2 or b_init.shape[0] != self.out_features or b_init.shape[1] < self.rank:
                raise ValueError(
                    f"invalid B initialization {tuple(b_init.shape)} for out={self.out_features}, rank={self.rank}"
                )
            b = b_init[:, : self.rank].detach().to(device=self.weight.device, dtype=self.weight.dtype).clone()

        self.lora_A = nn.Parameter(a)
        self.lora_B = nn.Parameter(b)

    def clear_lora(self) -> None:
        self.rank = 0
        self.lora_A = None
        self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            y = y + (self.alpha / self.rank) * F.linear(F.linear(x, self.lora_A, None), self.lora_B, None)
        return y

    def lora_parameter_count(self) -> int:
        if self.rank <= 0:
            return 0
        return self.rank * (self.in_features + self.out_features)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, prefix: str = ""):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = LoRALinear(d_model, d_model, bias=False, name=f"{prefix}.q_proj")
        self.k_proj = LoRALinear(d_model, d_model, bias=False, name=f"{prefix}.k_proj")
        self.v_proj = LoRALinear(d_model, d_model, bias=False, name=f"{prefix}.v_proj")
        self.o_proj = LoRALinear(d_model, d_model, bias=False, name=f"{prefix}.o_proj")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        probs = F.softmax(att, dim=-1)
        probs = self.dropout(probs)
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.0, prefix: str = ""):
        super().__init__()
        self.fc1 = LoRALinear(d_model, d_mlp, name=f"{prefix}.fc1")
        self.fc2 = LoRALinear(d_mlp, d_model, name=f"{prefix}.fc2")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float, idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, prefix=f"blocks.{idx}.attn")
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, dropout, prefix=f"blocks.{idx}.mlp")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int, n_heads: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_mlp, dropout, i) for i in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def build_model(vocab_size: int, seq_len: int, cfg: Dict) -> TinyTransformer:
    return TinyTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        d_mlp=int(cfg["d_mlp"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )


def iter_lora_modules(model: nn.Module) -> Iterable[Tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def clear_all_lora(model: nn.Module) -> None:
    for _, module in iter_lora_modules(model):
        module.clear_lora()


def freeze_base_enable_lora(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for _, module in iter_lora_modules(model):
        if module.rank > 0 and module.lora_A is not None and module.lora_B is not None:
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)


def enable_base_training(model: nn.Module) -> None:
    clear_all_lora(model)
    for p in model.parameters():
        p.requires_grad_(True)


def make_lora_init_bank(
    model: nn.Module,
    max_rank: int,
    seed: int,
    init_scale: float = 0.01,
) -> LoraInitBank:
    """Create per-module maximum-rank tensors for nested rank comparisons.

    Every lower-rank condition receives a prefix slice of the same maximum-rank
    matrices, so changing rank does not also redraw the shared coordinates.
    The bank is kept on CPU and copied into each cloned model.
    """
    max_rank = int(max_rank)
    if max_rank < 0:
        raise ValueError("max_rank must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bank: LoraInitBank = {}
    for name, module in iter_lora_modules(model):
        a = init_scale * torch.randn(
            max_rank,
            module.in_features,
            generator=generator,
            dtype=torch.float32,
        )
        b = torch.zeros(module.out_features, max_rank, dtype=torch.float32)
        bank[name] = (a, b)
    return bank


def set_lora_ranks(
    model: nn.Module,
    ranks: Mapping[str, int],
    alpha: float,
    init_scale: float = 0.01,
    *,
    init_bank: Mapping[str, Tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    clear_all_lora(model)
    for name, module in iter_lora_modules(model):
        r = int(ranks.get(name, 0))
        if init_bank is None or r <= 0:
            module.set_lora(r, alpha=alpha, init_scale=init_scale)
        else:
            if name not in init_bank:
                raise KeyError(f"missing LoRA initialization for module {name}")
            a_init, b_init = init_bank[name]
            module.set_lora(
                r,
                alpha=alpha,
                init_scale=init_scale,
                a_init=a_init,
                b_init=b_init,
            )


def lora_parameters(model: nn.Module):
    for _, module in iter_lora_modules(model):
        if module.rank > 0 and module.lora_A is not None and module.lora_B is not None:
            yield module.lora_A
            yield module.lora_B


def lora_parameter_count(model: nn.Module) -> int:
    return sum(module.lora_parameter_count() for _, module in iter_lora_modules(model))
