import tiktoken
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):

    def __init__(self, num_embed, num_head, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
                           embed_dim=num_embed,
                           num_heads=num_head,
                           dropout=dropout,
                           bias=False,
                           batch_first=True)
        self.mask=None
        dim_feedforward = 4 * num_embed
        layer_norm_eps =  1e-5
        self.linear1 = nn.Linear(num_embed, dim_feedforward, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, num_embed, bias=False)
        self.norm1 = nn.LayerNorm(num_embed, eps=layer_norm_eps, bias=False)
        self.norm2 = nn.LayerNorm(num_embed, eps=layer_norm_eps, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x):
        if self.mask is None:
            self.mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), x.device)
        x = self.self_attn(x, x, x, attn_mask=self.mask, need_weights=False, is_causal=True)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class Transformer(nn.Module):

    def __init__(self, vocab_size, block_size, num_embed, num_head, num_layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, num_embed)
        self.position_embedding = nn.Embedding(block_size, num_embed)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TransformerEncoderLayer(num_embed, num_head, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(num_embed, eps=1e-5, bias=False)
        self.lm_head = nn.Linear(num_embed, vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, vanilla=True):
        device = x.device
        _, t = x.size()
        pos = torch.arange(0, t, dtype=torch.int64, device=device)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        # Token embedding flavoring
        x[:,:,-5:-1] = 1.0 if vanilla else -1.0
        #if not vanilla: x = -x

        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return self.lm_head(x)

    def count_parameters(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        params -= self.token_embedding.weight.numel()
        params -= self.position_embedding.weight.numel()
        return params

    @torch.no_grad()
    def generate(self, idx, vanilla, temperature=0.8, top_k=200):
        if idx.size(1) != self.block_size:
            padded = torch.zeros((idx.size(0), self.block_size), dtype=idx.dtype, device=idx.device)
            num, start = min(idx.size(1), self.block_size), max(0, idx.size(1) - self.block_size)
            padded[:, -num:] = idx[:, start:]
            logits = self(padded, vanilla)
        else:
            logits = self(idx, vanilla)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = self.softmax(logits)
        idx_next = torch.multinomial(probs, num_samples=1)
        return torch.cat((idx, idx_next), dim=1)
