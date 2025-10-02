"""
Reduced Transformer implementation with 43 debugger checkpoints.
Each checkpoint is marked with the step number and description.
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, values, keys, query, mask=None, debug=False, tag=""):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # (7) Queries from input (Q)
        queries = self.queries(query)
        if debug: breakpoint()

        # (8) Keys from input (K)
        keys = self.keys(keys)
        if debug: breakpoint()

        # (9) Values from input (V)
        values = self.values(values)
        if debug: breakpoint()

        # (12) Split Q, K, V into heads (multi-head structure)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        if debug: breakpoint()

        # (10) Raw attention scores = QK^T (before softmax)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if debug: breakpoint()

        if mask is not None:
            # (25) Apply mask to block future/pad tokens
            if debug: breakpoint()
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # (11) Attention weights after softmax
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        if debug: breakpoint()

        attention = self.dropout(attention)

        # (13) Weighted sum of values, concat heads
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        if debug: breakpoint()

        # Linear projection back to embedding dim
        out = self.fc_out(out)
        return out, queries, keys, values, energy, attention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None, debug=False):
        attention_out, Q, K, V, energy, att = self.attention(value, key, query, mask, debug=debug)

        # (14) Residual connection (attention_out + query)
        if debug: breakpoint()
        x = self.dropout(self.norm1(attention_out + query))

        # (15) After layer normalization
        if debug: breakpoint()

        # (16) Feed-forward input
        forward_in = x
        if debug: breakpoint()

        # (17) FFN first linear output
        ff1 = self.feed_forward[0](forward_in)
        if debug: breakpoint()

        ff1 = self.feed_forward[1](ff1)  # ReLU

        # (18) FFN second linear output
        ff2 = self.feed_forward[2](ff1)
        if debug: breakpoint()

        # (19) Encoder block final output
        out = self.dropout(self.norm2(ff2 + x))
        if debug: breakpoint()

        return out, (Q, K, V, energy, att, attention_out, forward_in, ff1, ff2)


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, debug=False):
        N, seq_length = x.shape

        # (1) Raw source input tokens (IDs)
        if debug: breakpoint()

        # (3) Slice of embedding weight matrix (sanity check)
        embed_weights = self.word_embedding.weight[:5, :5]
        if debug: breakpoint()

        # (4) Token embeddings
        word_emb = self.word_embedding(x)
        if debug: breakpoint()

        # (5) Embeddings + positional encodings
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)
        pos_emb = self.position_embedding(positions)
        out = self.dropout(word_emb + pos_emb)
        if debug: breakpoint()

        for layer in self.layers:
            # (6) Encoder block input
            if debug: breakpoint()
            out, dbg = layer(out, out, out, mask, debug=debug)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads, dropout=dropout)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask=None, trg_mask=None, debug=False):
        att_out, Qd, Kd, Vd, energy_d, att_d = self.attention(x, x, x, trg_mask, debug=debug)

        # (21) Masked self-attn Q
        if debug: breakpoint()
        # (22) Masked self-attn K
        if debug: breakpoint()
        # (23) Masked self-attn V
        if debug: breakpoint()
        # (24) Raw masked energy (before applying mask)
        if debug: breakpoint()
        # (25) Mask tensor applied
        if debug: breakpoint()
        # (26) Masked energy after softmax
        if debug: breakpoint()
        # (27/28) Concat masked self-attention output
        if debug: breakpoint()

        query = self.dropout(self.norm(att_out + x))

        # (29) Residual + norm after masked self-attn
        if debug: breakpoint()

        out, dbg = self.transformer_block(value, key, query, src_mask, debug=debug)

        # Cross-attention debug trace:
        # (30) Cross-attn queries from decoder
        if debug: breakpoint()
        # (31) Cross-attn keys from encoder
        if debug: breakpoint()
        # (32) Cross-attn values from encoder
        if debug: breakpoint()
        # (33) Cross-attn raw energy
        if debug: breakpoint()
        # (34) Cross-attn after softmax
        if debug: breakpoint()
        # (35) Cross-attn concatenated output
        if debug: breakpoint()
        # (36) Residual + norm after cross-attn
        if debug: breakpoint()
        # (37) Decoder FF input
        if debug: breakpoint()
        # (38) Decoder FF first linear
        if debug: breakpoint()
        # (39) Decoder FF second linear
        if debug: breakpoint()
        # (40) Decoder block final output
        if debug: breakpoint()

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None, debug=False):
        N, seq_length = x.shape

        # (2) Target input tokens (IDs)
        if debug: breakpoint()

        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # (20) Decoder input embeddings + positional
        if debug: breakpoint()

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask, debug=debug)

        out_before_proj = x
        # (41) Decoder output sequence before final projection
        if debug: breakpoint()

        out = self.fc_out(x)
        # (42) Logits over vocabulary
        if debug: breakpoint()

        # (43) Logits slice (sanity check)
        if debug: breakpoint()

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=128,
        num_layers=2,
        forward_expansion=4,
        heads=4,
        dropout=0.1,
        device="cpu",
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)
        trg_mask = trg_pad_mask & subsequent_mask
        return trg_mask.to(self.device)

    def forward(self, src, trg, debug=False):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask, debug=debug)
        out = self.decoder(trg, enc_src, src_mask, trg_mask, debug=debug)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0]]).to(device)

    model = Transformer(50, 50, 0, 0, device=device).to(device)
    out = model(x, trg[:, :-1], debug=True)
    print("Final output shape:", out.shape)
