import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            pe = self.pe[:, :seq_len]
        else:
            pe = self.pe[:, :self.max_len]

        x = x + pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size, pad_id, d_model=768, nhead=12, num_layers=6,
                 dropout=0.2, use_checkpointing=False):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _run_encoder(self, src, src_key_padding_mask):
        return self.transformer.encoder(
            src,
            mask=None,
            src_key_padding_mask=src_key_padding_mask
        )

    def _run_decoder(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        return self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

    def forward(self, src_ids, tgt_ids, src_mask, tgt_mask):
        src = self.pos_encoder(self.embedding(src_ids))
        tgt = self.pos_encoder(self.embedding(tgt_ids))

        tgt_seq_len = tgt.size(1)
        tgt_attn_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        src_key_padding_mask = (src_mask == 0)
        tgt_key_padding_mask = (tgt_mask == 0)

        if self.use_checkpointing and self.training:
            memory = checkpoint(self._run_encoder, src, src_key_padding_mask)
            output = checkpoint(
                self._run_decoder,
                tgt,
                memory,
                tgt_attn_mask,
                tgt_key_padding_mask,
                src_key_padding_mask
            )
        else:
            memory = self._run_encoder(src, src_key_padding_mask)
            output = self._run_decoder(
                tgt,
                memory,
                tgt_attn_mask,
                tgt_key_padding_mask,
                src_key_padding_mask
            )

        return self.output_layer(output)

    def encode(self, src_ids, src_mask):
        src = self.pos_encoder(self.embedding(src_ids))
        memory = self.transformer.encoder(src, src_key_padding_mask=(src_mask == 0))
        return memory

    def decode(self, src_ids, src_mask, tgt_input_ids):
        memory = self.encode(src_ids, src_mask)
        tgt_mask = self.generate_square_subsequent_mask(tgt_input_ids.size(1)).to(tgt_input_ids.device)
        output = self.transformer.decoder(
            self.embedding(tgt_input_ids) * (self.d_model ** 0.5),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(tgt_input_ids == self.pad_id),
            memory_key_padding_mask=(src_ids == self.pad_id)
        )
        return self.output_layer(output)
