<<<<<<< HEAD
import torch
import torch.nn.functional as F

def nucleus_sampling_decode(
    model, src_ids, src_mask, tokenizer,
    max_len=30, p=0.9, temperature=1.0, repetition_penalty=1.1
):
    device = src_ids.device
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("</s>")
    bos_id = tokenizer.token_to_id("<s>")

    generated = torch.tensor([[bos_id]], device=device)
    memory = model.transformer.encoder(
        model.pos_encoder(model.embedding(src_ids)),
        src_key_padding_mask=(src_mask == 0)
    )
    past_tokens = set()

    for _ in range(max_len):
        tgt = model.pos_encoder(model.embedding(generated))
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)

        decoder_output = model.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=(src_mask == 0),
            tgt_key_padding_mask=(generated == pad_id)
        )

        logits = model.output_layer(decoder_output[:, -1, :])  # (batch_size, vocab_size)

        # 🔥 Température
        logits = logits / temperature

        # Pénalité de répétition
        for token_id in past_tokens:
            logits[0, token_id] /= repetition_penalty

        probs = torch.softmax(logits, dim=-1)

        #Nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0
        sorted_probs /= sorted_probs.sum()

        next_token = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices.gather(-1, next_token)

        past_tokens.add(next_token.item())
        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == eos_id:
            break

    return tokenizer.decode(generated.squeeze().tolist(), skip_special_tokens=True)
=======
import torch
import torch.nn.functional as F

def nucleus_sampling_decode(
    model, src_ids, src_mask, tokenizer,
    max_len=30, p=0.9, temperature=1.0, repetition_penalty=1.1
):
    device = src_ids.device
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("</s>")
    bos_id = tokenizer.token_to_id("<s>")

    generated = torch.tensor([[bos_id]], device=device)
    memory = model.transformer.encoder(
        model.pos_encoder(model.embedding(src_ids)),
        src_key_padding_mask=(src_mask == 0)
    )
    past_tokens = set()

    for _ in range(max_len):
        tgt = model.pos_encoder(model.embedding(generated))
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)

        decoder_output = model.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=(src_mask == 0),
            tgt_key_padding_mask=(generated == pad_id)
        )

        logits = model.output_layer(decoder_output[:, -1, :])  # (batch_size, vocab_size)

        # 🔥 Température
        logits = logits / temperature

        # Pénalité de répétition
        for token_id in past_tokens:
            logits[0, token_id] /= repetition_penalty

        probs = torch.softmax(logits, dim=-1)

        #Nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0
        sorted_probs /= sorted_probs.sum()

        next_token = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices.gather(-1, next_token)

        past_tokens.add(next_token.item())
        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() == eos_id:
            break

    return tokenizer.decode(generated.squeeze().tolist(), skip_special_tokens=True)
>>>>>>> ecbe693 (Mise à jour depuis EC2 : dernières modifs locales)
