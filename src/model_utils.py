"""Model loading, batched greedy generation, and the feature pass.

The feature pass runs the model once on (prompt + answer) and pulls out:
  - mean log-prob of the answer tokens (baseline 1)
  - hidden states at every layer for the last question token and the last answer token (probe input)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .features import cut_at_eos
from .prompts import MODELS


def load_model(name):
    model_id = MODELS[name]
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"                     # needed for batched generation
    if tok.pad_token is None:                     # Llama has no pad token
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.pad_token_id}
    return tok, model, stop_ids


@torch.no_grad()
def generate_batch(tok, model, stop_ids, message_lists, max_new_tokens):
    """Greedy decode a batch. Returns (prompt_ids, answer_ids, answer_texts), all unpadded."""
    texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in message_lists]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    plen = enc["input_ids"].shape[1]
    prompt_ids, answer_ids, answers = [], [], []
    for i in range(len(texts)):
        p = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
        a = cut_at_eos(out[i, plen:].tolist(), stop_ids)
        prompt_ids.append(p)
        answer_ids.append(a)
        answers.append(tok.decode(a, skip_special_tokens=True).strip())
    return prompt_ids, answer_ids, answers


@torch.no_grad()
def feature_pass(model, prompt_ids, answer_ids):
    """One forward pass on prompt + answer.

    Returns mean_logprob (None if the answer is empty) and two arrays of shape
    (num_layers + 1, hidden) in float16: hidden at the last question token and at the last answer token.
    """
    full = torch.tensor([prompt_ids + answer_ids], device=model.device)
    out = model(full, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0]          # (layers+1, seq, hidden)
    q_pos = len(prompt_ids) - 1
    a_pos = full.shape[1] - 1
    h_q = hs[:, q_pos].float().cpu().numpy().astype("float16")
    h_a = hs[:, a_pos].float().cpu().numpy().astype("float16")

    mean_lp = None
    if answer_ids:
        logits = out.logits[0, q_pos:a_pos].float()            # position t predicts token t+1
        lp = torch.log_softmax(logits, dim=-1)
        tgt = full[0, q_pos + 1:a_pos + 1]
        mean_lp = lp.gather(1, tgt[:, None]).mean().item()
    return mean_lp, h_q, h_a


@torch.no_grad()
def sample_batch(tok, model, stop_ids, message_lists, max_new_tokens, k, temperature=1.0):
    """k sampled answers per prompt (plain temperature sampling, no top-k/top-p).

    Returns a list (one per prompt) of (answers, n_hit_limit)."""
    texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in message_lists]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                         top_p=1.0, top_k=0, num_return_sequences=k, pad_token_id=tok.pad_token_id)
    plen = enc["input_ids"].shape[1]
    res = []
    for i in range(len(texts)):
        answers, hit = [], 0
        for j in range(k):
            gen = out[i * k + j, plen:].tolist()
            a = cut_at_eos(gen, stop_ids)
            hit += len(a) >= max_new_tokens
            answers.append(tok.decode(a, skip_special_tokens=True).strip())
        res.append((answers, hit))
    return res
