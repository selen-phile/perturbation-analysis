import os
import ast
import gc
import math
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama": {"model_name": "meta-llama/Meta-Llama-3.1-8B"},
    "qwen":  {"model_name": "Qwen/Qwen2.5-7B"},
    "gemma": {"model_name": "google/gemma-2-9b"},
}

DATASETS = {
    "squad": {
        "test_csv":   "dataset/squad_3_column_test.csv",
        "robust_csv": "dataset/squad_test_robustness_all.csv",
        "schema":     "generic",
    },
    "ms_marco": {
        "test_csv":   "dataset/ms_marco_3_column_test.csv",
        "robust_csv": "dataset/ms_marco_test_robustness_all.csv",
        "schema":     "generic",
    },
    "qtsumm": {
        "test_csv":   "dataset/qtsumm_test.csv",
        "robust_csv": "dataset/qtsumm_test_robustness_all.csv",
        "schema":     "qtsumm",
    },
}

NOISE_CONDITIONS = [
    "noise_1_character",
    "noise_2_word_dropout",
    "noise_3_token_shuffle",
    "noise_4_semantic",
    "noise_5_adversarial",
]

METRICS = [
    "entropy",
    "entropy_norm",
    "query_mass",
    "query_mass_per_tok",
    "context_mass",
    "vnorm_query_mass",
]

# (text before context, text between context and query, text after query)
TEMPLATES = {
    "generic": ("Context: ", "\n\nQuery: ", "\n\nAnswer:"),
    "qtsumm":  ("Table Data:\n", "\nQuery: ", "\n\nSummary:"),
}

MAX_LEN = 512            # hard cap on total prompt tokens (no right-truncation)
QUERY_BUDGET = 128       # tokens reserved for query + template text
TEMPLATE_OVERHEAD = 32   # BOS + fixed template strings
CTX_BUDGET = MAX_LEN - QUERY_BUDGET - TEMPLATE_OVERHEAD

OUTPUT_DIR = "attention_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def render_context(schema, raw):
    """Context/table as text. Same text for clean and noisy prompts."""
    if schema == "qtsumm":
        try:
            table_data = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            table_data = {"title": "", "headers": [], "data": []}
        title = table_data.get("title", "")
        headers = table_data.get("header", table_data.get("headers", []))
        rows = table_data.get("rows", table_data.get("data", []))
        md = ""
        if title:
            md += f"### {title}\n\n"
        if headers:
            md += "| " + " | ".join(str(h) for h in headers) + " |\n"
            md += "|" + "|".join(["---"] * len(headers)) + "|\n"
        for r in rows:
            md += "| " + " | ".join(str(c) for c in r) + " |\n"
        return md
    return str(raw)[:1000]


def truncate_text(tokenizer, text, max_tokens):
    """Truncate the CONTEXT (never the query / answer cue) to a token budget."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


def build_prompt(schema, ctx, query):
    """Returns prompt and the (start_char, end_char) span of the query text."""
    pre, mid, post = TEMPLATES[schema]
    prefix = pre + ctx + mid
    query = str(query)
    return prefix + query + post, len(prefix), len(prefix) + len(query)


def encode_with_span(tokenizer, prompt, char_start, char_end):
    """Tokenize (no truncation) and map the query char span to token span
    [tok_start, tok_end). Returns an empty span (0, 0) if the query has no
    tokens (e.g. an empty perturbed query)."""
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    idxs = [
        i for i, (s, e) in enumerate(offsets)
        if e > s and s < char_end and e > char_start   # skips special tokens
    ]
    if not idxs:
        return enc, 0, 0
    return enc, idxs[0], idxs[-1] + 1


# --------------------------------------------------------------------------
# Value-norm hooks (for the norm-based attention measure)
# --------------------------------------------------------------------------
def register_value_hooks(model, store):
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    group = n_heads // n_kv
    handles = []
    for li, layer in enumerate(model.model.layers):
        def hook(mod, inp, out, li=li):
            v = out[0].float().view(-1, n_kv, head_dim)            # [S, KV, hd]
            vn = v.norm(dim=-1).repeat_interleave(group, dim=1)    # [S, H]
            store[li] = vn.T.contiguous()                          # [H, S]
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook))
    return handles


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def analyze_attentions(attentions, vnorms, q_start, q_end):
    """
    attentions: tuple of [1, H, S, S] tensors (one per layer)
    vnorms:     list of [H, S] value-norm tensors (one per layer)
    Returns dict metric -> [L, H] CPU tensor. Everything is measured at the
    last token's attention row.
    """
    eps = 1e-12
    out = {m: [] for m in METRICS}
    n_q = max(q_end - q_start, 1)
    for l, attn in enumerate(attentions):
        last = attn[0, :, -1, :].float()                  # [H, S]
        _, S = last.shape
        ent = -(last * (last + eps).log()).sum(-1)        # [H]
        qm = last[:, q_start:q_end].sum(-1)
        w = last * vnorms[l].to(last.device)              # [H, S]
        out["entropy"].append(ent)
        out["entropy_norm"].append(ent / math.log(max(S, 2)))
        out["query_mass"].append(qm)
        out["query_mass_per_tok"].append(qm / n_q)
        out["context_mass"].append(last[:, :q_start].sum(-1))
        out["vnorm_query_mass"].append(w[:, q_start:q_end].sum(-1) / (w.sum(-1) + eps))
    return {m: torch.stack([t.cpu() for t in v]) for m, v in out.items()}


def forward_stats(model, tokenizer, prompt, cs, ce, device, store, num_layers):
    enc, qs, qe = encode_with_span(tokenizer, prompt, cs, ce)
    seq_len = enc["input_ids"].shape[1]
    if seq_len > MAX_LEN:
        return None
    enc = {k: v.to(device) for k, v in enc.items()}
    store.clear()
    out = model(**enc, output_attentions=True)
    vnorms = [store[l] for l in range(num_layers)]
    stats = analyze_attentions(out.attentions, vnorms, qs, qe)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return stats, logp, qe - qs, seq_len


# --------------------------------------------------------------------------
# Main per-(model, dataset) loop
# --------------------------------------------------------------------------
def run_pair(model_key, dataset_key, num_samples, seed):
    model_cfg, dataset_cfg = MODELS[model_key], DATASETS[dataset_key]
    schema = dataset_cfg["schema"]
    doc_col = "table" if schema == "qtsumm" else "document"

    print(f"\n=== Attention analysis: {model_key} / {dataset_key} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"], dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",  # required for output_attentions
    )
    model.eval()
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    df_test = pd.read_csv(dataset_cfg["test_csv"])
    df_robust = pd.read_csv(dataset_cfg["robust_csv"])
    total = min(len(df_test), len(df_robust))
    df_test, df_robust = df_test.head(total), df_robust.head(total)

    # Sanity check that the two CSVs are row-aligned.
    for qcol in ("query", "question"):
        if qcol in df_test.columns:
            mism = (df_test[qcol].astype(str).str.strip().values
                    != df_robust["original_query"].astype(str).str.strip().values).sum()
            if mism:
                print(f"  WARNING: {mism}/{total} rows where test '{qcol}' != robust 'original_query'")
            break

    # Random (seeded) subsample instead of head(n), to avoid ordering bias.
    rng = np.random.RandomState(seed)
    n = min(num_samples, total)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    print(f"Using {n} of {total} samples (seed={seed})")

    store = {}
    handles = register_value_hooks(model, store)

    zeros = lambda: torch.zeros(num_layers, num_heads, dtype=torch.float64)
    acc = {c: {m: {k: zeros() for k in ("c", "n", "d", "abs", "sq")} for m in METRICS}
           for c in NOISE_CONDITIONS}
    counts = {c: 0 for c in NOISE_CONDITIONS}
    sample_rows = []
    n_skipped = 0

    with torch.no_grad():
        for k, i in enumerate(idx):
            ctx = truncate_text(tokenizer, render_context(schema, df_test.loc[i, doc_col]), CTX_BUDGET)
            clean_q = str(df_robust.loc[i, "original_query"])
            p, cs, ce = build_prompt(schema, ctx, clean_q)
            res = forward_stats(model, tokenizer, p, cs, ce, device, store, num_layers)
            if res is None:
                n_skipped += 1
                continue
            c_stats, c_logp, c_nq, c_len = res

            for cond in NOISE_CONDITIONS:
                noisy_q = str(df_robust.loc[i, cond])
                p, ns, ne = build_prompt(schema, ctx, noisy_q)
                res = forward_stats(model, tokenizer, p, ns, ne, device, store, num_layers)
                if res is None:
                    n_skipped += 1
                    continue
                n_stats, n_logp, n_nq, n_len = res

                for m in METRICS:
                    c, nz = c_stats[m].double(), n_stats[m].double()
                    d = nz - c
                    a = acc[cond][m]
                    a["c"] += c; a["n"] += nz; a["d"] += d
                    a["abs"] += d.abs(); a["sq"] += d ** 2
                counts[cond] += 1

                # Output-level change for this sample (links internals to outputs)
                p_c = c_logp.exp()
                sample_rows.append({
                    "model": model_key, "dataset": dataset_key, "condition": cond,
                    "sample_idx": int(i),
                    "n_query_tok_clean": c_nq, "n_query_tok_noisy": n_nq,
                    "prompt_len_clean": c_len, "prompt_len_noisy": n_len,
                    "kl_clean_to_noisy": float((p_c * (c_logp - n_logp)).sum().item()),
                    "top1_changed": int(c_logp.argmax().item() != n_logp.argmax().item()),
                })

            if k % 10 == 0:
                torch.cuda.empty_cache()
                print(f"  [{k}/{n}]")

    for h in handles:
        h.remove()

    records = []
    for cond in NOISE_CONDITIONS:
        cnt = counts[cond]
        if cnt == 0:
            continue
        for l in range(num_layers):
            for h in range(num_heads):
                rec = {"model": model_key, "dataset": dataset_key, "condition": cond,
                       "layer": l, "head": h, "n_samples": cnt}
                for m in METRICS:
                    a = acc[cond][m]
                    mean_d = a["d"][l, h].item() / cnt
                    var = max(a["sq"][l, h].item() / cnt - mean_d ** 2, 0.0)
                    rec[f"{m}_clean"] = a["c"][l, h].item() / cnt
                    rec[f"{m}_noisy"] = a["n"][l, h].item() / cnt
                    rec[f"{m}_delta"] = mean_d
                    rec[f"{m}_absdelta"] = a["abs"][l, h].item() / cnt
                    rec[f"{m}_delta_std"] = math.sqrt(var)
                records.append(rec)

    print(f"Skipped {n_skipped} (sample, condition) pairs exceeding MAX_LEN={MAX_LEN}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records, sample_rows


# --------------------------------------------------------------------------
# Summaries: full per-model rankings, consistency, per-layer profile
# --------------------------------------------------------------------------
def summarize(grand_path, top_k=20, metrics=("query_mass", "vnorm_query_mass")):
    df = pd.read_csv(grand_path)
    for metric in metrics:
        col = f"{metric}_absdelta"
        if col not in df.columns:
            continue
        profiles = []
        for model, sub in df.groupby("model"):
            # Per-model (NOT averaged across models): full head distribution
            rank = (sub.groupby(["layer", "head"])[col].mean()
                       .reset_index().sort_values(col, ascending=False))
            rank.to_csv(os.path.join(OUTPUT_DIR, f"head_ranking_{model}_{metric}.csv"), index=False)

            # Consistency: fraction of (dataset, condition) settings in which a
            # head is in the top-k.
            counter, n_settings = Counter(), 0
            for _, s in sub.groupby(["dataset", "condition"]):
                n_settings += 1
                for l, h in s.nlargest(top_k, col)[["layer", "head"]].itertuples(index=False):
                    counter[(l, h)] += 1
            cons = pd.DataFrame(
                [{"layer": l, "head": h, "frac_settings_in_topk": c / n_settings}
                 for (l, h), c in counter.items()]
            ).sort_values("frac_settings_in_topk", ascending=False)
            cons.to_csv(os.path.join(OUTPUT_DIR, f"head_consistency_{model}_{metric}.csv"), index=False)

            n_layers = sub["layer"].max() + 1
            lp = sub.groupby("layer")[col].mean().reset_index()
            lp["model"] = model
            lp["rel_depth"] = lp["layer"] / (n_layers - 1)
            profiles.append(lp)
        pd.concat(profiles).to_csv(os.path.join(OUTPUT_DIR, f"layer_profile_{metric}.csv"), index=False)
    print("Summaries written to", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Samples per (model, dataset); attentions are streamed, not stored")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()

    grand_path = os.path.join(OUTPUT_DIR, "grand_attention.csv")
    samples_path = os.path.join(OUTPUT_DIR, "grand_samples.csv")

    if not args.summarize_only:
        all_records, all_samples = [], []
        for model_key in args.models:
            for dataset_key in args.datasets:
                pair_path = os.path.join(OUTPUT_DIR, f"attention_{model_key}_{dataset_key}.csv")
                samp_path = os.path.join(OUTPUT_DIR, f"samples_{model_key}_{dataset_key}.csv")
                if os.path.exists(pair_path) and os.path.exists(samp_path):
                    print(f"Skipping {model_key}/{dataset_key} - already done.")
                    all_records.extend(pd.read_csv(pair_path).to_dict("records"))
                    all_samples.extend(pd.read_csv(samp_path).to_dict("records"))
                    continue
                records, sample_rows = run_pair(model_key, dataset_key, args.num_samples, args.seed)
                pd.DataFrame(records).to_csv(pair_path, index=False)
                pd.DataFrame(sample_rows).to_csv(samp_path, index=False)
                all_records.extend(records)
                all_samples.extend(sample_rows)
                pd.DataFrame(all_records).to_csv(grand_path, index=False)
                pd.DataFrame(all_samples).to_csv(samples_path, index=False)
                print(f"Saved: {pair_path}")

    summarize(grand_path)
    print(f"\nDone. Grand file: {grand_path}")


if __name__ == "__main__":
    main()
