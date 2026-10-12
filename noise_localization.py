"""
noise_localization.py  (fixed version)

Quantifies WHERE (which layer) noise changes internal representations, for the
SAME document/row, clean original_query vs perturbed query.

Why this differs from the first version
---------------------------------------
Clean and perturbed prompts have different lengths (typos, dropout and
paraphrases change the token count), so comparing "token t vs token t" is
ill-defined for the query span, and the first version also truncated to the
first 128 tokens, which for most prompts is just the (identical) context. This
version only compares positions that genuinely correspond:

  prefix   tokens before the query. Token ids are identical, and in a causal
           LM their hidden states should be identical. Any drift here is
           numerical noise (bf16 kernels vary with sequence length), so it is
           reported as a NOISE FLOOR. Real drift must be read against it.
  suffix   template tokens after the query ("\\n\\nAnswer:"). Same token ids,
           aligned from the end; these tokens see the perturbed query through
           attention. Measures how the perturbation is carried downstream.
  last     the final prompt token, i.e. the prediction position.
  query_pooled
           mean-pooled hidden state over the query span (clean vs perturbed).
           Pooling avoids token-by-token alignment when lengths differ.

Metrics per layer: rel_l2 = ||h_c - h_n|| / ||h_c||, cos = 1 - cos_sim,
for each of prefix / suffix / last / query_pooled.

Per (sample, condition) we also record perturbation magnitude (token-level
edit distance of the query), prompt/query lengths, whether the query actually
changed, and output change (KL of the next-token distribution + top-1 flip),
so drift can be controlled for magnitude and related to the outputs.

Layer 0 = embedding output; the LAST entry of hidden_states is POST final norm
in HF models, so it is not on the same scale as earlier layers (flagged by
is_final_hidden).

Output (OUTPUT_DIR):
    layerwise_samples_<model>_<dataset>.csv   per sample x condition x layer
    localization_<model>_<dataset>.csv        per condition x layer mean/std
                                              (only samples whose query changed)
    grand_localization.csv, layer_summary.csv, drift_vs_output.csv

Usage:
    python noise_localization.py --num_samples 100
    python noise_localization.py --summarize_only
"""

import os
import ast
import gc
import math
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

# (text before context, text between context and query, text after query)
TEMPLATES = {
    "generic": ("Context: ", "\n\nQuery: ", "\n\nAnswer:"),
    "qtsumm":  ("Table Data:\n", "\nQuery: ", "\n\nSummary:"),
}

MAX_LEN = 512
QUERY_BUDGET = 128
TEMPLATE_OVERHEAD = 32
CTX_BUDGET = MAX_LEN - QUERY_BUDGET - TEMPLATE_OVERHEAD

REGIONS = ("prefix", "suffix", "last", "query_pooled")
METRIC_COLS = [f"{r}_{m}" for r in REGIONS for m in ("rel_l2", "cos")]

OUTPUT_DIR = "localization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Prompt construction (identical context for clean and noisy; no right-truncation)
# --------------------------------------------------------------------------
def render_context(schema, raw):
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
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens])


def build_prompt(schema, ctx, query):
    pre, mid, post = TEMPLATES[schema]
    prefix = pre + ctx + mid
    query = str(query)
    return prefix + query + post, len(prefix), len(prefix) + len(query)


def encode_with_span(tokenizer, prompt, char_start, char_end):
    """Tokenize without truncation; map the query char span to token span
    [tok_start, tok_end). Empty span (0, 0) if the query has no tokens."""
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    idxs = [i for i, (s, e) in enumerate(offsets)
            if e > s and s < char_end and e > char_start]
    if not idxs:
        return enc, 0, 0
    return enc, idxs[0], idxs[-1] + 1


# --------------------------------------------------------------------------
# Pure-python alignment helpers
# --------------------------------------------------------------------------
def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def common_suffix_len(a, b, max_len):
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y or n >= max_len:
            break
        n += 1
    return n


def edit_distance(a, b):
    """Token-level Levenshtein distance between two id lists."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------
# Drift metrics
# --------------------------------------------------------------------------
def _rel_l2(c, n):
    return ((c - n).norm(dim=-1) / c.norm(dim=-1).clamp_min(1e-8)).mean().item()


def _cos(c, n):
    return (1 - F.cosine_similarity(c, n, dim=-1)).mean().item()


@torch.no_grad()
def layerwise_metrics(hs_c, hs_n, n_prefix, n_suffix, span_c, span_n):
    """hs_*: tuples of [1, S, d] tensors (embedding + every block).
    Only genuinely corresponding positions are compared (see module docstring).
    Returns a list (one dict per layer) of region metrics."""
    out = []
    nan = float("nan")
    (qs_c, qe_c), (qs_n, qe_n) = span_c, span_n
    for hc, hn in zip(hs_c, hs_n):
        hc, hn = hc[0].float(), hn[0].float()      # same device for a given layer
        m = {k: nan for k in METRIC_COLS}
        if n_prefix > 0:
            m["prefix_rel_l2"] = _rel_l2(hc[:n_prefix], hn[:n_prefix])
            m["prefix_cos"] = _cos(hc[:n_prefix], hn[:n_prefix])
        if n_suffix > 0:
            m["suffix_rel_l2"] = _rel_l2(hc[-n_suffix:], hn[-n_suffix:])
            m["suffix_cos"] = _cos(hc[-n_suffix:], hn[-n_suffix:])
        m["last_rel_l2"] = _rel_l2(hc[-1:], hn[-1:])
        m["last_cos"] = _cos(hc[-1:], hn[-1:])
        if qe_c > qs_c and qe_n > qs_n:
            pc = hc[qs_c:qe_c].mean(0, keepdim=True)
            pn = hn[qs_n:qe_n].mean(0, keepdim=True)
            m["query_pooled_rel_l2"] = _rel_l2(pc, pn)
            m["query_pooled_cos"] = _cos(pc, pn)
        out.append(m)
    return out


def forward_pass(model, tokenizer, prompt, cs, ce, device):
    enc, qs, qe = encode_with_span(tokenizer, prompt, cs, ce)
    ids = enc["input_ids"][0].tolist()
    if len(ids) > MAX_LEN:
        return None
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return {"ids": ids, "span": (qs, qe), "hidden": out.hidden_states, "logp": logp}


# --------------------------------------------------------------------------
# Per-(model, dataset) loop
# --------------------------------------------------------------------------
def run_pair(model_key, dataset_key, num_samples, seed):
    model_cfg, dataset_cfg = MODELS[model_key], DATASETS[dataset_key]
    schema = dataset_cfg["schema"]
    doc_col = "table" if schema == "qtsumm" else "document"

    print(f"\n=== Localization: {model_key} / {dataset_key} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"], dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers + 1   # +1 embedding output

    df_test = pd.read_csv(dataset_cfg["test_csv"])
    df_robust = pd.read_csv(dataset_cfg["robust_csv"])
    total = min(len(df_test), len(df_robust))
    df_test, df_robust = df_test.head(total), df_robust.head(total)

    for qcol in ("query", "question"):
        if qcol in df_test.columns:
            mism = (df_test[qcol].astype(str).str.strip().values
                    != df_robust["original_query"].astype(str).str.strip().values).sum()
            if mism:
                print(f"  WARNING: {mism}/{total} rows where test '{qcol}' != robust 'original_query'")
            break

    rng = np.random.RandomState(seed)
    n = min(num_samples, total)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    print(f"Using {n} of {total} samples (seed={seed})")

    rows, n_skipped = [], 0
    with torch.no_grad():
        for k, i in enumerate(idx):
            ctx = truncate_text(tokenizer, render_context(schema, df_test.loc[i, doc_col]), CTX_BUDGET)
            p, cs, ce = build_prompt(schema, ctx, str(df_robust.loc[i, "original_query"]))
            clean = forward_pass(model, tokenizer, p, cs, ce, device)
            if clean is None:
                n_skipped += 1
                continue
            c_ids, (qs_c, qe_c) = clean["ids"], clean["span"]
            p_c = clean["logp"].exp()

            for cond in NOISE_CONDITIONS:
                p, ns, ne = build_prompt(schema, ctx, str(df_robust.loc[i, cond]))
                noisy = forward_pass(model, tokenizer, p, ns, ne, device)
                if noisy is None:
                    n_skipped += 1
                    continue
                n_ids, (qs_n, qe_n) = noisy["ids"], noisy["span"]

                # Alignment: identical prefix, identical template suffix.
                min_len = min(len(c_ids), len(n_ids))
                n_prefix = min(common_prefix_len(c_ids, n_ids), qs_c, qs_n)
                tpl_suffix = max(min(len(c_ids) - qe_c, len(n_ids) - qe_n), 1)
                n_suffix = common_suffix_len(c_ids, n_ids, tpl_suffix)
                n_suffix = max(min(n_suffix, min_len - n_prefix), 0)

                ed = edit_distance(c_ids[qs_c:qe_c], n_ids[qs_n:qe_n])
                query_changed = int(c_ids != n_ids)
                kl = float((p_c * (clean["logp"] - noisy["logp"])).sum().item())
                flip = int(clean["logp"].argmax().item() != noisy["logp"].argmax().item())

                per_layer = layerwise_metrics(
                    clean["hidden"], noisy["hidden"], n_prefix, n_suffix,
                    (qs_c, qe_c), (qs_n, qe_n))
                for l, m in enumerate(per_layer):
                    rows.append({
                        "model": model_key, "dataset": dataset_key, "condition": cond,
                        "sample_idx": int(i), "layer": l,
                        "is_final_hidden": int(l == num_layers - 1),
                        "n_prefix": n_prefix, "n_suffix": n_suffix,
                        "n_query_tok_clean": qe_c - qs_c, "n_query_tok_noisy": qe_n - qs_n,
                        "query_edit_dist": ed,
                        "query_edit_norm": ed / max(qe_c - qs_c, qe_n - qs_n, 1),
                        "query_changed": query_changed,
                        "kl_clean_to_noisy": kl, "top1_changed": flip,
                        **m,
                    })
                del noisy
            del clean
            if k % 10 == 0:
                torch.cuda.empty_cache()
                print(f"  [{k}/{n}]")

    print(f"Skipped {n_skipped} prompts exceeding MAX_LEN={MAX_LEN}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def aggregate(df):
    """Mean/std per (model, dataset, condition, layer) over samples whose query
    actually changed (no-op perturbations would dilute the effect)."""
    chg = df[df["query_changed"] == 1]
    keys = ["model", "dataset", "condition", "layer"]
    g = chg.groupby(keys)
    agg = g[METRIC_COLS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_samples"] = g.size()
    agg["mean_edit_norm"] = g["query_edit_norm"].mean()
    return agg.reset_index()


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def summarize():
    files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith("localization_") and f.endswith(".csv")
                   and f != "grand_localization.csv")
    if not files:
        print("Nothing to summarize.")
        return
    grand = pd.concat([pd.read_csv(os.path.join(OUTPUT_DIR, f)) for f in files])
    n_layers = grand.groupby("model")["layer"].max() + 1
    grand["rel_depth"] = grand["layer"] / (grand["model"].map(n_layers) - 1)
    grand.to_csv(os.path.join(OUTPUT_DIR, "grand_localization.csv"), index=False)

    # Per-model, per-condition layer profile (mean over datasets), with noise floor.
    cols = [c for c in grand.columns if c.endswith("_mean")]
    grand.groupby(["model", "condition", "layer", "rel_depth"])[cols].mean().reset_index() \
         .to_csv(os.path.join(OUTPUT_DIR, "layer_summary.csv"), index=False)

    # Spearman correlation between last-token drift and output KL.
    sfiles = sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith("layerwise_samples_"))
    if sfiles:
        s = pd.concat([pd.read_csv(os.path.join(OUTPUT_DIR, f),
                                   usecols=["model", "condition", "layer", "last_rel_l2",
                                            "kl_clean_to_noisy", "query_changed"]) for f in sfiles])
        s = s[s["query_changed"] == 1]
        corr = (s.groupby(["model", "condition", "layer"])
                  .apply(lambda g: g["last_rel_l2"].rank().corr(g["kl_clean_to_noisy"].rank()))
                  .rename("spearman_last_drift_vs_kl").reset_index())
        corr.to_csv(os.path.join(OUTPUT_DIR, "drift_vs_output.csv"), index=False)
    print("Summaries written to", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100, help="Samples per (model, dataset)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()

    if not args.summarize_only:
        for model_key in args.models:
            for dataset_key in args.datasets:
                agg_path = os.path.join(OUTPUT_DIR, f"localization_{model_key}_{dataset_key}.csv")
                samp_path = os.path.join(OUTPUT_DIR, f"layerwise_samples_{model_key}_{dataset_key}.csv")
                if os.path.exists(agg_path) and os.path.exists(samp_path):
                    print(f"Skipping {model_key}/{dataset_key} - already done.")
                    continue
                df = run_pair(model_key, dataset_key, args.num_samples, args.seed)
                df.to_csv(samp_path, index=False)
                aggregate(df).to_csv(agg_path, index=False)
                print(f"Saved: {agg_path}")
    summarize()


if __name__ == "__main__":
    main()
