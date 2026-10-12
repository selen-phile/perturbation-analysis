"""
activation_patching.py

Causal test of the question the observational scripts cannot answer:
do the "perturbation-sensitive" attention heads / layers actually MEDIATE the
change in the model's output caused by a perturbed query?

Setup
-----
For every (sample, perturbation) we have a clean prompt C and a perturbed
prompt N (same document, only the query differs). Output divergence is
measured at the next-token distribution of the final prompt token:
    KL_noisy = KL( p_clean || p_noisy ).

Patching positions. C and N have different lengths, so activations are only
patched at positions that correspond: the template suffix after the query
(e.g. "\\n\\nAnswer:", identical token ids, aligned from the end) which includes
the final prediction token. These positions read the query through attention.
(--patch_positions last restricts this to the final token only.)

Two directions
  denoise (sufficiency): run N, replace the chosen component's activation at
          those positions with the one from C.
          recovery = 1 - KL(p_clean || p_patched) / KL_noisy
          (1 = fully restores clean behaviour, 0 = nothing, <0 = worse).
  noise   (necessity):   run C, replace the component's activation with the
          one from N.
          induced = KL(p_clean || p_patched) / KL_noisy
          (1 = reproduces the full divergence, 0 = no effect).
A log-prob version using the clean top-1 token t* is also stored:
    (lp_patched - lp_noisy) / (lp_clean - lp_noisy)   (denoise).

Components patched (per head = the head's output before W_O, o_proj input):
  topk                the top-K heads from the attention analysis ranking
  bottomk             the K LEAST sensitive heads
  random              K random heads (n_random draws)       -- control
  layer_matched_random  K random heads in the SAME layers as topk -- control
                      (separates "these heads" from "these layers")
  single_head         each top-K head alone                 (--single_heads)
  resid_layer         whole residual stream at layer l      (--layer_stride)
                      = localization curve; the last layer must give ~100%
                      recovery and is used as a built-in positive control.

Selection vs evaluation. Heads are ranked on samples chosen with
--selection_seed/--selection_n (matching attention_head_analysis.py). The
patching samples are drawn from the REMAINING rows so the heads are not
evaluated on the data used to select them.

Samples with KL_noisy < --min_kl are not patched (no effect to recover).

Output (OUTPUT_DIR):
    patching_<model>_<dataset>.csv   one row per sample x cond x direction x group
    patching_groups_<model>.json     the exact head groups used
    patching_summary.csv, patching_topk_vs_controls.csv,
    patching_layer_sweep.csv, patching_single_heads.csv

Usage:
    python activation_patching.py --num_samples 50 --k 20
    python activation_patching.py --summarize_only
"""

import os
import ast
import gc
import json
import argparse
from contextlib import contextmanager

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
    "squad": {"test_csv": "dataset/squad_3_column_test.csv",
              "robust_csv": "dataset/squad_test_robustness_all.csv", "schema": "generic"},
    "ms_marco": {"test_csv": "dataset/ms_marco_3_column_test.csv",
                 "robust_csv": "dataset/ms_marco_test_robustness_all.csv", "schema": "generic"},
    "qtsumm": {"test_csv": "dataset/qtsumm_test.csv",
               "robust_csv": "dataset/qtsumm_test_robustness_all.csv", "schema": "qtsumm"},
}

NOISE_CONDITIONS = [
    "noise_1_character", "noise_2_word_dropout", "noise_3_token_shuffle",
    "noise_4_semantic", "noise_5_adversarial",
]

TEMPLATES = {
    "generic": ("Context: ", "\n\nQuery: ", "\n\nAnswer:"),
    "qtsumm":  ("Table Data:\n", "\nQuery: ", "\n\nSummary:"),
}

MAX_LEN = 512
QUERY_BUDGET = 128
TEMPLATE_OVERHEAD = 32
CTX_BUDGET = MAX_LEN - QUERY_BUDGET - TEMPLATE_OVERHEAD

OUTPUT_DIR = "patching_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================================
# Prompts (identical to the other scripts)
# ==========================================================================
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
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    idxs = [i for i, (s, e) in enumerate(offsets)
            if e > s and s < char_end and e > char_start]
    if not idxs:
        return enc, 0, 0
    return enc, idxs[0], idxs[-1] + 1


def common_suffix_len(a, b, max_len):
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y or n >= max_len:
            break
        n += 1
    return n


# ==========================================================================
# Head groups
# ==========================================================================
def build_groups(rank_df, n_layers, n_heads, k, n_random, seed):
    """Returns list of (group_type, group_id, [(layer, head), ...])."""
    ranked = list(zip(rank_df["layer"].astype(int), rank_df["head"].astype(int)))
    top, bottom = ranked[:k], ranked[-k:]
    top_set = set(top)
    rest = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in top_set]
    rng = np.random.RandomState(seed)

    groups = [("topk", "0", top), ("bottomk", "0", bottom)]
    for r in range(n_random):
        sel = rng.choice(len(rest), size=min(k, len(rest)), replace=False)
        groups.append(("random", str(r), [rest[i] for i in sel]))

    by_layer = {}
    for l, h in top:
        by_layer.setdefault(l, []).append(h)
    for r in range(n_random):
        lm = []
        for l, hs in by_layer.items():
            cands = [h for h in range(n_heads) if (l, h) not in top_set]
            take = min(len(hs), len(cands))
            lm += [(l, int(h)) for h in rng.choice(cands, size=take, replace=False)]
        groups.append(("layer_matched_random", str(r), lm))
    return groups


def heads_to_spec(heads):
    spec = {}
    for l, h in heads:
        spec.setdefault(int(l), []).append(int(h))
    return spec


# ==========================================================================
# Hooks: capture and patch (Llama / Qwen2 / Gemma2 share this module layout)
# ==========================================================================
@contextmanager
def _hooks(handles):
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _layer_out(out):
    return out[0] if isinstance(out, tuple) else out


def _like_layer_out(out, hs):
    return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


@torch.no_grad()
def capture(model, enc, n_pos):
    """Forward pass; returns last-token log-probs and the activations at the
    last n_pos positions: head outputs (o_proj input) and residual per layer."""
    heads, resid, handles = {}, {}, []
    for l, layer in enumerate(model.model.layers):
        def pre(mod, args, l=l):
            heads[l] = args[0][0, -n_pos:].detach().to("cpu")
        def post(mod, inp, out, l=l):
            resid[l] = _layer_out(out)[0, -n_pos:].detach().to("cpu")
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(pre))
        handles.append(layer.register_forward_hook(post))
    with _hooks(handles):
        out = model(**enc, use_cache=False)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return logp, {"heads": heads, "resid": resid}


@torch.no_grad()
def patched_logp(model, enc, source, n_pos, head_dim, head_spec=None, resid_layer=None):
    """Forward pass with the chosen component's activation at the last n_pos
    positions replaced by `source` (from the other prompt)."""
    handles = []
    for l, heads in (head_spec or {}).items():
        def pre(mod, args, l=l, heads=tuple(heads)):
            x = args[0].clone()
            src = source["heads"][l].to(x.device, x.dtype)
            for h in heads:
                sl = slice(h * head_dim, (h + 1) * head_dim)
                x[:, -n_pos:, sl] = src[:, sl]
            return (x,) + tuple(args[1:])
        handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
    if resid_layer is not None:
        def post(mod, inp, out, l=resid_layer):
            hs = _layer_out(out).clone()
            hs[:, -n_pos:, :] = source["resid"][l].to(hs.device, hs.dtype)
            return _like_layer_out(out, hs)
        handles.append(model.model.layers[resid_layer].register_forward_hook(post))
    with _hooks(handles):
        out = model(**enc, use_cache=False)
    return torch.log_softmax(out.logits[0, -1].float(), dim=-1)


def kl(logp_p, logp_q):
    return float((logp_p.exp() * (logp_p - logp_q)).sum().item())


# ==========================================================================
# Per-(model, dataset) run
# ==========================================================================
def sanity_check(model, enc_c, enc_n, src_c, src_n, lp_c, lp_n, n_pos, head_dim, n_layers, n_heads):
    """(a) patching a run with its OWN activations must be a no-op.
       (b) patching the final layer's residual must recover the clean output."""
    all_heads = {l: list(range(n_heads)) for l in range(n_layers)}
    lp_id = patched_logp(model, enc_n, src_n, n_pos, head_dim, head_spec=all_heads)
    diff = float((lp_id - lp_n).abs().max().item())
    lp_last = patched_logp(model, enc_n, src_c, n_pos, head_dim, resid_layer=n_layers - 1)
    kl_n, kl_last = kl(lp_c, lp_n), kl(lp_c, lp_last)
    rec = 1 - kl_last / max(kl_n, 1e-9)
    print(f"  [sanity] identity-patch max|dlogp|={diff:.4f} (want ~0); "
          f"final-layer resid recovery={rec:.4f} (want ~1)")
    if diff > 0.1 or rec < 0.95:
        print("  [sanity] WARNING: hooks are not behaving as expected for this model/transformers "
              "version. Check the module layout and layer output type before trusting results.")


def run_pair(model_key, dataset_key, args):
    dataset_cfg = DATASETS[dataset_key]
    schema = dataset_cfg["schema"]
    doc_col = "table" if schema == "qtsumm" else "document"

    print(f"\n=== Patching: {model_key} / {dataset_key} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key]["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_key]["model_name"], dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads

    rank_path = os.path.join(args.ranking_dir, f"head_ranking_{model_key}_{args.ranking_metric}.csv")
    rank_df = pd.read_csv(rank_path)          # sorted by sensitivity, descending
    groups = build_groups(rank_df, n_layers, n_heads, args.k, args.n_random, args.group_seed)
    with open(os.path.join(OUTPUT_DIR, f"patching_groups_{model_key}.json"), "w") as f:
        json.dump([{"type": t, "id": i, "heads": hs} for t, i, hs in groups], f)
    top_heads = next(hs for t, i, hs in groups if t == "topk")

    df_test = pd.read_csv(dataset_cfg["test_csv"])
    df_robust = pd.read_csv(dataset_cfg["robust_csv"])
    total = min(len(df_test), len(df_robust))
    df_test, df_robust = df_test.head(total), df_robust.head(total)

    # Held-out evaluation: exclude rows used to rank the heads.
    sel = set(np.random.RandomState(args.selection_seed)
              .choice(total, size=min(args.selection_n, total), replace=False).tolist())
    pool = np.array([i for i in range(total) if i not in sel])
    n = min(args.num_samples, len(pool))
    idx = np.sort(np.random.RandomState(args.eval_seed).choice(pool, size=n, replace=False))
    print(f"Evaluating {n} held-out rows (excluded {len(sel)} used for head selection)")

    layer_list = []
    if args.layer_stride > 0:
        layer_list = sorted(set(range(0, n_layers, args.layer_stride)) | {n_layers - 1})

    directions = ["denoise", "noise"] if args.direction == "both" else [args.direction]
    rows, did_sanity, skipped = [], False, 0

    for k_i, i in enumerate(idx):
        ctx = truncate_text(tokenizer, render_context(schema, df_test.loc[i, doc_col]), CTX_BUDGET)
        p_c, cs, ce = build_prompt(schema, ctx, str(df_robust.loc[i, "original_query"]))
        enc_c, _, qe_c = encode_with_span(tokenizer, p_c, cs, ce)
        ids_c = enc_c["input_ids"][0].tolist()
        if len(ids_c) > MAX_LEN:
            skipped += 1
            continue

        for cond in NOISE_CONDITIONS:
            p_n, ns, ne = build_prompt(schema, ctx, str(df_robust.loc[i, cond]))
            enc_n, _, qe_n = encode_with_span(tokenizer, p_n, ns, ne)
            ids_n = enc_n["input_ids"][0].tolist()
            if len(ids_n) > MAX_LEN or ids_c == ids_n or ids_c[-1] != ids_n[-1]:
                skipped += 1
                continue

            tpl = max(min(len(ids_c) - qe_c, len(ids_n) - qe_n), 1)
            n_pos = 1 if args.patch_positions == "last" else max(common_suffix_len(ids_c, ids_n, tpl), 1)

            ec = {k: v.to(device) for k, v in enc_c.items()}
            en = {k: v.to(device) for k, v in enc_n.items()}
            lp_c, src_c = capture(model, ec, n_pos)
            lp_n, src_n = capture(model, en, n_pos)
            kl_noisy = kl(lp_c, lp_n)
            t_star = int(lp_c.argmax().item())
            base = {"model": model_key, "dataset": dataset_key, "condition": cond,
                    "sample_idx": int(i), "n_pos": n_pos, "kl_noisy": kl_noisy,
                    "lp_clean_t": float(lp_c[t_star]), "lp_noisy_t": float(lp_n[t_star]),
                    "top1_flipped": int(int(lp_n.argmax().item()) != t_star)}

            if not did_sanity:
                sanity_check(model, ec, en, src_c, src_n, lp_c, lp_n, n_pos, head_dim, n_layers, n_heads)
                did_sanity = True

            rows.append({**base, "direction": "none", "group_type": "baseline", "group_id": "-",
                         "layer": -1, "kl_patched": kl_noisy, "lp_patched_t": float(lp_n[t_star])})
            if kl_noisy < args.min_kl:
                continue

            specs = [(t, i_, {"head_spec": heads_to_spec(hs)}, -1) for t, i_, hs in groups]
            if args.single_heads:
                specs += [("single_head", f"L{l}H{h}", {"head_spec": {l: [h]}}, l) for l, h in top_heads]
            specs += [("resid_layer", f"L{l}", {"resid_layer": l}, l) for l in layer_list]

            for direction in directions:
                target, source = (en, src_c) if direction == "denoise" else (ec, src_n)
                for gtype, gid, kwargs, layer in specs:
                    lp_p = patched_logp(model, target, source, n_pos, head_dim, **kwargs)
                    rows.append({**base, "direction": direction, "group_type": gtype,
                                 "group_id": gid, "layer": layer,
                                 "kl_patched": kl(lp_c, lp_p), "lp_patched_t": float(lp_p[t_star])})
            del src_c, src_n
        if k_i % 5 == 0:
            torch.cuda.empty_cache()
            print(f"  [{k_i}/{n}]  rows={len(rows)}")

    print(f"Skipped {skipped} (sample, condition) pairs (too long / identical / mismatched last token)")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ==========================================================================
# Summaries (pure pandas/numpy)
# ==========================================================================
def add_effect_columns(df, min_kl, min_lp_gap=0.05):
    """rec_kl : denoise -> recovery; noise -> fraction of divergence induced.
       rec_lp : same, on the clean top-1 token's log-prob."""
    df = df[(df["group_type"] != "baseline") & (df["kl_noisy"] >= min_kl)].copy()
    den = df["direction"] == "denoise"
    df["rec_kl"] = np.where(den, 1 - df["kl_patched"] / df["kl_noisy"], df["kl_patched"] / df["kl_noisy"])
    gap = df["lp_clean_t"] - df["lp_noisy_t"]
    rec_den = (df["lp_patched_t"] - df["lp_noisy_t"]) / gap
    rec_noi = (df["lp_clean_t"] - df["lp_patched_t"]) / gap
    df["rec_lp"] = np.where(gap > min_lp_gap, np.where(den, rec_den, rec_noi), np.nan)
    return df


def bootstrap_ci(x, n_boot=1000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(min_kl=0.05):
    files = sorted(f for f in os.listdir(OUTPUT_DIR)
                   if f.startswith("patching_") and f.endswith(".csv") and not f.startswith("patching_summary")
                   and f not in ("patching_topk_vs_controls.csv", "patching_layer_sweep.csv",
                                 "patching_single_heads.csv"))
    if not files:
        print("Nothing to summarize.")
        return
    raw = pd.concat([pd.read_csv(os.path.join(OUTPUT_DIR, f)) for f in files])
    df = add_effect_columns(raw, min_kl)
    keys = ["model", "condition", "direction"]

    # 1) group-level summary
    rows = []
    for (m, c, d, g), s in df.groupby(keys + ["group_type"]):
        for metric in ("rec_kl", "rec_lp"):
            lo, hi = bootstrap_ci(s[metric])
            rows.append({"model": m, "condition": c, "direction": d, "group_type": g, "metric": metric,
                         "n": int(s[metric].notna().sum()), "mean": s[metric].mean(),
                         "median": s[metric].median(), "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "patching_summary.csv"), index=False)

    # 2) paired top-K vs controls, per sample (control draws averaged within a sample)
    per = (df[df["group_type"].isin(["topk", "bottomk", "random", "layer_matched_random"])]
           .groupby(keys + ["dataset", "sample_idx", "group_type"])[["rec_kl", "rec_lp"]].mean().reset_index())
    out = []
    for metric in ("rec_kl", "rec_lp"):
        wide = per.pivot_table(index=keys + ["dataset", "sample_idx"], columns="group_type", values=metric)
        for ctrl in ("random", "layer_matched_random", "bottomk"):
            if "topk" not in wide or ctrl not in wide:
                continue
            w = wide[["topk", ctrl]].dropna()
            w = w.assign(diff=w["topk"] - w[ctrl]).reset_index()
            for (m, c, d), s in w.groupby(keys):
                lo, hi = bootstrap_ci(s["diff"])
                out.append({"model": m, "condition": c, "direction": d, "metric": metric,
                            "control": ctrl, "n": len(s), "topk_mean": s["topk"].mean(),
                            "control_mean": s[ctrl].mean(), "diff_mean": s["diff"].mean(),
                            "diff_ci_lo": lo, "diff_ci_hi": hi,
                            "frac_topk_better": float((s["diff"] > 0).mean())})
    pd.DataFrame(out).to_csv(os.path.join(OUTPUT_DIR, "patching_topk_vs_controls.csv"), index=False)

    # 3) layer sweep and single heads
    ls = df[df["group_type"] == "resid_layer"].groupby(keys + ["layer"])[["rec_kl", "rec_lp"]].mean().reset_index()
    ls.to_csv(os.path.join(OUTPUT_DIR, "patching_layer_sweep.csv"), index=False)
    sh = df[df["group_type"] == "single_head"].groupby(["model", "direction", "group_id"])[["rec_kl", "rec_lp"]] \
        .mean().reset_index().sort_values("rec_kl", ascending=False)
    sh.to_csv(os.path.join(OUTPUT_DIR, "patching_single_heads.csv"), index=False)
    print("Summaries written to", OUTPUT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--num_samples", type=int, default=50, help="held-out rows per (model, dataset)")
    ap.add_argument("--k", type=int, default=20, help="heads per group")
    ap.add_argument("--n_random", type=int, default=10, help="random control draws")
    ap.add_argument("--min_kl", type=float, default=0.05, help="skip samples with smaller output divergence")
    ap.add_argument("--direction", choices=["both", "denoise", "noise"], default="both")
    ap.add_argument("--patch_positions", choices=["suffix", "last"], default="suffix")
    ap.add_argument("--single_heads", action="store_true", help="also patch each top-K head alone")
    ap.add_argument("--layer_stride", type=int, default=2, help="residual-stream layer sweep; 0 disables")
    ap.add_argument("--ranking_dir", default="attention_results")
    ap.add_argument("--ranking_metric", default="query_mass")
    ap.add_argument("--selection_seed", type=int, default=0, help="seed used by attention_head_analysis.py")
    ap.add_argument("--selection_n", type=int, default=100, help="--num_samples used by attention_head_analysis.py")
    ap.add_argument("--eval_seed", type=int, default=1)
    ap.add_argument("--group_seed", type=int, default=0)
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args()

    if not args.summarize_only:
        for model_key in args.models:
            for dataset_key in args.datasets:
                path = os.path.join(OUTPUT_DIR, f"patching_{model_key}_{dataset_key}.csv")
                if os.path.exists(path):
                    print(f"Skipping {model_key}/{dataset_key} - already done.")
                    continue
                run_pair(model_key, dataset_key, args).to_csv(path, index=False)
                print(f"Saved: {path}")
    summarize(args.min_kl)


if __name__ == "__main__":
    main()
