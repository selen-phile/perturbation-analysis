"""
attention_head_analysis.py

Compares attention head behavior between clean (original_query) and noisy
prompts, per (model, dataset, noise_condition, layer, head).

For each head we compute, per prompt:
    - entropy   : average entropy of that head's attention distribution
                  (low entropy = focused/"peaky" head, high = diffuse)
    - last_tok_query_mass : how much attention the LAST token (next-token
                  prediction position) places on the QUERY span tokens
                  (a simple proxy for "is this head attending to the query").

We then report, per head:
    - entropy_delta   = entropy_noisy - entropy_clean
    - query_mass_delta = query_mass_noisy - query_mass_clean

A head with large |query_mass_delta| is a head whose attention to the
query shifts a lot under noise -- exactly the "heads active at clean vs
noisy query" comparison your teammate asked for.

NOTE: attention matrices are the biggest memory item here
      ([num_heads, seq, seq] per layer per prompt), so num_samples should
      be kept SMALL (10-30) unless you have a lot of VRAM/RAM. We compute
      running means on the fly and never store the full attentions.

Output:
    attention_results/attention_<model>_<dataset>.csv
    attention_results/grand_attention.csv

Usage:
    python attention_head_analysis.py --num_samples 20
"""

import os
import ast
import gc
import argparse
import torch
import pandas as pd
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

MAX_LEN = 512
OUTPUT_DIR = "attention_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_generic(document, query):
    doc = str(document)[:1000]
    # returns full prompt AND the (start_char, end_char) span of the query text
    prefix = f"Context: {doc}\n\nQuery: "
    query_start = len(prefix)
    prompt = f"{prefix}{query}\n\nAnswer:"
    query_end = query_start + len(str(query))
    return prompt, query_start, query_end


def format_qtsumm(table_raw, query):
    try:
        table_data = ast.literal_eval(table_raw)
    except (ValueError, SyntaxError):
        table_data = {"title": "", "headers": [], "data": []}
    title, headers, rows = table_data.get("title", ""), table_data.get("header", table_data.get("headers", [])), table_data.get("rows", table_data.get("data", []))
    md = ""
    if title:
        md += f"### {title}\n\n"
    if headers:
        md += "| " + " | ".join(str(h) for h in headers) + " |\n"
        md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        md += "| " + " | ".join(str(c) for c in r) + " |\n"
    prefix = f"Table Data:\n{md}\nQuery: "
    query_start = len(prefix)
    prompt = f"{prefix}{query}\n\nSummary:"
    query_end = query_start + len(str(query))
    return prompt, query_start, query_end


def build_prompt_with_span(schema, doc_or_table, query):
    if schema == "qtsumm":
        return format_qtsumm(doc_or_table, query)
    return format_generic(doc_or_table, query)


def get_doc_col(schema):
    return "table" if schema == "qtsumm" else "document"


def char_span_to_token_span(tokenizer, prompt, char_start, char_end):
    """Map a character span in `prompt` to a token index span using the
    tokenizer's offset mapping."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN,
                     return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    tok_start, tok_end = None, None
    for idx, (s, e) in enumerate(offsets):
        if s == e == 0:
            continue
        if tok_start is None and e > char_start:
            tok_start = idx
        if s < char_end:
            tok_end = idx
    if tok_start is None:
        tok_start = 0
    if tok_end is None or tok_end < tok_start:
        tok_end = tok_start
    enc.pop("offset_mapping")
    return enc, tok_start, tok_end + 1  # exclusive end


def analyze_attentions(attentions, query_tok_start, query_tok_end):
    
    per_layer = {}
    for layer_idx, attn in enumerate(attentions):
        attn = attn.squeeze(0).float()  # [num_heads, seq, seq]
        num_heads, seq, _ = attn.shape
        eps = 1e-12

        # entropy of each head's attention dist, averaged over query positions (rows)
        ent = -(attn * (attn + eps).log()).sum(dim=-1)  # [num_heads, seq]
        mean_entropy = ent.mean(dim=-1)  # [num_heads]

        # attention mass the LAST token places on the query span
        q_start = min(query_tok_start, seq - 1)
        q_end = min(query_tok_end, seq)
        if q_end <= q_start:
            q_end = q_start + 1
        last_row = attn[:, -1, :]  # [num_heads, seq]  attention from last token
        query_mass = last_row[:, q_start:q_end].sum(dim=-1)  # [num_heads]

        per_layer[layer_idx] = list(zip(mean_entropy.cpu().tolist(), query_mass.cpu().tolist()))
    return per_layer


def run_pair(model_key, dataset_key, num_samples):
    model_cfg, dataset_cfg = MODELS[model_key], DATASETS[dataset_key]
    schema = dataset_cfg["schema"]
    doc_col = get_doc_col(schema)

    print(f"\n=== Attention analysis: {model_key} / {dataset_key} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"], dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",  # required to get output_attentions
    )
    model.eval()
    device = next(model.parameters()).device

    df_test = pd.read_csv(dataset_cfg["test_csv"])
    df_robust = pd.read_csv(dataset_cfg["robust_csv"])
    n = min(len(df_test), len(df_robust), num_samples)
    df_test, df_robust = df_test.head(n).reset_index(drop=True), df_robust.head(n).reset_index(drop=True)
    print(f"Using {n} samples")

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # running sums: sums[cond][layer][head] = [ent_c, ent_n, qm_c, qm_n, count]
    sums = {
        cond: [[[0.0, 0.0, 0.0, 0.0, 0] for _ in range(num_heads)] for _ in range(num_layers)]
        for cond in NOISE_CONDITIONS
    }

    with torch.no_grad():
        for i in range(n):
            doc = str(df_test.loc[i, doc_col])
            clean_query = str(df_robust.loc[i, "original_query"])
            clean_prompt, cs, ce = build_prompt_with_span(schema, doc, clean_query)
            clean_inputs, c_tok_s, c_tok_e = char_span_to_token_span(tokenizer, clean_prompt, cs, ce)
            clean_inputs = {k: v.to(device) for k, v in clean_inputs.items()}
            clean_out = model(**clean_inputs, output_attentions=True)
            clean_attn_stats = analyze_attentions(clean_out.attentions, c_tok_s, c_tok_e)
            del clean_inputs, clean_out

            for cond in NOISE_CONDITIONS:
                noisy_query = str(df_robust.loc[i, cond])
                noisy_prompt, ns, ne = build_prompt_with_span(schema, doc, noisy_query)
                noisy_inputs, n_tok_s, n_tok_e = char_span_to_token_span(tokenizer, noisy_prompt, ns, ne)
                noisy_inputs = {k: v.to(device) for k, v in noisy_inputs.items()}
                noisy_out = model(**noisy_inputs, output_attentions=True)
                noisy_attn_stats = analyze_attentions(noisy_out.attentions, n_tok_s, n_tok_e)
                del noisy_inputs, noisy_out

                for layer_idx in range(num_layers):
                    for head_idx in range(num_heads):
                        ent_c, qm_c = clean_attn_stats[layer_idx][head_idx]
                        ent_n, qm_n = noisy_attn_stats[layer_idx][head_idx]
                        rec = sums[cond][layer_idx][head_idx]
                        rec[0] += ent_c
                        rec[1] += ent_n
                        rec[2] += qm_c
                        rec[3] += qm_n
                        rec[4] += 1

            if i % 10 == 0:
                torch.cuda.empty_cache()
                print(f"  [{i}/{n}]")

    records = []
    for cond in NOISE_CONDITIONS:
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                ent_c, ent_n, qm_c, qm_n, cnt = sums[cond][layer_idx][head_idx]
                if cnt == 0:
                    continue
                records.append({
                    "model": model_key,
                    "dataset": dataset_key,
                    "condition": cond,
                    "layer": layer_idx,
                    "head": head_idx,
                    "entropy_clean": ent_c / cnt,
                    "entropy_noisy": ent_n / cnt,
                    "entropy_delta": (ent_n - ent_c) / cnt,
                    "query_mass_clean": qm_c / cnt,
                    "query_mass_noisy": qm_n / cnt,
                    "query_mass_delta": (qm_n - qm_c) / cnt,
                })

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20, help="Rows per (model,dataset) pair (keep small - attention is memory heavy)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    args = parser.parse_args()

    all_records = []
    for model_key in args.models:
        for dataset_key in args.datasets:
            pair_path = os.path.join(OUTPUT_DIR, f"attention_{model_key}_{dataset_key}.csv")
            if os.path.exists(pair_path):
                print(f"Skipping {model_key}/{dataset_key} — already done.")
                all_records.extend(pd.read_csv(pair_path).to_dict("records"))
                continue
            records = run_pair(model_key, dataset_key, args.num_samples)
            all_records.extend(records)
            pd.DataFrame(records).to_csv(pair_path, index=False)
            pd.DataFrame(all_records).to_csv(os.path.join(OUTPUT_DIR, "grand_attention.csv"), index=False)
            print(f"Saved: {pair_path}")

    print(f"\nDone. Grand file: {os.path.join(OUTPUT_DIR, 'grand_attention.csv')}")


if __name__ == "__main__":
    main()
