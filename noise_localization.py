
import os
import ast
import gc
import argparse
import torch
import torch.nn.functional as F
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

SEQ_TRUNCATE = 128
OUTPUT_DIR = "localization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_generic(document, query):
    doc = str(document)[:1000]
    return f"Context: {doc}\n\nQuery: {query}\n\nAnswer:"


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
    return f"Table Data:\n{md}\nQuery: {query}\n\nSummary:"


def build_prompt(schema, doc_or_table, query):
    return format_qtsumm(doc_or_table, query) if schema == "qtsumm" else format_generic(doc_or_table, query)


def get_doc_col(schema):
    return "table" if schema == "qtsumm" else "document"


def layerwise_drift(hidden_states_clean, hidden_states_noisy):
    
    results = []
    for h_c, h_n in zip(hidden_states_clean, hidden_states_noisy):
        h_c = h_c.squeeze(0)[:SEQ_TRUNCATE].float()
        h_n = h_n.squeeze(0)[:SEQ_TRUNCATE].float()
        min_len = min(h_c.shape[0], h_n.shape[0])
        h_c, h_n = h_c[:min_len], h_n[:min_len]

        diff_norm = (h_c - h_n).norm(dim=-1)          # [seq]
        clean_norm = h_c.norm(dim=-1).clamp_min(1e-8)  # [seq]
        rel_l2 = (diff_norm / clean_norm).mean().item()

        cos_sim = F.cosine_similarity(h_c, h_n, dim=-1)  # [seq]
        cos_dist = (1 - cos_sim).mean().item()

        results.append((rel_l2, cos_dist))
    return results


def run_pair(model_key, dataset_key, num_samples):
    model_cfg, dataset_cfg = MODELS[model_key], DATASETS[dataset_key]
    schema = dataset_cfg["schema"]
    doc_col = get_doc_col(schema)

    print(f"\n=== Localization: {model_key} / {dataset_key} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"], dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    df_test = pd.read_csv(dataset_cfg["test_csv"])
    df_robust = pd.read_csv(dataset_cfg["robust_csv"])
    n = min(len(df_test), len(df_robust), num_samples)
    df_test, df_robust = df_test.head(n).reset_index(drop=True), df_robust.head(n).reset_index(drop=True)
    print(f"Using {n} samples")

    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    # accumulate sums per (condition, layer) to compute mean at the end
    sums = {cond: {"rel_l2": [0.0] * num_layers, "cos_dist": [0.0] * num_layers, "count": 0} for cond in NOISE_CONDITIONS}

    with torch.no_grad():
        for i in range(n):
            doc = str(df_test.loc[i, doc_col])
            clean_query = str(df_robust.loc[i, "original_query"])
            clean_prompt = build_prompt(schema, doc, clean_query)
            clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            clean_out = model(**clean_inputs, output_hidden_states=True)
            clean_hidden = clean_out.hidden_states
            del clean_inputs, clean_out

            for cond in NOISE_CONDITIONS:
                noisy_query = str(df_robust.loc[i, cond])
                noisy_prompt = build_prompt(schema, doc, noisy_query)
                noisy_inputs = tokenizer(noisy_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                noisy_out = model(**noisy_inputs, output_hidden_states=True)

                per_layer = layerwise_drift(clean_hidden, noisy_out.hidden_states)
                for layer_idx, (rel_l2, cos_dist) in enumerate(per_layer):
                    sums[cond]["rel_l2"][layer_idx] += rel_l2
                    sums[cond]["cos_dist"][layer_idx] += cos_dist
                sums[cond]["count"] += 1

                del noisy_inputs, noisy_out

            del clean_hidden
            if i % 20 == 0:
                torch.cuda.empty_cache()
                print(f"  [{i}/{n}]")

    records = []
    for cond in NOISE_CONDITIONS:
        cnt = sums[cond]["count"]
        for layer_idx in range(num_layers):
            records.append({
                "model": model_key,
                "dataset": dataset_key,
                "condition": cond,
                "layer": layer_idx,  # 0 = embedding output, 1..N = after each block
                "rel_l2": sums[cond]["rel_l2"][layer_idx] / cnt,
                "cos_dist": sums[cond]["cos_dist"][layer_idx] / cnt,
            })

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50, help="Rows per (model,dataset) pair")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    args = parser.parse_args()

    all_records = []
    for model_key in args.models:
        for dataset_key in args.datasets:
            pair_path = os.path.join(OUTPUT_DIR, f"localization_{model_key}_{dataset_key}.csv")
            if os.path.exists(pair_path):
                print(f"Skipping {model_key}/{dataset_key} — already done.")
                all_records.extend(pd.read_csv(pair_path).to_dict("records"))
                continue
            records = run_pair(model_key, dataset_key, args.num_samples)
            all_records.extend(records)
            pd.DataFrame(records).to_csv(pair_path, index=False)
            pd.DataFrame(all_records).to_csv(os.path.join(OUTPUT_DIR, "grand_localization.csv"), index=False)
            print(f"Saved: {pair_path}")

    print(f"\nDone. Grand file: {os.path.join(OUTPUT_DIR, 'grand_localization.csv')}")


if __name__ == "__main__":
    main()
