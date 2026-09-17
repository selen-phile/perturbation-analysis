
import os
import ast
import gc
import argparse

import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from transcoder import Transcoder


# ============================================================
# CONFIG
# ============================================================

MODELS = {
    "llama": {
        "model_name": "meta-llama/Meta-Llama-3.1-8B",
        "layers": [15, 18, 20, 22, 25],
        "d_model": 4096,
        "transcoder_root": "models_llama",
    },

    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B",
        "layers": [13, 15, 18, 20, 22],
        "d_model": 3584,
        "transcoder_root": "models_qwen",
    },

    "gemma": {
        "model_name": "google/gemma-2-9b",
        "layers": [19, 23, 26, 29, 33],
        "d_model": 3584,
        "transcoder_root": "models_gemma",
    },
}


DATASETS = {
    "squad": {
        "test_csv": "dataset/squad_3_column_test.csv",
        "robust_csv": "dataset/squad_test_robustness_all.csv",
        "schema": "generic",
        "transcoder_subdir": "models_squad",
    },

    "ms_marco": {
        "test_csv": "dataset/ms_marco_3_column_test.csv",
        "robust_csv": "dataset/ms_marco_test_robustness_all.csv",
        "schema": "generic",
        "transcoder_subdir": "models_ms_marco",
    },

    "qtsumm": {
        "test_csv": "dataset/qtsumm_test.csv",
        "robust_csv": "dataset/qtsumm_test_robustness_all.csv",
        "schema": "qtsumm",
        "transcoder_subdir": "models_qtsumm",
    },
}


NOISE_CONDITIONS = [
    "noise_1_character",
    "noise_2_word_dropout",
    "noise_3_token_shuffle",
    "noise_4_semantic",
    "noise_5_adversarial",
]


# Same truncation used in the original feature analysis.
SEQ_TRUNCATE = 128

# Transcoder expansion factor.
EXPANSION = 32

OUTPUT_DIR = "feature_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PROMPT FORMATTERS
# ============================================================

def format_generic(document, query):
    """
    Construct prompt for SQuAD / MS MARCO style datasets.
    """
    doc = str(document)[:1000]

    return (
        f"Context: {doc}\n\n"
        f"Query: {query}\n\n"
        f"Answer:"
    )


def format_qtsumm(table_raw, query):
    """
    Construct prompt for QTSumm.
    """

    try:
        table_data = ast.literal_eval(table_raw)
    except (ValueError, SyntaxError):
        table_data = {
            "title": "",
            "headers": [],
            "data": []
        }

    title = table_data.get("title", "")

    headers = table_data.get(
        "header",
        table_data.get("headers", [])
    )

    rows = table_data.get(
        "rows",
        table_data.get("data", [])
    )

    md = ""

    if title:
        md += f"### {title}\n\n"

    if headers:
        md += "| " + " | ".join(
            str(h) for h in headers
        ) + " |\n"

        md += "|" + "|".join(
            ["---"] * len(headers)
        ) + "|\n"

    for r in rows:
        md += "| " + " | ".join(
            str(c) for c in r
        ) + " |\n"

    return (
        f"Table Data:\n"
        f"{md}\n"
        f"Query: {query}\n\n"
        f"Summary:"
    )


def build_prompt(schema, doc_or_table, query):

    if schema == "qtsumm":
        return format_qtsumm(
            doc_or_table,
            query
        )

    return format_generic(
        doc_or_table,
        query
    )


def get_doc_col(schema):

    if schema == "qtsumm":
        return "table"

    return "document"


# ============================================================
# CACHE HIDDEN STATES
# ============================================================

def cache_hidden_states(
    model,
    tokenizer,
    df_test,
    df_robust,
    schema,
    target_layers,
):
    """
    Run the LLM once for each clean query and once for each
    perturbation condition.

    Returns:

        clean_cache[layer][sample_id]
        noisy_cache[layer][condition][sample_id]

    Hidden states are stored on CPU as float16 to reduce memory.
    """

    doc_col = get_doc_col(schema)

    num_samples = min(
        len(df_test),
        len(df_robust)
    )

    print(
        f"Caching hidden states for "
        f"{num_samples} samples..."
    )

    clean_cache = {
        layer: {}
        for layer in target_layers
    }

    noisy_cache = {
        layer: {
            cond: {}
            for cond in NOISE_CONDITIONS
        }
        for layer in target_layers
    }

    device = next(model.parameters()).device

    with torch.no_grad():

        for i in range(num_samples):

            doc = str(
                df_test.loc[i, doc_col]
            )

            # ------------------------------------------------
            # CLEAN QUERY
            # ------------------------------------------------

            clean_query = str(
                df_robust.loc[i, "original_query"]
            )

            clean_prompt = build_prompt(
                schema,
                doc,
                clean_query
            )

            clean_inputs = tokenizer(
                clean_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            clean_out = model(
                **clean_inputs,
                output_hidden_states=True
            )

            for layer in target_layers:

                # hidden_states[0] = embedding output
                # hidden_states[layer + 1] = transformer block output
                h = clean_out.hidden_states[
                    layer + 1
                ]

                h = (
                    h.squeeze(0)
                    [:SEQ_TRUNCATE]
                    .to(torch.float16)
                    .cpu()
                )

                clean_cache[layer][i] = h

            del clean_inputs
            del clean_out

            # ------------------------------------------------
            # NOISY QUERIES
            # ------------------------------------------------

            for cond in NOISE_CONDITIONS:

                noisy_query = str(
                    df_robust.loc[i, cond]
                )

                noisy_prompt = build_prompt(
                    schema,
                    doc,
                    noisy_query
                )

                noisy_inputs = tokenizer(
                    noisy_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)

                noisy_out = model(
                    **noisy_inputs,
                    output_hidden_states=True
                )

                for layer in target_layers:

                    h = noisy_out.hidden_states[
                        layer + 1
                    ]

                    h = (
                        h.squeeze(0)
                        [:SEQ_TRUNCATE]
                        .to(torch.float16)
                        .cpu()
                    )

                    noisy_cache[
                        layer
                    ][cond][i] = h

                del noisy_inputs
                del noisy_out

            if i % 100 == 0:

                torch.cuda.empty_cache()

                print(
                    f"    [{i}/{num_samples}]"
                )

    return (
        clean_cache,
        noisy_cache,
        num_samples
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def analyze_feature_changes(
    tc,
    clean_hidden,
    noisy_hidden,
):
    """
    Compare sparse feature activations for ONE sample.

    Returns:

        abs_delta
            Absolute feature activation change.

        signed_delta
            Signed feature activation change.

        clean_top
            Dominant feature for clean input.

        noisy_top
            Dominant feature for noisy input.
    """

    # --------------------------------------------------------
    # Make sequence lengths comparable.
    # --------------------------------------------------------

    min_len = min(
        clean_hidden.shape[0],
        noisy_hidden.shape[0]
    )

    clean_hidden = clean_hidden[
        :min_len
    ]

    noisy_hidden = noisy_hidden[
        :min_len
    ]

    # --------------------------------------------------------
    # Move to GPU and convert to bfloat16.
    # --------------------------------------------------------

    clean_hidden = (
        clean_hidden
        .unsqueeze(0)
        .bfloat16()
        .cuda()
    )

    noisy_hidden = (
        noisy_hidden
        .unsqueeze(0)
        .bfloat16()
        .cuda()
    )

    with torch.no_grad():

        # [1, seq, d_feature]
        clean_feat = tc.encoder(
            clean_hidden
        )

        noisy_feat = tc.encoder(
            noisy_hidden
        )

        # ----------------------------------------------------
        # Average feature activations across tokens.
        # ----------------------------------------------------

        clean_avg = (
            clean_feat
            .mean(dim=1)
            .squeeze(0)
        )

        noisy_avg = (
            noisy_feat
            .mean(dim=1)
            .squeeze(0)
        )

        # ----------------------------------------------------
        # Direct clean-vs-noisy difference.
        # ----------------------------------------------------

        signed_delta = (
            noisy_avg - clean_avg
        )

        abs_delta = signed_delta.abs()

        # ----------------------------------------------------
        # Dominant feature identity.
        # ----------------------------------------------------

        clean_top = torch.argmax(
            clean_avg
        ).item()

        noisy_top = torch.argmax(
            noisy_avg
        ).item()

    # Move results to CPU before freeing GPU tensors.

    abs_delta = (
        abs_delta
        .detach()
        .cpu()
        .float()
    )

    signed_delta = (
        signed_delta
        .detach()
        .cpu()
        .float()
    )

    del clean_hidden
    del noisy_hidden
    del clean_feat
    del noisy_feat
    del clean_avg
    del noisy_avg

    return (
        abs_delta,
        signed_delta,
        clean_top,
        noisy_top
    )


# ============================================================
# ONE MODEL × DATASET
# ============================================================

def run_pair(
    model_key,
    dataset_key,
):

    model_cfg = MODELS[
        model_key
    ]

    dataset_cfg = DATASETS[
        dataset_key
    ]

    target_layers = model_cfg[
        "layers"
    ]

    schema = dataset_cfg[
        "schema"
    ]

    doc_col = get_doc_col(
        schema
    )

    print("\n" + "=" * 70)
    print(
        f"FEATURE ANALYSIS: "
        f"{model_key} / {dataset_key}"
    )
    print("=" * 70)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        f"Loading model: "
        f"{model_cfg['model_name']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name"]
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    # ========================================================
    # LOAD DATA
    # ========================================================

    df_test = pd.read_csv(
        dataset_cfg["test_csv"]
    )

    df_robust = pd.read_csv(
        dataset_cfg["robust_csv"]
    )

    num_samples = min(
        len(df_test),
        len(df_robust)
    )

    df_test = (
        df_test
        .head(num_samples)
        .reset_index(drop=True)
    )

    df_robust = (
        df_robust
        .head(num_samples)
        .reset_index(drop=True)
    )

    print(
        f"Samples: {num_samples}"
    )

    # ========================================================
    # PHASE A:
    # CACHE CLEAN + NOISY HIDDEN STATES
    # ========================================================

    (
        clean_cache,
        noisy_cache,
        num_samples,
    ) = cache_hidden_states(
        model=model,
        tokenizer=tokenizer,
        df_test=df_test,
        df_robust=df_robust,
        schema=schema,
        target_layers=target_layers,
    )

    # ========================================================
    # FREE LLM
    # ========================================================

    del model

    gc.collect()

    torch.cuda.empty_cache()

    print(
        "\nLLM unloaded. "
        "Starting transcoder analysis..."
    )

    # ========================================================
    # PHASE B:
    # TRANSCODER ANALYSIS
    # ========================================================

    all_records = []

    for layer in target_layers:

        print(
            f"\n{'-' * 60}"
        )

        print(
            f"Layer {layer}"
        )

        print(
            f"{'-' * 60}"
        )

        # ----------------------------------------------------
        # Load transcoder
        # ----------------------------------------------------

        tc_path = os.path.join(
            model_cfg[
                "transcoder_root"
            ],
            dataset_cfg[
                "transcoder_subdir"
            ],
            f"transcoder_layer_{layer}_float32.pt"
        )

        print(
            f"Loading: {tc_path}"
        )

        tc = Transcoder(
            model_cfg["d_model"],
            expansion=EXPANSION,
        )

        tc.load_state_dict(
            torch.load(
                tc_path,
                map_location="cpu",
                weights_only=True,
            )
        )

        tc = (
            tc
            .bfloat16()
            .cuda()
        )

        tc.eval()

        # ====================================================
        # EACH NOISE CONDITION
        # ====================================================

        for cond in NOISE_CONDITIONS:

            print(
                f"\nCondition: {cond}"
            )

            # ------------------------------------------------
            # Store one feature vector per sample.
            # ------------------------------------------------

            sample_abs_deltas = []

            sample_signed_deltas = []

            stability_count = 0

            # ------------------------------------------------
            # Per-sample analysis
            # ------------------------------------------------

            for i in range(num_samples):

                clean_h = (
                    clean_cache[
                        layer
                    ][i]
                )

                noisy_h = (
                    noisy_cache[
                        layer
                    ][cond][i]
                )

                (
                    abs_delta,
                    signed_delta,
                    clean_top,
                    noisy_top,
                ) = analyze_feature_changes(
                    tc=tc,
                    clean_hidden=clean_h,
                    noisy_hidden=noisy_h,
                )

                # Store feature-wise changes.
                sample_abs_deltas.append(
                    abs_delta
                )

                sample_signed_deltas.append(
                    signed_delta
                )

                # ------------------------------------------------
                # Feature stability
                # ------------------------------------------------

                if clean_top == noisy_top:
                    stability_count += 1

                if i % 100 == 0:

                    print(
                        f"    [{i}/{num_samples}]"
                    )

            # ====================================================
            # AGGREGATE ACROSS SAMPLES
            # ====================================================

            abs_delta_matrix = torch.stack(
                sample_abs_deltas
            )

            signed_delta_matrix = torch.stack(
                sample_signed_deltas
            )

            # ----------------------------------------------------
            # Mean absolute change per feature
            # ----------------------------------------------------

            mean_abs_delta = (
                abs_delta_matrix
                .mean(dim=0)
            )

            # ----------------------------------------------------
            # Mean signed change per feature
            # ----------------------------------------------------

            mean_signed_delta = (
                signed_delta_matrix
                .mean(dim=0)
            )

            # ----------------------------------------------------
            # Most affected feature
            # ----------------------------------------------------

            top1_idx = torch.argmax(
                mean_abs_delta
            ).item()

            top1_abs_delta = (
                mean_abs_delta[
                    top1_idx
                ].item()
            )

            top1_signed_delta = (
                mean_signed_delta[
                    top1_idx
                ].item()
            )

            # ----------------------------------------------------
            # Other useful aggregate statistics
            # ----------------------------------------------------

            mean_feature_abs_change = (
                mean_abs_delta
                .mean()
                .item()
            )

            max_feature_abs_change = (
                mean_abs_delta
                .max()
                .item()
            )

            # ----------------------------------------------------
            # Feature stability
            # ----------------------------------------------------

            stability = (
                100.0
                * stability_count
                / num_samples
            )

            # ====================================================
            # SAVE
            # ====================================================

            record = {

                "model":
                    model_key,

                "dataset":
                    dataset_key,

                "layer":
                    layer,

                "condition":
                    cond,

                # --------------------------------------------
                # Most affected feature
                # --------------------------------------------

                "top1_feature":
                    top1_idx,

                "top1_abs_delta":
                    round(
                        top1_abs_delta,
                        6
                    ),

                "top1_signed_delta":
                    round(
                        top1_signed_delta,
                        6
                    ),

                # --------------------------------------------
                # Overall feature statistics
                # --------------------------------------------

                "mean_feature_abs_change":
                    round(
                        mean_feature_abs_change,
                        6
                    ),

                "max_feature_abs_change":
                    round(
                        max_feature_abs_change,
                        6
                    ),

                # --------------------------------------------
                # Feature identity stability
                # --------------------------------------------

                "feature_stability":
                    round(
                        stability,
                        2
                    ),

                "num_samples":
                    num_samples,
            }

            all_records.append(
                record
            )

            print(
                f"    Top affected feature: "
                f"{top1_idx}"
            )

            print(
                f"    Mean |Δz|: "
                f"{top1_abs_delta:.6f}"
            )

            print(
                f"    Signed Δz: "
                f"{top1_signed_delta:.6f}"
            )

            print(
                f"    Feature stability: "
                f"{stability:.2f}%"
            )

            # Free condition-specific tensors.

            del sample_abs_deltas
            del sample_signed_deltas
            del abs_delta_matrix
            del signed_delta_matrix
            del mean_abs_delta
            del mean_signed_delta

            gc.collect()

            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Free transcoder
        # ----------------------------------------------------

        del tc

        gc.collect()

        torch.cuda.empty_cache()

    # ========================================================
    # FREE CACHES
    # ========================================================

    del clean_cache
    del noisy_cache

    gc.collect()

    torch.cuda.empty_cache()

    return all_records


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(
            MODELS.keys()
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(
            DATASETS.keys()
        ),
    )

    args = parser.parse_args()

    all_records = []

    # ========================================================
    # RUN ALL MODEL × DATASET PAIRS
    # ========================================================

    for model_key in args.models:

        for dataset_key in args.datasets:

            pair_path = os.path.join(
                OUTPUT_DIR,
                f"features_{model_key}_{dataset_key}.csv"
            )

            # ------------------------------------------------
            # Skip completed runs
            # ------------------------------------------------

            if os.path.exists(
                pair_path
            ):

                print(
                    f"\nSkipping "
                    f"{model_key} / "
                    f"{dataset_key} "
                    f"— already done."
                )

                existing = pd.read_csv(
                    pair_path
                ).to_dict(
                    "records"
                )

                all_records.extend(
                    existing
                )

                continue

            # ------------------------------------------------
            # Run
            # ------------------------------------------------

            records = run_pair(
                model_key,
                dataset_key,
            )

            all_records.extend(
                records
            )

            # ------------------------------------------------
            # Save pair result
            # ------------------------------------------------

            pd.DataFrame(
                records
            ).to_csv(
                pair_path,
                index=False
            )

            print(
                f"\nSaved: {pair_path}"
            )

            # ------------------------------------------------
            # Crash-safe grand file
            # ------------------------------------------------

            pd.DataFrame(
                all_records
            ).to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    "grand_features.csv"
                ),
                index=False
            )

            print(
                "Grand summary updated."
            )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    grand_path = os.path.join(
        OUTPUT_DIR,
        "grand_features.csv"
    )

    pd.DataFrame(
        all_records
    ).to_csv(
        grand_path,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL FEATURE ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nGrand summary: "
        f"{grand_path}"
    )


if __name__ == "__main__":
    main()