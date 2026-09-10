# Latent Attribution Regularization (LAR)

Code used to produce all results reported in:

> Hassan, J. (2026). *Latent Attribution Regularization: Cross-Domain Attribution-Consistency Training for Stable NLP Explanations.* Preprint: [Research Square](https://www.researchsquare.com/article/rs-10984984/v1). Currently under submission to a peer-reviewed journal.

## What this paper does

Gradient-based feature attributions in fine-tuned Transformers can look highly unstable under small, meaning-preserving input perturbations (e.g. swapping a word for a synonym) — but a large part of that apparent instability turns out to be a measurement artifact caused by comparing attributions at the wrong token positions after tokenization shifts. This code:

1. Implements an **exact token-identity alignment** procedure that fixes the measurement artifact.
2. Implements **Latent Attribution Regularization (LAR)** — a training-time loss that adds a cosine-similarity consistency term between the attributions of an original input and a synonym-perturbed version of it, on top of standard cross-entropy fine-tuning.
3. Fine-tunes DistilBERT on SST-2 under both a plain cross-entropy control and LAR, across multiple random seeds and perturbation strengths, and evaluates attribution stability with the corrected alignment procedure.

## File

- **`code.py`** — the full training + evaluation pipeline (`LAR Kaggle Full-Scale Pipeline, v8`). Written to run as a Kaggle Notebook, with self-persisting checkpoints across Kaggle sessions (see "Running on Kaggle" below). The core methodology (perturbation, alignment, LAR loss, training loop) is plain PyTorch/HuggingFace and can be adapted to run outside Kaggle by removing the Kaggle-specific persistence layer (Section 1–2 of the script).

## Method summary

- **Base model:** `distilbert-base-uncased` (not a pre-fine-tuned SST-2 checkpoint — training starts from a randomly initialized classification head).
- **Task/data:** SST-2 (GLUE benchmark), via `nyu-mll/glue`.
- **Perturbation:** WordNet synonym substitution on adjectives/adverbs, at a configurable strength (fraction of eligible words swapped).
- **Attribution:** input-gradient norm — `‖∇_e logit_y(f(e))‖₂` per token embedding.
- **LAR loss:** `CE + λ · (1 − cosine_similarity(attr_orig, attr_perturbed))`, computed via a double-backward pass (gradient of a gradient), which requires disabling Flash/memory-efficient attention kernels (the "math" SDPA backend is used instead).
- **Evaluation metric:** mean Spearman correlation between original and perturbed attribution vectors, computed only over **exactly token-identity-matched** positions (the alignment fix).
- **Conditions compared:** `ce_only_control` (λ = 0) vs. `lar_lambda_0.5` (λ = 0.5).

## Reproducing the paper's results

The paper reports two training-scale regimes, each run as **10 independent seeds**:

| Regime | `TRAIN_SIZE` | `EPOCHS` | Seeds | Notes |
|---|---|---|---|---|
| Small scale | 1,500 | 1 | 10 seeds | Initial λ sweep + confirmation run |
| Large scale | 20,000 | 2 | 10 seeds | Confirms the effect at a more realistic fine-tuning budget |

The config block near the top of `code.py` (`TRAIN_SIZE`, `EPOCHS`, `SEEDS`) is set for one slice of the **large-scale** run (`TRAIN_SIZE=20000`, `EPOCHS=2`, `SEEDS=[5,6,7,8,9]` — extending a prior 5-seed run to the full 10). To reproduce a different regime, edit these three values and rerun; already-completed `(condition, seed)` pairs found in an existing `lar_v8_results.json` are automatically skipped, so the script can be safely re-run incrementally across sessions.

Set `SMOKE_TEST = True` once to sanity-check the full path (data loading, training step, checkpointing, auto-commit) on a tiny slice before committing to a full run.

## Running on Kaggle

This script is designed to survive Kaggle's session time limits by checkpointing to a Kaggle Dataset it manages via the Kaggle API, and restoring from that dataset on the next session. One-time setup:

1. Notebook Settings → **Internet: ON**.
2. Add-ons → **Secrets**: add `KAGGLE_USERNAME` and `KAGGLE_KEY` (from a token generated at `kaggle.com/settings` → API → Create New Token).
3. Set `KAGGLE_DATASET_SLUG` in the script to a dataset slug you own (it's created automatically on first commit).
4. After the first run (even partial), attach the dataset it created as an **input** to the notebook (Add Input → search for it). From then on, every session restores from it and commits back to it automatically.
5. If a session stops, just reopen and "Run All" — it resumes from the last checkpoint.

Auto-commit is best-effort: if Secrets aren't configured or the API call fails, the script logs a warning and continues with local saves only, so it degrades gracefully outside this setup.

## Running outside Kaggle

The Kaggle persistence layer (`bootstrap_from_input_dataset`, `kaggle_auto_commit`, and the `KAGGLE_*` config block) is self-contained and best-effort — it will no-op safely if `kaggle_secrets` isn't importable. The core loop (`main()`, `train_condition()`, `evaluate_aligned_stability()`) will run in any environment with the dependencies below and a CUDA GPU (or CPU, with much longer runtime); only local checkpoint/result saving to `/kaggle/working` paths will apply (adjust `OUTPUT_DIR` if you want a different location).

## Requirements

```
torch
transformers
datasets
nltk
scipy
pandas
numpy
```

Plus, on first run, NLTK data (`wordnet`, `omw-1.4`, `averaged_perceptron_tagger`, `punkt`) — the script downloads these automatically.

## Output

Running `main()` produces:
- `lar_v8_results.json` / `lar_v8_results.csv` — one row per `(condition, seed, perturbation_strength)`, with columns: `condition`, `lambda`, `seed`, `train_size`, `epochs`, `smoke_test`, `perturbation_strength`, `perturbation_fraction`, `mean_rho`, `n_valid_pairs`, `n_attempted`, `val_accuracy`.
- A printed summary table (mean/std of `mean_rho` and `val_accuracy`, grouped by condition/λ/perturbation strength) once all configured seeds finish.

These are the same result files included as supplementary material with the paper submission.

## Citation

```bibtex
@misc{hassan2026lar,
  title  = {Latent Attribution Regularization: Cross-Domain Attribution-Consistency Training for Stable NLP Explanations},
  author = {Hassan, Junaid},
  year   = {2026},
  note   = {Preprint},
  url    = {https://www.researchsquare.com/article/rs-10984984/v1}
}
```

(Will be updated with the peer-reviewed venue's citation upon acceptance.)
