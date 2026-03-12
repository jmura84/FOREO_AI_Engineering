# 📓 Notebooks — LLM Translation Evaluation

This folder contains Jupyter notebooks used for **quantitative evaluation** of LLM-based translation quality (English ↔ Spanish). Each notebook progressively builds on the previous one, exploring different evaluation dimensions.

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 0 | `0 - llms_evaluation_prototype.ipynb` | **Prototype evaluation** — Initial benchmark comparing multiple LLMs on a small corpus using COMET, BLEU, TER and chrF scores. Serves as the baseline experiment. |
| 1 | `1 - llms_evaluation_biggercorpus.ipynb` | **Larger corpus evaluation** — Extends the prototype to a bigger dataset for more statistically robust results. Evaluates with COMET, BLEU, TER and chrF. |
| 2 | `2 - llms_evaluation_difftemps.ipynb` | **Temperature comparison** — Compares translation quality across different temperature settings using COMET, BLEU and TER to find optimal generation parameters. |
| 3 | `3 - llms_evaluation_rag_test.ipynb` | **RAG vs Standard** — Evaluates the performance improvement of RAG-augmented translation versus standard LLM translation using COMET, BLEU and TER scores. |
| 4 | `4 - llms_quality_estimation_rag_test.ipynb` | **Quality Estimation (QE)** — Uses reference-free QE models (e.g. CometKiwi) to score translations without needing ground-truth references. Compares standard vs RAG translations. |

---

## Supporting Files

### `metrics_scores.png`
Summary chart of evaluation metrics across experiments.

### `STANDARD VS RAG COMPARISONS/`
Screenshots comparing Standard vs RAG translation scores at different stages:
- `2025-12-04.png` — Initial comparison.
- `After alignment and prompt change 2025-12-05 153117.png` — After alignment and prompt tuning.
- `After improving corpus and prompt change 2025-12-05 1758.png` — After corpus and prompt improvements.

### `testing scores/`
Visualizations of detailed test results:
- `COMET_vs_COMET-QE_scores.png` — COMET vs COMET-QE score comparison.
- `comet_scoring_testing_translation_barchart_temp07.png` — COMET scores bar chart (temperature 0.7).
- `testing_segment_translated_elapsedtime_temp07.png` — Translation elapsed time per segment.

---

## Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **COMET** | Reference-based | Neural MT evaluation metric; correlates highly with human judgement. |
| **BLEU** | Reference-based | Classic n-gram overlap metric for machine translation. |
| **TER** | Reference-based | Translation Edit Rate — measures the number of edits needed to transform the MT output into the reference. Lower is better. |
| **chrF** | Reference-based | Character n-gram F-score — evaluates character-level overlap between translation and reference. More robust for morphologically rich languages. |
| **CometKiwi (QE)** | Reference-free | Quality Estimation model that scores translations without needing a reference. |

---

## Prerequisites

These notebooks require the packages listed in the project root's `requirements.txt`. Key dependencies include:

- `unbabel-comet` — COMET and CometKiwi scoring
- `sacrebleu` — BLEU scoring
- `langchain` / `langchain-ollama` — LLM interaction
- `chromadb` — Vector store for RAG
- `matplotlib` / `seaborn` — Plotting

```bash
pip install -r ../requirements.txt
```
