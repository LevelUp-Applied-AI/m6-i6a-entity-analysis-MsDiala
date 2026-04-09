# Integration 6A — Entity Analysis Pipeline

Module 6 Week A integration task for AI.SPIRE Applied AI & ML Systems.

Build a corpus-level entity analysis pipeline that processes climate articles, extracts entities, computes statistics, and produces visualizations.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Tasks

Complete the five functions in `entity_analysis.py`:
1. `load_and_filter_corpus` — Load dataset and filter to English texts
2. `run_ner_pipeline` — Extract entities using spaCy or Hugging Face
3. `aggregate_entity_stats` — Compute frequency and co-occurrence statistics
4. `visualize_entity_distribution` — Create a bar chart of top entities
5. `generate_report` — Produce a structured entity analysis report

## Submission

1. Create a branch: `integration-6a-entity-analysis`
2. Complete `entity_analysis.py`
3. Open a PR to `main`
4. Paste your PR URL into TalentLMS → Module 6 Week A → Integration 6A

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
