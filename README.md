# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains our team-developed code and the main experiment notebook.

## Source of truth

Keep the official `student_resource/` directory exactly as provided by the organizers.
Do not edit or regenerate its files. In particular, do not copy or replace the official
`utils/validate_submission.py`.

## Project layout

```text
student_resource/                  # official challenge files; untouched

notebooks/
  01_Main_Amazon_ML_Challenge.ipynb

src/
  config.py
  io_utils.py
  normalization.py
  blocking.py
  features.py
  evaluation.py
  pipeline.py

scripts/
  run_pipeline.py

tests/
  test_core.py

outputs/
  experiments/
  candidates/
  final/
```

## Design principle

The pipeline is intentionally separated into:

1. data loading and validation
2. normalization
3. blocking / candidate generation
4. pairwise feature generation
5. pairwise matching model
6. threshold selection using validation macro F0.5
7. final multi-match inference
8. submission generation
9. official submission validation

The code in `src/` is reusable. The notebook is the research and integration layer.

## Current status

The starter is ready for the next phase: profile Source 2 and Source 3, measure
blocking recall/volume, then implement and benchmark matching models. We should not
freeze blocking rules or thresholds before that analysis.

## Environment

```bash
pip install -r requirements.txt
pytest -q
```

The official validator remains in `student_resource/utils/validate_submission.py`.
