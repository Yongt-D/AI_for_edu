# Reproduction guide

Run all commands from the repository root in the `aiedu` environment. The environment file records the main training-library versions; no claim of bitwise equivalence across hardware is made.

## A. Validate and redraw saved results

Download, verify and extract the companion package according to DATA_ACCESS.md. Then run:

```bash
python scripts/15_validate_revision.py
python scripts/20_validate_peer_revision.py
python scripts/23_validate_second_review.py
python scripts/13_revision_figures.py
python scripts/19_peer_manuscript.py
python scripts/22_second_review_manuscript.py
python scripts/25_decision_composition.py
```

These public versions of scripts 19 and 22 generate tables and figures only. The validation scripts check numerical results; private manuscript checks have been removed. File identifiers retain their original names to connect the code with saved experiment metadata.

## B. Refit from original source data

1. Obtain `dataset.zip` and `course_info.json` from the KU Leuven record cited in DATA_ACCESS.md. Put `course_info.json` directly in `data/raw/` and extract the archive below that directory, preserving filenames and year folders. The loader recursively discovers `*_log_activity.csv` and reads the matching participation/content/forum XLSX files.
2. Run the following sequence. OULAD is downloaded separately by its dedicated script.

```bash
python scripts/09_prepare_revision.py
python scripts/10_run_revision.py
python scripts/download_oulad.py
python scripts/11_external_oulad.py
python scripts/12_revision_statistics.py
python scripts/13_revision_figures.py
python scripts/15_validate_revision.py
python scripts/17_peer_experiments.py
python scripts/18_peer_statistics.py
python scripts/19_peer_manuscript.py
python scripts/20_validate_peer_revision.py
python scripts/21_second_review_analysis.py
python scripts/22_second_review_manuscript.py
python scripts/23_validate_second_review.py
python scripts/25_decision_composition.py
```

The full workflow refits models and runs 2,000-resample analyses; it is more expensive than reading saved results. Do not combine old outputs with a different configuration. Numerical validators compare saved metrics against prediction-level results. The decision-composition generator checks 150 fitted decompositions and exports the five-seed means and ranges.

## Analytical files

`configs/revision_v2.json` specifies the base analysis. Protocol JSON files describe the factorial sensitivity and course-stratified analyses. Historical provenance strings in those files are retained; they do not imply preregistration. Saved source-code hashes describe the original runs, whereas Git identifies this public packaging version. Public-only changes separate figure generation from manuscript writing and make the Windows interpreter path portable; analytical fitting formulas are unchanged.
