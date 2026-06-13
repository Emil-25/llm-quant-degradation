# Paper

LaTeX source for the paper, built on the **ACL 2023** template.

## Getting the style files

`main.tex` uses `\usepackage{acl}` and `\bibliographystyle{acl_natbib}`. Those
files are **not** in this repo. Download the ACL 2023 template bundle and copy
these into `paper/`:

- `acl.sty`
- `acl_natbib.bst`

Source: the official ACL template (e.g. the ACL 2023 release on the ACL/Overleaf
template page). On Overleaf, start from the "ACL 2023 Proceedings" template and
paste `main.tex` / `references.bib` in.

## Build

```bash
# 1. Produce results and analysis (from repo root):
python scripts/aggregate_results.py --raw_dir results/raw/ --output_dir results/processed/
python scripts/analyze_results.py   --results_csv results/processed/results_table.csv --output_dir analysis/
python analysis/make_paper_tables.py --output_dir paper/tables/

# 2. Copy the generated figures the paper expects:
#    analysis/figures/degradation_heatmap.png  -> paper/figures/
#    analysis/figures/ranking_comparison.png   -> paper/figures/

# 3. Compile:
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## What is auto-generated vs. hand-written

- **Auto-generated** (do not edit by hand; regenerate instead):
  - `tables/results_main.tex`, `tables/degradation.tex`, `tables/kendall_tau.tex`
    — from `analysis/make_paper_tables.py`
  - `figures/*.png` — from `scripts/analyze_results.py`
- **Hand-written**: `main.tex` prose, `references.bib`.

## Before submission

- Every `\TODO{...}` in `main.tex` must be resolved (they mark prose that depends
  on results not yet produced).
- Every `VERIFY ...` note in `references.bib` must be replaced with the confirmed
  citation. Several related-work entries are reconstructed from the project brief
  and are **not** verified.
- Pin the exact `lm-evaluation-harness` commit in the setup section and the bib.
- De-anonymize the author block only for the camera-ready.
