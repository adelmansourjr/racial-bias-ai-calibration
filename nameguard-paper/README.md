# Exact Name-Field Noninterference in Language-Model Screening

**Adel Mansour · Technical research preprint · 16 November 2026**

[Read the PDF](https://github.com/adelmansourjr/racial-bias-ai-calibration/raw/refs/heads/main/nameguard-paper/nameguard-preprint.pdf) · [Download the LaTeX source archive](https://github.com/adelmansourjr/racial-bias-ai-calibration/raw/refs/heads/main/nameguard-paper/nameguard-latex-source.zip) · [Manuscript build checks](https://github.com/adelmansourjr/racial-bias-ai-calibration/actions/workflows/nameguard-paper.yml)

The report derives a conditional noninterference guarantee for a frozen scoring path and tests it on real résumé excerpts and a public evidence benchmark. Its front matter includes a linked table of contents, lists of figures and tables, abbreviations, notation and a vocabulary. The source includes all five TikZ figures, five tables, the bibliography and the replication appendix.

## Build the PDF

Extract the source archive or enter the `nameguard-paper` directory of a clone. Use TeX Live with Latin Modern, TikZ and `latexmk`. No model or dataset is needed to typeset the paper.

```sh
latexmk -pdf -halt-on-error -interaction=nonstopmode manuscript.tex
```

From a repository clone, `python3 nameguard-paper/verify.py` checks key reported counts against the archived results and independently verifies the finite audit bound and fidelity identities. Install `pypdf` and add `--build-dir nameguard-paper` to check the compiled PDF's navigation and layout diagnostics as well.

## Research artifacts

- [Real résumé evidence study](https://github.com/adelmansourjr/racial-bias-ai-calibration/tree/d51b96f96982bc41a42b368af6d38d459b2a95d5/nameguard-real-evidence) documents the paired experiment, reference labels, and released aggregates.
- [Public evidence study](https://github.com/adelmansourjr/racial-bias-ai-calibration/tree/d51b96f96982bc41a42b368af6d38d459b2a95d5/nameguard-risk-study) contains the protocols, case IDs, finite audit and CPU replay.
- [Runtime and admission study](https://github.com/adelmansourjr/racial-bias-ai-calibration/tree/d51b96f96982bc41a42b368af6d38d459b2a95d5/nameguard-peer-review) contains the checkpoint reconstruction and capability controls.

The evidence archives are pinned to the experimental commit. The latest report package is maintained on the default branch. The report is a venue-neutral preprint.
