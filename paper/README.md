# paper/

**Draft — proposed, not accepted.**  No human revision or
external review yet; the numbers in the paper are repo-internal
evidence — this file ships in the public `handl-lang/kernelopt`
tree but the paper itself is not an archival claim.

- `paper.qmd` — the quarto source
- `references.bib` — bibliography (incl. the internal tech-report
  entries for the eqsat platform paper and the graphopt companion)
- `paper.pdf` — the rendered draft

Rebuild: `quarto render paper.qmd --to pdf` (quarto + a LaTeX engine;
TinyTeX works).

Every quantitative claim cites the audit doc, bench module, or note
that produced it — the evidence ledger is `../docs/` (the three
results docs) plus `../notes/kernel-engineer-board.md`.
