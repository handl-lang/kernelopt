# paper/

**Draft — proposed, not accepted.  PRIVATE.**  No human revision or
external review yet; this repository is private and the numbers in
the paper are repo-internal — do not copy into public trees.

- `paper.qmd` — the quarto source
- `references.bib` — bibliography (incl. the internal tech-report
  entries for the eqsat platform paper and the graphopt companion)
- `paper.pdf` — the rendered draft

Rebuild: `quarto render paper.qmd --to pdf` (quarto + a LaTeX engine;
TinyTeX works).

Every quantitative claim cites the audit doc, bench module, or note
that produced it — the evidence ledger is `../docs/` (the three
results docs) plus `../notes/kernel-engineer-board.md`.
