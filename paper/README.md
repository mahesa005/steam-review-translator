# Paper

`main.tex` + `references.bib`, plain `article` class so it converts cleanly to Word.

- **Overleaf:** upload `main.tex`, `references.bib` and the `beam/` folder (New Project > Upload Project) and compile with pdfLaTeX.
- **Local:** `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
- **Word (.docx):** `pandoc main.tex --citeproc --bibliography references.bib -o paper.docx`
  (tables and red `\todo{}` notes survive; check the figure by hand).

Red `[TODO: ...]` marks text that is still missing. Search for `\todo` before submitting, and
verify every entry in `references.bib`.
