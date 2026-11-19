# Report, README and demo numbers are generated from committed JSON

Tables in the LaTeX report, the results blocks of README.md and results/SUMMARY.md, the figures
and the demo data are written by scripts that read only the committed run summaries and benchmark
files. A test regenerates the tables and README blocks and fails if the committed text differs.

Step 1 found hand-typed numbers that disagreed with the JSON in README, docs and the report.
Generated text costs a little readability in the sources (marker comments, \input files) and buys
the guarantee that every number traces to a run.

Consequence: prose may describe results qualitatively, but a number in prose must be copied from
a generated table in the same commit, and the provenance file says which commit produced the JSON.
