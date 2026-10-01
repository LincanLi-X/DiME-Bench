# DiME-Bench v1 Dataset Cards

This directory documents the data contracts used by the four DiME-Bench
tracks. The repository commits manifests, cards, and synthetic schema fixtures
only. It does not redistribute full upstream datasets or restricted data.

The file `data/samples/dime_bench_v1_samples.jsonl` contains one synthetic,
non-evaluation fixture per logical dataset. Running `dimebench prepare` without
`--full` prepares these fixtures so package and schema changes can be tested
without downloading benchmark data.

Full preparation follows each YAML manifest under `data/manifests/`. Upstream
licenses, access requirements, and immutable revisions must be reviewed before
publishing a frozen release split. Generated processed files, raw downloads,
and caches are intentionally ignored by Git.

| Track | Card |
|---|---|
| General capability | [track1-general.md](track1-general.md) |
| Text infilling | [track2-infilling.md](track2-infilling.md) |
| Text editing | [track3-editing.md](track3-editing.md) |
| Reasoning regimes | [track4-reasoning.md](track4-reasoning.md) |
