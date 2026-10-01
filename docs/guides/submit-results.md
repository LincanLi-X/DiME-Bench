# Submit results to the leaderboard

The public leaderboard accepts only the versioned v1 submission envelope.

```bash
python scripts/release_benchmark.py \
  --summary reports/benchmark-summary.json \
  --output-dir results/releases/v1.0.0

python scripts/build_leaderboard.py \
  --submission results/releases/v1.0.0/result-submission.json
```

Before submission:

- independently run `dimebench evaluate` on every sealed run;
- retain predictions, sample metrics, manifests, and environment files;
- confirm protocol/model/dataset revisions and AR/dLLM decoding budgets;
- validate `release-manifest.json` hashes;
- do not edit the embedded summary by hand.

The static builder verifies that its copied `benchmark-summary.json` is
byte-identical to the release summary. Bare summaries and tampered submission
hashes are rejected.
