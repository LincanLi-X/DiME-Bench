# Add a dataset

1. Add a versioned source manifest under `data/manifests/` with license,
   citation, source revision, splits, record kind, and preprocessing strategy.
2. Add the dataset to `data/registry.yaml` and the intended suite.
3. Normalize upstream rows into one of the typed sample records.
4. Commit only a small legal fixture plus the data card; do not commit full
   restricted datasets.
5. Freeze the split and verify that repeated preparation produces identical
   sample IDs and hashes.

Custom local data may also be written directly as normalized JSONL. The run
configuration must use `source: local` and reference that file.

See [`examples/add_custom_dataset.py`](../../examples/add_custom_dataset.py).
