# Add a metric

1. Subclass `StandardEvaluator` for a primary task score or
   `MechanismEvaluator` for a diagnostic whose semantic inputs may be absent.
2. Fix `METRIC_ID`, `VERSION`, valid range, and every setting that affects the
   configuration hash.
3. Return a value and auditable details from `_score()`.
4. Add unit tests, frozen golden cases, invalid-input tests, and aggregation
   coverage tests.
5. Register the metric in the applicable task configuration and frozen
   benchmark specification.

Primary failures receive zero. Missing evidence for a mechanism metric must be
represented as ineligible/null, never silently dropped or converted to zero.

See [`examples/add_custom_metric.py`](../../examples/add_custom_metric.py).
