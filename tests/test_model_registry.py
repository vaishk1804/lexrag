"""
Tests for MLflow Model Registry integration.

These use mock MLflow run objects rather than a real MLflow server,
so they run instantly and don't depend on any experiment data existing.
"""

from unittest.mock import MagicMock

from src.evaluation.model_registry import _is_run_complete, MIN_COMPLETION_FRACTION


def make_mock_run(metrics: dict, run_name: str = "test_run"):
    """Build a mock MLflow Run object with the given metrics dict."""
    run = MagicMock()
    run.data.metrics = metrics
    run.data.tags = {"mlflow.runName": run_name}
    run.info.run_id = "fake_run_id_123"
    return run


class TestIsRunComplete:

    def test_complete_run_returns_true(self):
        run = make_mock_run({
            "faithfulness": 0.85,
            "faithfulness_completeness": 1.0,
            "context_precision": 0.91,
            "context_precision_completeness": 1.0,
            "context_recall": 0.73,
            "context_recall_completeness": 1.0,
            "composite": 0.83,
        })
        assert _is_run_complete(run) is True

    def test_partial_run_returns_false(self):
        run = make_mock_run({
            "faithfulness": 0.88,
            "faithfulness_completeness": 0.7,  # only 70% completed
            "context_precision": 0.94,
            "context_precision_completeness": 0.9,
            "context_recall": 0.74,
            "context_recall_completeness": 0.8,
            "composite": 0.85,
        })
        assert _is_run_complete(run) is False

    def test_run_with_no_completeness_metrics_returns_false(self):
        """
        Older runs that predate completeness tracking should be excluded,
        not silently trusted as complete.
        """
        run = make_mock_run({
            "faithfulness": 0.85,
            "context_precision": 0.91,
            "context_recall": 0.73,
            "composite": 0.83,
            # no *_completeness keys at all
        })
        assert _is_run_complete(run) is False

    def test_mixed_completeness_one_partial_metric_fails_whole_run(self):
        """
        If even ONE metric's completeness is below threshold, the whole
        run should be excluded — partial data on any metric makes the
        composite score (which averages across all metrics) untrustworthy.
        """
        run = make_mock_run({
            "faithfulness_completeness": 1.0,
            "context_precision_completeness": 1.0,
            "context_recall_completeness": 0.9,  # this one metric is partial
            "composite": 0.85,
        })
        assert _is_run_complete(run) is False

    def test_exactly_at_threshold_passes(self):
        run = make_mock_run({
            "faithfulness_completeness": MIN_COMPLETION_FRACTION,
            "context_precision_completeness": MIN_COMPLETION_FRACTION,
            "context_recall_completeness": MIN_COMPLETION_FRACTION,
            "composite": 0.83,
        })
        assert _is_run_complete(run) is True


class TestGetBestRunSelection:
    """
    Tests the selection logic conceptually: given a mix of complete and
    partial runs, the best COMPLETE run should win, even if a partial
    run has a higher raw composite score.
    """

    def test_partial_run_excluded_even_with_higher_composite(self):
        """
        This is the exact bug being prevented: a partial run with a
        higher (less trustworthy) composite score must not beat a
        complete run with a lower but fully-measured composite score.
        """
        partial_run = make_mock_run({
            "faithfulness_completeness": 0.7,
            "context_precision_completeness": 0.9,
            "context_recall_completeness": 0.8,
            "composite": 0.90,  # looks better...
        }, run_name="partial_high_score")

        complete_run = make_mock_run({
            "faithfulness_completeness": 1.0,
            "context_precision_completeness": 1.0,
            "context_recall_completeness": 1.0,
            "composite": 0.83,  # ...but this is the trustworthy one
        }, run_name="complete_lower_score")

        assert _is_run_complete(partial_run) is False
        assert _is_run_complete(complete_run) is True
        # In get_best_run, partial_run would be filtered out entirely
        # regardless of its higher composite score.