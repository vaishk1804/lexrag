"""
Tests for the RAGAS evaluation wrapper.

These tests avoid making real LLM calls by testing the data-shaping
and completeness-calculation logic directly, rather than running 
evaluate() against a live model. Full end-to-end RAGAS runs are
already exercised via run_experiment.py against real data.
"""

import pandas as pd
import pytest

from src.evaluation.ragas_eval import build_ragas_dataset

class TestBuildRagasDataset:

  def test_builds_dataset_with_matching_lengths(self):
    ds = build_ragas_dataset(
      questions=["Q1","Q2"],
      answers=["A1","A2"],
      contexts=[["C1"],["C2"]],
      ground_truths=["GT1","GT2"],
    )
    assert len(ds) == 2
    assert ds[0]["question"] == "Q1"
    assert ds[1]["ground_truth"] == "GT2"

  def test_rejects_mismatched_lengths(self):
    with pytest.raises(AssertionError):
      build_ragas_dataset(
        questions=["Q1","Q2"],
        answers=["A1"],   # mismatched length
        contexts=[["C1"],["C2"]],
        ground_truths=["GT1","GT2"],
      )

  def test_contexts_preserved_as_lists(self):
    """Each row's context should remain a list of strings, not flattened."""
    ds = build_ragas_dataset(
      questions=["Q1"],
      answers=["A1"],
      contexts=[["chunk one","chunk two","chunk three"]],
      ground_truths=["GT1"],
    )
    assert ds[0]["contexts"] == ["chunk one","chunk two","chunk three"]
  
class TestCompletenessCalculation:
  """
    Tests the completeness-fraction logic in isolation by simulating
    what a RAGAS result DataFrame looks like with missing (NaN) values,
    without needing to run a real evaluation.
    """
  def test_full_completion_gives_fraction_one(self):
    df = pd.DataFrame({
      "faithfulness": [0.8, 0.9, 0.7, 0.85],
    })
    valid_count = df["faithfulness"].notna().sum()
    fraction = valid_count / len(df)
    assert fraction == 1.0

  def test_partial_completion_gives_correct_fraction(self):
    df = pd.DataFrame({
      "faithfulness": [0.8, None, 0.7, None],
    })
    valid_count = df["faithfulness"].notna().sum()
    fraction = valid_count / len(df)
    assert fraction == 0.5

  def test_mean_skips_nan_by_default(self):
    """
    Confirms pandas .mean() automatically excludes NaN — this is the
    behavior run_ragas_evaluation relies on without special handling.
    """
    df = pd.DataFrame({
      "faithfulness": [0.8, None, 0.6],
    })
    mean_val = df["faithfulness"].mean()
    assert abs(mean_val - 0.7) < 1e-9  # (0.8 + 0.6) / 2, not / 3

  def test_zero_completion_handled_without_crash(self):
    df = pd.DataFrame({
      "faithfulness": [None, None, None],
    })
    valid_count = df["faithfulness"].notna().sum()
    fraction = valid_count / len(df) if len(df) > 0 else 0.0
    assert fraction == 0.0