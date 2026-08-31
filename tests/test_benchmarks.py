from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from optiflow.benchmarks import (
  BENCHMARK_ALGORITHMS,
  convergence_plateau_iteration,
  ensure_allowed_controls,
  format_benchmark_markdown,
  precision_vs_baseline,
  random_benchmark_snapshot,
  run_optimization_benchmark,
)
from optiflow.models.scoring import DataType, EfficiencyTriple, FieldSpec, FunctionRegistry
from optiflow.optimization.algorithms import (
  CriterionWeights,
  DecisionSpace,
  ObjectiveEvaluator,
  OptimizationControl,
  brute_force,
  brute_force_search_space_size,
  calculate_fitness,
  classic_genetic_algorithm,
  compute_total_efficiency,
  random_search,
  redistribute_weight_ticks,
)


class CriterionWeightsTests(unittest.TestCase):
  def test_from_raw_normalizes_to_unit_sum(self) -> None:
    weights = CriterionWeights.from_raw(2.0, 1.0, 1.0)
    self.assertAlmostEqual(sum(weights.as_tuple()), 1.0)
    self.assertAlmostEqual(weights.w_potency, 0.5)
    self.assertAlmostEqual(weights.w_operativeness, 0.25)
    self.assertAlmostEqual(weights.w_resource_saving, 0.25)

  def test_zero_vector_falls_back_to_equal_weights(self) -> None:
    weights = CriterionWeights.from_raw(0.0, 0.0, 0.0)
    self.assertAlmostEqual(sum(weights.as_tuple()), 1.0)
    for value in weights.as_tuple():
      self.assertAlmostEqual(value, 1.0 / 3.0)

  def test_rejects_unnormalized_constructor(self) -> None:
    with self.assertRaises(ValueError):
      CriterionWeights(0.5, 0.5, 0.5)

  def test_ticks_round_trip_sums_to_100(self) -> None:
    weights = CriterionWeights.from_raw(0.70, 0.20, 0.10)
    ticks = weights.to_ticks()
    self.assertEqual(sum(ticks), 100)
    restored = CriterionWeights.from_ticks(*ticks)
    self.assertAlmostEqual(sum(restored.as_tuple()), 1.0)

  def test_balanced_weights_are_exactly_one_third(self) -> None:
    weights = CriterionWeights.balanced()
    self.assertTrue(weights.is_equal())
    for value in weights.as_tuple():
      self.assertAlmostEqual(value, 1.0 / 3.0)
    self.assertEqual(weights.display_parts(), ("1/3", "1/3", "1/3"))
    self.assertAlmostEqual(sum(weights.as_tuple()), 1.0)

  def test_legacy_rounded_balance_is_not_equal(self) -> None:
    weights = CriterionWeights.from_raw(0.34, 0.33, 0.33)
    self.assertFalse(weights.is_equal())
    self.assertEqual(weights.display_parts(digits=2), ("0.34", "0.33", "0.33"))

  def test_balance_preset_tuple_is_equal(self) -> None:
    from optiflow.optimization.algorithms import DEFAULT_WEIGHT_PRESET, WEIGHT_PRESETS

    weights = CriterionWeights.from_raw(*WEIGHT_PRESETS[DEFAULT_WEIGHT_PRESET])
    self.assertTrue(weights.is_equal())

  def test_redistribute_keeps_unit_simplex(self) -> None:
    ticks = redistribute_weight_ticks((34, 33, 33), 0, 70)
    self.assertEqual(sum(ticks), 100)
    self.assertEqual(ticks[0], 70)

  def test_redistribute_from_equal_illustration_sums_to_100(self) -> None:
    ticks = redistribute_weight_ticks((33, 33, 33), 0, 50)
    self.assertEqual(sum(ticks), 100)
    self.assertEqual(ticks[0], 50)
    self.assertEqual(ticks[1], ticks[2])

  def test_calculate_fitness_is_weighted_sum_minus_penalties(self) -> None:
    triple = EfficiencyTriple(0.8, 0.4, 0.2)
    weights = CriterionWeights.from_raw(0.5, 0.3, 0.2)
    expected = 0.5 * 0.8 + 0.3 * 0.4 + 0.2 * 0.2 - 0.1
    self.assertAlmostEqual(calculate_fitness(triple, weights, penalties=0.1), expected)


class ProgressControlTests(unittest.TestCase):
  def test_random_search_stops_when_cancelled(self) -> None:
    fields = [
      FieldSpec("A", DataType.BOOLEAN, 1),
      FieldSpec("B", DataType.UNSIGNED, 2),
    ]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=1)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), CriterionWeights.balanced())
    control = OptimizationControl(min_interval_s=0.0)

    def on_progress(report) -> None:
      if report.iteration >= 3:
        control.cancel()

    control._on_progress = on_progress
    result = random_search(space, evaluator, iterations=400, random_seed=3, control=control)
    self.assertLess(len(result["history"]), 400)
    self.assertIsNotNone(result["best_layout"])
    self.assertTrue(control.cancelled)


class BruteForceTests(unittest.TestCase):
  def _small_space(self) -> tuple[DecisionSpace, ObjectiveEvaluator]:
    fields = [
      FieldSpec("A", DataType.BOOLEAN, 1),
      FieldSpec("B", DataType.UNSIGNED, 2),
    ]
    ensure_allowed_controls(fields)
    weights = CriterionWeights.from_raw(1.0, 1.0, 1.0)
    space = DecisionSpace(fields, max_forms=2)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)
    return space, evaluator

  def test_search_space_size(self) -> None:
    space, _ = self._small_space()
    # 1 control × 2 controls × C(2+2-1, 2-1)=3 partitions = 6
    self.assertEqual(brute_force_search_space_size(space), 6)

  def test_brute_force_finds_global_maximum(self) -> None:
    space, evaluator = self._small_space()
    result = brute_force(space, evaluator)
    self.assertIn("best_score", result)
    self.assertIn("best_layout", result)
    self.assertEqual(len(result["history"]), brute_force_search_space_size(space))

    best_score = float(result["best_score"])
    layout = result["best_layout"]
    assert layout is not None
    exhaustive_scores = []
    from optiflow.optimization.algorithms import _enumerate_field_partitions
    import itertools

    cardinalities = space.cardinalities()
    for combo in itertools.product(*[range(c) for c in cardinalities]):
      controls = space.all_controls(list(combo))
      for partition in _enumerate_field_partitions(space.control_dim(), space.partition_dim()):
        from optiflow.models.scoring import build_interface_layout_from_partition

        trial = build_interface_layout_from_partition(space.fields, controls, partition)
        exhaustive_scores.append(evaluator.scalar_fitness(trial))
    self.assertAlmostEqual(best_score, max(exhaustive_scores))

  def test_brute_force_raises_when_space_too_large(self) -> None:
    fields = [FieldSpec(f"F{i}", DataType.TEXT, 10) for i in range(8)]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=5)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), CriterionWeights.balanced())
    self.assertGreater(brute_force_search_space_size(space), 50_000)
    with self.assertRaises(ValueError):
      brute_force(space, evaluator)


class ClassicGATests(unittest.TestCase):
  def test_classic_ga_runs_and_returns_layout(self) -> None:
    fields = [
      FieldSpec("Age", DataType.UNSIGNED, 3),
      FieldSpec("Name", DataType.TEXT, 8),
      FieldSpec("OK", DataType.BOOLEAN, 1),
    ]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=2)
    weights = CriterionWeights.from_raw(0.70, 0.20, 0.10)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)
    result = classic_genetic_algorithm(space, evaluator, pop_size=12, generations=10, random_seed=7)
    self.assertGreater(float(result["best_score"]), 0.0)
    self.assertEqual(len(result["history"]), 11)
    layout = result["best_layout"]
    self.assertIsNotNone(layout)
    triple = compute_total_efficiency(layout, evaluator.registry)
    self.assertGreater(triple.potency, 0.0)
    self.assertAlmostEqual(
      float(result["best_score"]),
      calculate_fitness(triple, weights),
      places=6,
    )


class BenchmarkHelperTests(unittest.TestCase):
  def test_convergence_plateau(self) -> None:
    history = [0.1, 0.2, 0.25, 0.25, 0.25, 0.25]
    self.assertEqual(convergence_plateau_iteration(history, patience=3), 2)

  def test_precision_vs_baseline(self) -> None:
    self.assertAlmostEqual(precision_vs_baseline(0.9, 1.0) or 0.0, 0.9)
    self.assertIsNone(precision_vs_baseline(0.5, 0.0))

  def test_random_snapshot_weights_are_normalized(self) -> None:
    import random

    fields, max_forms, weights = random_benchmark_snapshot(random.Random(7))
    self.assertGreaterEqual(len(fields), 2)
    self.assertGreaterEqual(max_forms, 1)
    self.assertAlmostEqual(sum(weights.as_tuple()), 1.0)


class MonteCarloBenchmarkTests(unittest.TestCase):
  def test_run_optimization_benchmark_small(self) -> None:
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmp:
      stats = run_optimization_benchmark(
        runs_count=8,
        random_seed=123,
        output_dir=Path(tmp),
        log_markdown=False,
      )
      self.assertEqual(set(stats.keys()), set(BENCHMARK_ALGORITHMS))
      report = Path(tmp) / "benchmark_report.md"
      self.assertTrue(report.exists())
      text = report.read_text(encoding="utf-8")
      self.assertIn("Precision Rate", text)
      self.assertIn("BruteForce", text)

  def test_benchmark_brute_force_beats_or_matches_metaheuristics_on_tiny_space(self) -> None:
    fields = [
      FieldSpec("X", DataType.BOOLEAN, 1),
      FieldSpec("Y", DataType.UNSIGNED, 2),
    ]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=1)
    weights = CriterionWeights.from_raw(0.5, 0.3, 0.2)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)
    bf = brute_force(space, evaluator)
    ga = classic_genetic_algorithm(space, evaluator, pop_size=10, generations=15, random_seed=1)
    self.assertGreaterEqual(float(bf["best_score"]), float(ga["best_score"]) - 1e-9)


class MarkdownFormatTests(unittest.TestCase):
  def test_format_benchmark_markdown_table(self) -> None:
    from optiflow.benchmarks import AlgorithmBenchmarkStats

    stats = {name: AlgorithmBenchmarkStats() for name in BENCHMARK_ALGORITHMS}
    stats["GA"].precision_rates = [0.95, 0.98]
    stats["GA"].convergence_iterations = [10, 12]
    stats["GA"].baseline_runs = 2
    md = format_benchmark_markdown(stats, runs_count=2, baseline_computable_runs=2)
    self.assertIn("| GA |", md)
    self.assertIn("0.9650", md)


if __name__ == "__main__":
  unittest.main()
