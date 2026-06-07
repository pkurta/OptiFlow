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
from optiflow.models.scoring import DataType, FieldSpec, FunctionRegistry
from optiflow.optimization.algorithms import (
  DecisionSpace,
  ObjectiveEvaluator,
  ComponentTarget,
  TargetMode,
  TargetProfile,
  brute_force,
  brute_force_search_space_size,
  classic_genetic_algorithm,
  compute_total_efficiency,
  profile_constraints_satisfied,
  profile_from_slider_values,
)


class BruteForceTests(unittest.TestCase):
  def _small_space(self) -> tuple[DecisionSpace, ObjectiveEvaluator]:
    fields = [
      FieldSpec("A", DataType.BOOLEAN, 1),
      FieldSpec("B", DataType.UNSIGNED, 2),
    ]
    ensure_allowed_controls(fields)
    profile = profile_from_slider_values(50, 50, 50)
    space = DecisionSpace(fields, max_forms=2)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), profile)
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
    evaluator = ObjectiveEvaluator(FunctionRegistry(), TargetProfile.balanced())
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
    profile = profile_from_slider_values(100, 40, 40)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), profile)
    result = classic_genetic_algorithm(space, evaluator, pop_size=12, generations=10, random_seed=7)
    self.assertGreater(float(result["best_score"]), 0.0)
    self.assertEqual(len(result["history"]), 11)
    layout = result["best_layout"]
    self.assertIsNotNone(layout)
    triple = compute_total_efficiency(layout, evaluator.registry)
    self.assertGreater(triple.potency, 0.0)


class BenchmarkHelperTests(unittest.TestCase):
  def test_convergence_plateau(self) -> None:
    history = [0.1, 0.2, 0.25, 0.25, 0.25, 0.25]
    self.assertEqual(convergence_plateau_iteration(history, patience=3), 2)

  def test_precision_vs_baseline(self) -> None:
    self.assertAlmostEqual(precision_vs_baseline(0.9, 1.0) or 0.0, 0.9)
    self.assertIsNone(precision_vs_baseline(0.5, 0.0))


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
    profile = TargetProfile(
      potency=ComponentTarget(TargetMode.Certain, 0.5),
      operativeness=ComponentTarget(TargetMode.Any),
      resource_saving=ComponentTarget(TargetMode.Any),
    )
    evaluator = ObjectiveEvaluator(FunctionRegistry(), profile)
    bf = brute_force(space, evaluator)
    ga = classic_genetic_algorithm(space, evaluator, pop_size=10, generations=15, random_seed=1)
    self.assertGreaterEqual(float(bf["best_score"]), float(ga["best_score"]) - 1e-9)

  def test_constraint_validation_integration(self) -> None:
    fields = [FieldSpec("Flag", DataType.BOOLEAN, 1)]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=1)
    profile = profile_from_slider_values(95, 0, 0)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), profile)
    result = brute_force(space, evaluator)
    layout = result["best_layout"]
    assert layout is not None
    triple = compute_total_efficiency(layout, evaluator.registry)
    self.assertIsInstance(profile_constraints_satisfied(triple, profile), bool)


class MarkdownFormatTests(unittest.TestCase):
  def test_format_benchmark_markdown_table(self) -> None:
    from optiflow.benchmarks import AlgorithmBenchmarkStats

    stats = {name: AlgorithmBenchmarkStats() for name in BENCHMARK_ALGORITHMS}
    stats["GA"].precision_rates = [0.95, 0.98]
    stats["GA"].convergence_iterations = [10, 12]
    stats["GA"].constraint_passes = 2
    stats["GA"].constraint_total = 2
    stats["GA"].baseline_runs = 2
    md = format_benchmark_markdown(stats, runs_count=2, baseline_computable_runs=2)
    self.assertIn("| GA |", md)
    self.assertIn("0.9650", md)


if __name__ == "__main__":
  unittest.main()
