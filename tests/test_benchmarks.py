from __future__ import annotations

import logging
import math
import random
import tempfile
import unittest
from pathlib import Path
from typing import List

from optiflow.benchmarks import (
  BENCHMARK_ALGORITHMS,
  convergence_plateau_iteration,
  ensure_allowed_controls,
  format_benchmark_markdown,
  precision_vs_baseline,
  random_benchmark_snapshot,
  run_optimization_benchmark,
)
from optiflow.models.scoring import (
  ControlType,
  DataType,
  EfficiencyTriple,
  FieldSpec,
  FunctionRegistry,
  build_interface_layout_from_partition,
)
from optiflow.models.layout_io import interface_layout_to_payload
from optiflow.optimization.algorithms import (
  CriterionWeights,
  DecisionSpace,
  ObjectiveEvaluator,
  OptimizationControl,
  brute_force,
  brute_force_search_space_size,
  calculate_fitness,
  classic_genetic_algorithm,
  clip_fitness_for_display,
  compute_total_efficiency,
  nsga2,
  pso,
  random_search,
  redistribute_weight_ticks,
  simulated_annealing,
)
from optiflow.optimization.corrections import (
  DEFAULT_MILLER_PENALTY_WEIGHT,
  MILLER_HARD_LIMIT,
  compute_miller_penalty,
  is_miller_feasible,
  miller_feasibility_margin,
  miller_feasibility_warning,
)
from optiflow.optimization.runner import run_optimization_suite
from optiflow.ui.run_results import build_algorithm_summaries


def _mixed_fields_for_miller_tests(n: int, seed: int) -> List[FieldSpec]:
  """D fields with a TEXT/UNSIGNED/BOOLEAN mix, sized like a realistic form (not the
  Monte Carlo benchmark's D<=4 toy fixtures) -- for the D=15-20 Inv_3 scenarios."""
  rng = random.Random(seed)
  types = [DataType.TEXT, DataType.UNSIGNED, DataType.BOOLEAN]
  fields: List[FieldSpec] = []
  for i in range(n):
    dtype = types[i % 3]
    size = 1 if dtype == DataType.BOOLEAN else rng.randint(1, 10)
    fields.append(FieldSpec(f"F{i}", dtype, size))
  ensure_allowed_controls(fields)
  return fields


def _max_form_size(layout) -> int:
  return max((len(form.elements) for form in layout.forms), default=0)


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

  def test_legacy_organisation_preset_aliases_resolve(self) -> None:
    from optiflow.optimization.algorithms import (
      WEIGHT_PRESETS,
      resolve_weight_preset,
    )

    self.assertEqual(
      resolve_weight_preset("Call-центр / МЧС — упор на оперативность"),
      "Упор на оперативность",
    )
    self.assertEqual(
      resolve_weight_preset("Банк — упор на результативность"),
      "Упор на результативность",
    )
    self.assertEqual(
      resolve_weight_preset("Массовый сервис — упор на ресурсоэкономность"),
      "Упор на ресурсоэкономность",
    )
    self.assertIn("Упор на оперативность", WEIGHT_PRESETS)
    self.assertNotIn("Call-центр / МЧС — упор на оперативность", WEIGHT_PRESETS)

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


class MillerConstraintTests(unittest.TestCase):
  """Inv_3 (Miller 7±2): ∑ c_elem <= 9 enforced as a penalty in F."""

  def _layout_with_form_sizes(self, sizes: list[int]):
    total = sum(sizes)
    fields = [FieldSpec(f"F{i}", DataType.BOOLEAN, 1) for i in range(total)]
    ensure_allowed_controls(fields)
    controls = [ControlType.CHECKBOX] * total
    return build_interface_layout_from_partition(fields, controls, sizes)

  def test_penalty_zero_at_and_below_hard_limit(self) -> None:
    self.assertEqual(MILLER_HARD_LIMIT, 9)
    layout = self._layout_with_form_sizes([MILLER_HARD_LIMIT])
    self.assertEqual(compute_miller_penalty(layout), 0.0)
    small_layout = self._layout_with_form_sizes([3, 2])
    self.assertEqual(compute_miller_penalty(small_layout), 0.0)

  def test_penalty_grows_quadratically_beyond_hard_limit(self) -> None:
    layout_10 = self._layout_with_form_sizes([10])  # excess = 1
    layout_12 = self._layout_with_form_sizes([12])  # excess = 3
    penalty_10 = compute_miller_penalty(layout_10, penalty_weight=1.0)
    penalty_12 = compute_miller_penalty(layout_12, penalty_weight=1.0)
    self.assertAlmostEqual(penalty_10, 1.0)   # 1**2
    self.assertAlmostEqual(penalty_12, 9.0)   # 3**2
    self.assertGreater(penalty_12, penalty_10)
    # Quadratic, not linear: penalty ratio (9x) outpaces excess ratio (3x).
    self.assertAlmostEqual(penalty_12 / penalty_10, 9.0)

  def test_multiple_forms_sum_independently(self) -> None:
    layout = self._layout_with_form_sizes([10, 11, 5])  # excess 1 and 2, one compliant
    penalty = compute_miller_penalty(layout, penalty_weight=1.0)
    self.assertAlmostEqual(penalty, 1.0 + 4.0)

  def test_violating_layout_scores_lower_fitness_for_equal_efficiency(self) -> None:
    """Same underlying E = (P, O, R), worse Inv_3 distribution -> lower F."""
    from unittest.mock import patch

    weights = CriterionWeights.balanced()
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)
    compliant_layout = self._layout_with_form_sizes([5, 4])
    violating_layout = self._layout_with_form_sizes([10, 9])
    fixed_triple = EfficiencyTriple(0.8, 0.8, 0.8)

    with patch(
      "optiflow.optimization.algorithms.compute_total_efficiency",
      return_value=fixed_triple,
    ):
      compliant_score = evaluator.scalar_fitness(compliant_layout)
      violating_score = evaluator.scalar_fitness(violating_layout)

    self.assertLess(violating_score, compliant_score)
    self.assertAlmostEqual(
      compliant_score,
      calculate_fitness(fixed_triple, weights, penalties=0.0),
    )

  def test_scalar_fitness_matches_manual_penalized_formula(self) -> None:
    fields = [FieldSpec(f"F{i}", DataType.UNSIGNED, 2) for i in range(11)]
    ensure_allowed_controls(fields)
    weights = CriterionWeights.from_raw(0.4, 0.3, 0.3)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)
    space = DecisionSpace(fields, max_forms=1)  # forces k=11 > hard limit
    layout = space.decode_layout(space.random_vector(), evaluator.registry)
    triple = compute_total_efficiency(layout, evaluator.registry)
    penalty = compute_miller_penalty(layout, evaluator.penalty_weight)
    self.assertGreater(penalty, 0.0)
    expected = calculate_fitness(triple, weights, penalties=penalty)
    self.assertAlmostEqual(evaluator.scalar_fitness(layout), expected, places=9)

  def test_brute_force_and_nsga2_apply_same_penalty_as_scalar_fitness(self) -> None:
    fields = [FieldSpec(f"F{i}", DataType.BOOLEAN, 1) for i in range(10)]
    ensure_allowed_controls(fields)
    weights = CriterionWeights.balanced()
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)
    space = DecisionSpace(fields, max_forms=1)  # single form, k=10 always > hard limit

    bf = brute_force(space, evaluator)
    bf_layout = bf["best_layout"]
    self.assertEqual(len(bf_layout.forms[0].elements), 10)
    self.assertAlmostEqual(float(bf["best_score"]), evaluator.scalar_fitness(bf_layout), places=9)

    result = nsga2(space, evaluator, pop_size=10, generations=5, random_seed=1)
    n_layout = result["best_layout"]
    self.assertIsNotNone(n_layout)
    self.assertAlmostEqual(
      float(result["best_score"]),
      evaluator.scalar_fitness(n_layout),
      places=6,
    )


class RealisticMillerConvergenceTests(unittest.TestCase):
  """Inv_3 открытый вопрос №1 (аудит, врезка раздела 2.3): действительно ли штраф с
  DEFAULT_MILLER_PENALTY_WEIGHT=0.01 меняет итоговый best_layout на реалистичном D
  (15-20 полей), а не остаётся арифметически ничтожным на фоне диапазона E.

  Это эмпирическая проверка поведения оптимизаторов, а не проверка формулы штрафа
  (та уже покрыта MillerConstraintTests) -- поэтому seed фиксирован для каждого
  прогона, но используется несколько seed и порог "не хуже N из M", чтобы не
  зависеть от одного случайного старта популяции/частиц.
  """

  # D=17, N=2: жёсткая ёмкость экранов 9*2=18 -> Inv_3 структурно выполним, но
  # с небольшим запасом (1 поле). При этом ансамбль метаэвристик, оптимизируя
  # только E (P,O,R), уже слегка предпочитает МЕНЬШЕ экранов: apply_form_step_correction
  # штрафует каждый дополнительный непустой экран мультипликативным коэффициентом
  # 0.995**(i-1), а evaluate_form/apply_element_position_correction почти
  # безразличны к тому, как элементы распределены между экранами (их вклад в
  # E зависит от суммарного числа элементов, а не от разбиения). Из-за этого
  # без явного штрафа Inv_3 у оптимизатора нет причины избегать k_i>9 -- он
  # предпочтёт [10, 7] варианту [9, 8], если это даёт чуть меньше "штрафуемых
  # мастер-шагов" по существующей (доInv_3) модели усталости оператора.
  _D = 17
  _N = 2

  def _run_ga(self, evaluator: ObjectiveEvaluator, space: DecisionSpace, seed: int):
    return classic_genetic_algorithm(
      space, evaluator, pop_size=30, generations=40, random_seed=seed
    )

  def _run_pso(self, evaluator: ObjectiveEvaluator, space: DecisionSpace, seed: int):
    return pso(space, evaluator, swarm_size=30, iterations=50, random_seed=seed)

  def _run_sa(self, evaluator: ObjectiveEvaluator, space: DecisionSpace, seed: int):
    # 300 = runner.py's own default SA iteration budget (see SUITE_STEPS/"SA"
    # in optiflow/optimization/runner.py) -- deliberately NOT bumped, so this
    # test reflects what a user actually gets out of the box.
    return simulated_annealing(space, evaluator, iterations=300, random_seed=seed)

  def test_default_penalty_weight_changes_best_layout_at_realistic_d(self) -> None:
    fields = _mixed_fields_for_miller_tests(self._D, seed=1)
    weights = CriterionWeights.balanced()
    space = DecisionSpace(fields, max_forms=self._N)
    self.assertLessEqual(self._D, MILLER_HARD_LIMIT * self._N, "фикстура должна быть Inv_3-выполнима")

    runners = {"GA": self._run_ga, "PSO": self._run_pso, "SA": self._run_sa}
    seeds = range(5)

    for name, runner in runners.items():
      with self.subTest(algorithm=name):
        unpenalized = ObjectiveEvaluator(FunctionRegistry(), weights, penalty_weight=0.0)
        penalized = ObjectiveEvaluator(
          FunctionRegistry(), weights, penalty_weight=DEFAULT_MILLER_PENALTY_WEIGHT
        )

        unpenalized_max_ks = [
          _max_form_size(runner(unpenalized, space, seed)["best_layout"]) for seed in seeds
        ]
        penalized_max_ks = [
          _max_form_size(runner(penalized, space, seed)["best_layout"]) for seed in seeds
        ]

        violations_without_penalty = sum(1 for k in unpenalized_max_ks if k > MILLER_HARD_LIMIT)
        compliant_with_penalty = sum(1 for k in penalized_max_ks if k <= MILLER_HARD_LIMIT)

        # Baseline: without the Inv_3 penalty, E's own gradient is not enough to
        # keep the optimizer under the hard limit at this D -- if this ever starts
        # failing, the baseline claim in the audit doc ("penalty_weight=0.01 changes
        # behavior, it isn't fighting a baseline that already complies") needs revisiting.
        self.assertGreaterEqual(
          violations_without_penalty, 4,
          f"{name}: unpenalized baseline unexpectedly complies with Inv_3 most of the "
          f"time at D={self._D}, N={self._N} (max_k per seed={unpenalized_max_ks}); "
          "this test's premise (penalty is needed here) may no longer hold.",
        )
        # With the default penalty_weight, compliance should be reliable, not incidental.
        self.assertGreaterEqual(
          compliant_with_penalty, 4,
          f"{name}: DEFAULT_MILLER_PENALTY_WEIGHT={DEFAULT_MILLER_PENALTY_WEIGHT} did not "
          f"reliably enforce Inv_3 at D={self._D}, N={self._N} (max_k per seed="
          f"{penalized_max_ks}); this is evidence the default weight may be too small "
          "for this class of problem sizes, not a formula bug.",
        )


class MillerInfeasibilityTests(unittest.TestCase):
  """Inv_3 открытый вопрос №2 (аудит, врезка раздела 2.3): поведение при D > 9N,
  когда ограничение структурно невыполнимо ни при каком разбиении полей по экранам.
  """

  _D = 30
  _N = 2  # capacity = MILLER_HARD_LIMIT * N = 18 < 30 -> Inv_3 unsatisfiable by construction

  def test_decision_space_construction_never_raises_but_warns(self) -> None:
    """DecisionSpace(fields, max_forms) сам по себе не проверяет и не блокирует
    D > MILLER_HARD_LIMIT * max_forms (конструктор не бросает исключений --
    синтез остаётся запускаемым), но теперь предоставляет space.miller_warning()
    для информирования пользователя до запуска (см. MillerFeasibilityWarningTests
    и run_optimization_suite в optiflow/optimization/runner.py)."""
    self.assertGreater(self._D, MILLER_HARD_LIMIT * self._N, "фикстура должна быть заведомо неразрешимой")
    fields = _mixed_fields_for_miller_tests(self._D, seed=2)
    space = DecisionSpace(fields, max_forms=self._N)  # no exception
    self.assertEqual(space.control_dim(), self._D)
    self.assertIsNotNone(space.miller_warning())

  def test_infeasible_space_degrades_predictably_not_crashes(self) -> None:
    """Inv_1 (полнота полей) остаётся обеспечен репарацией декодера, penalty
    остаётся положительным при ЛЮБОМ валидном разбиении (ожидаемо и корректно --
    не баг), и ни decode_layout, ни scalar_fitness не бросают исключений."""
    fields = _mixed_fields_for_miller_tests(self._D, seed=2)
    weights = CriterionWeights.balanced()
    space = DecisionSpace(fields, max_forms=self._N)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)

    for seed in range(10):
      random.seed(seed)
      vector = space.random_vector()
      layout = space.decode_layout(vector, evaluator.registry)  # must not raise
      total_fields_in_layout = sum(len(form.elements) for form in layout.forms)
      self.assertEqual(total_fields_in_layout, self._D, "Inv_1 must hold even when Inv_3 cannot")

      score = evaluator.scalar_fitness(layout)  # must not raise
      self.assertTrue(math.isfinite(score))
      penalty = compute_miller_penalty(layout, evaluator.penalty_weight)
      self.assertGreater(penalty, 0.0, "D>9N: every valid partition must violate Inv_3 somewhere")

  def test_metaheuristic_still_minimizes_violation_via_balanced_split(self) -> None:
    """Даже когда Inv_3 недостижим, минимум штрафа (суммы квадратов превышений
    при фиксированном D) достигается на максимально равномерном разбиении --
    поэтому осмысленный оптимизатор должен сходиться именно к нему, а не к
    произвольному дисбалансу. D=30 делится на N=2 ровно поровну (15/15), так
    что это однозначная эталонная точка."""
    fields = _mixed_fields_for_miller_tests(self._D, seed=2)
    weights = CriterionWeights.balanced()
    space = DecisionSpace(fields, max_forms=self._N)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)

    result = classic_genetic_algorithm(space, evaluator, pop_size=20, generations=20, random_seed=1)
    layout = result["best_layout"]
    sizes = [len(form.elements) for form in layout.forms]

    self.assertEqual(sum(sizes), self._D)
    # Balanced optimum for D=30, N=2 is exactly [15, 15]; allow a small margin
    # for GA's stochastic search instead of requiring the exact optimum.
    self.assertLessEqual(max(sizes) - min(sizes), 4)


class MillerFeasibilityWarningTests(unittest.TestCase):
  """D > MILLER_HARD_LIMIT * N теперь детектируется и явно сообщается пользователю
  (см. is_miller_feasible/miller_feasibility_margin/miller_feasibility_warning в
  optiflow/optimization/corrections.py, DecisionSpace.miller_warning в algorithms.py,
  и run_optimization_suite в optiflow/optimization/runner.py) -- по тому же принципу
  "предупреждать, не блокировать", что и |Ω| > BRUTE_FORCE_MAX_COMBINATIONS.
  """

  def test_is_miller_feasible_boundary_is_inclusive(self) -> None:
    # D == hard_limit * N is exactly satisfiable (each form gets exactly hard_limit).
    self.assertTrue(is_miller_feasible(9, 1))
    self.assertTrue(is_miller_feasible(18, 2))
    self.assertTrue(is_miller_feasible(27, 3))
    # One field over the boundary tips it into infeasible.
    self.assertFalse(is_miller_feasible(10, 1))
    self.assertFalse(is_miller_feasible(19, 2))
    # Comfortable margin on both sides.
    self.assertTrue(is_miller_feasible(5, 1))
    self.assertFalse(is_miller_feasible(100, 2))

  def test_feasibility_margin_sign_and_magnitude(self) -> None:
    self.assertEqual(miller_feasibility_margin(18, 2), 0)  # exact boundary, no slack
    self.assertEqual(miller_feasibility_margin(16, 2), 2)  # 2 fields of headroom
    self.assertEqual(miller_feasibility_margin(30, 2), -12)  # 12-field deficit

  def test_warning_is_none_when_feasible(self) -> None:
    for d, n in ((9, 1), (18, 2), (5, 3), (1, 1)):
      with self.subTest(d=d, n=n):
        self.assertIsNone(miller_feasibility_warning(d, n))

  def test_warning_carries_concrete_numbers_when_infeasible(self) -> None:
    message = miller_feasibility_warning(30, 2)
    self.assertIsNotNone(message)
    # Concrete numbers must be present, not just a generic sentence: D, N,
    # total capacity (hard_limit*N), and the minimum N needed (ceil(D/hard_limit)).
    self.assertIn("D=30", message)
    self.assertIn("N=2", message)
    self.assertIn("18", message)  # capacity = 9*2
    self.assertIn("4", message)  # ceil(30/9) forms needed

  def test_decision_space_miller_warning_matches_module_function(self) -> None:
    fields = _mixed_fields_for_miller_tests(30, seed=2)
    space = DecisionSpace(fields, max_forms=2)
    self.assertEqual(space.miller_warning(), miller_feasibility_warning(30, 2))

    feasible_fields = _mixed_fields_for_miller_tests(16, seed=2)
    feasible_space = DecisionSpace(feasible_fields, max_forms=2)
    self.assertIsNone(feasible_space.miller_warning())

  def test_run_optimization_suite_surfaces_warning_when_infeasible(self) -> None:
    """Тот же путь, что использует GUI (MainWindow.run_algorithms -> OptimizationWorker
    -> run_optimization_suite -> data["warning"] -> QMessageBox.warning), и headless
    (run_headless_cli печатает то же значение). D=30 полей, все BOOLEAN (единственный
    допустимый control на поле) -- сознательно, чтобы не задевать несвязанный численный
    edge-case в aco() на больших пространствах с TEXT/UNSIGNED полями."""
    fields = [FieldSpec(f"F{i}", DataType.BOOLEAN, 1) for i in range(30)]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=2)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), CriterionWeights.balanced())

    payload = run_optimization_suite(space, evaluator, {})

    self.assertIsNotNone(payload["warning"])
    self.assertIn("D=30", str(payload["warning"]))
    self.assertIn("N=2", str(payload["warning"]))
    # Not blocked: every algorithm in the suite still produced a result.
    self.assertEqual(len(payload["results"]), 10)
    for key, result in payload["results"].items():
      with self.subTest(algorithm=key):
        self.assertIsNotNone(result.get("best_layout"), f"{key} produced no layout")

  def test_run_optimization_suite_has_no_warning_when_feasible(self) -> None:
    fields = [FieldSpec(f"F{i}", DataType.BOOLEAN, 1) for i in range(10)]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=2)  # capacity 18 >= 10
    evaluator = ObjectiveEvaluator(FunctionRegistry(), CriterionWeights.balanced())

    payload = run_optimization_suite(space, evaluator, {})

    self.assertIsNone(payload["warning"])

  def test_miller_and_brute_force_warnings_combine_without_clobbering(self) -> None:
    """Оба предупреждения используют одно и то же поле warning/виджет -- если
    триггерятся оба условия (Inv_3 недостижим И |Ω| > BRUTE_FORCE_MAX_COMBINATIONS),
    ни одно не должно "потеряться". brute_force замокан, чтобы не зависеть от
    конкретного числа комбинаций и не задевать несвязанный edge-case в aco()."""
    from unittest.mock import patch

    fields = [FieldSpec(f"F{i}", DataType.BOOLEAN, 1) for i in range(30)]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=2)  # Miller-infeasible: 30 > 9*2
    evaluator = ObjectiveEvaluator(FunctionRegistry(), CriterionWeights.balanced())

    with patch(
      "optiflow.optimization.runner.brute_force",
      side_effect=ValueError("Brute force search space (999999 combinations) exceeds limit"),
    ):
      payload = run_optimization_suite(space, evaluator, {})

    warning = str(payload["warning"])
    self.assertIn("D=30", warning)  # Miller warning present
    self.assertIn("999999", warning)  # brute_force warning present too
    self.assertIsNone(payload["results"]["BruteForce"]["best_layout"])


class FitnessDisplayClippingTests(unittest.TestCase):
  """F, показываемый/экспортируемый пользователю (GUI, run_results.py, JSON),
  должен лежать в [0,1] -- см. clip_fitness_for_display в algorithms.py. Внутренний
  скаляр, которым все 10 алгоритмов сравнивают решения (ObjectiveEvaluator.
  scalar_fitness и прямые вызовы calculate_fitness в algorithms.py), НЕ
  клиппингуется и по-прежнему может уходить в отрицательную область при
  структурно невыполнимом Inv_3 (D>9N) -- это подтверждают неизменные
  RealisticMillerConvergenceTests и MillerInfeasibilityTests.
  """

  def test_clip_fitness_for_display_bounds(self) -> None:
    self.assertEqual(clip_fitness_for_display(-0.72), 0.0)
    self.assertEqual(clip_fitness_for_display(-1e-9), 0.0)
    self.assertEqual(clip_fitness_for_display(1.5), 1.0)
    self.assertEqual(clip_fitness_for_display(1.0 + 1e-9), 1.0)
    for value in (0.0, 0.25, 0.5, 0.999, 1.0):
      with self.subTest(value=value):
        self.assertAlmostEqual(clip_fitness_for_display(value), value)

  def test_build_algorithm_summaries_clips_fitness_but_keeps_raw_at_d_gt_9n(self) -> None:
    fields = [FieldSpec(f"F{i}", DataType.BOOLEAN, 1) for i in range(30)]
    ensure_allowed_controls(fields)
    weights = CriterionWeights.balanced()
    space = DecisionSpace(fields, max_forms=2)  # D=30 > MILLER_HARD_LIMIT*N=18
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)
    result = classic_genetic_algorithm(space, evaluator, pop_size=20, generations=15, random_seed=1)

    summaries = build_algorithm_summaries({"GA": result}, FunctionRegistry(), weights)
    ga_summary = next(s for s in summaries if s.key == "GA")

    self.assertIsNotNone(ga_summary.layout)
    self.assertLess(ga_summary.raw_fitness, 0.0, "D>9N: unclipped F should be negative here")
    self.assertGreaterEqual(ga_summary.fitness, 0.0)
    self.assertLessEqual(ga_summary.fitness, 1.0)
    self.assertEqual(ga_summary.fitness, 0.0)  # raw < 0 -> clipped to the lower bound

  def test_build_algorithm_summaries_raw_equals_fitness_when_feasible(self) -> None:
    fields = [FieldSpec(f"F{i}", DataType.BOOLEAN, 1) for i in range(5)]
    ensure_allowed_controls(fields)
    weights = CriterionWeights.balanced()
    space = DecisionSpace(fields, max_forms=2)  # D=5 <= 9*2, comfortably feasible
    evaluator = ObjectiveEvaluator(FunctionRegistry(), weights)
    result = classic_genetic_algorithm(space, evaluator, pop_size=10, generations=10, random_seed=1)

    summaries = build_algorithm_summaries({"GA": result}, FunctionRegistry(), weights)
    ga_summary = next(s for s in summaries if s.key == "GA")

    self.assertGreaterEqual(ga_summary.raw_fitness, 0.0)
    # No clipping should occur when raw_fitness is already within [0, 1].
    self.assertAlmostEqual(ga_summary.fitness, ga_summary.raw_fitness, places=9)

  def test_interface_layout_payload_stores_both_fitness_fields(self) -> None:
    layout = build_interface_layout_from_partition(
      [FieldSpec("A", DataType.BOOLEAN, 1)], [ControlType.CHECKBOX], [1],
    )
    payload = interface_layout_to_payload(
      layout, optiflow_version="test", fitness=0.0, raw_fitness=-0.72,
    )
    self.assertEqual(payload["metrics"]["fitness"], 0.0)
    self.assertEqual(payload["metrics"]["raw_fitness"], -0.72)

  def test_internal_comparison_path_is_never_clipped(self) -> None:
    """Sanity guard for the exclusion in this task: ObjectiveEvaluator.scalar_fitness
    -- the function every metaheuristic uses to compare/select solutions -- must still
    be able to return a value outside [0,1]. If this assertion ever fails, clipping has
    leaked into the internal search path, which would silently break the validated
    property in RealisticMillerConvergenceTests/MillerInfeasibilityTests (that the
    optimizer keeps a meaningful gradient among already-infeasible solutions)."""
    fields = [FieldSpec(f"F{i}", DataType.BOOLEAN, 1) for i in range(30)]
    ensure_allowed_controls(fields)
    space = DecisionSpace(fields, max_forms=2)
    evaluator = ObjectiveEvaluator(FunctionRegistry(), CriterionWeights.balanced())
    layout = space.decode_layout(space.random_vector(), evaluator.registry)
    self.assertLess(evaluator.scalar_fitness(layout), 0.0)

  def test_progress_overlay_metrics_text_is_clipped_with_raw_annotation(self) -> None:
    """ProgressOverlay.apply_report (live "F = ..." during a run) was missed in the
    first clipping pass -- it reads ProgressReport.best_fitness directly, bypassing
    AlgorithmRunSummary entirely. format_progress_metrics_text is the pure function
    it now delegates to, importable without a QApplication."""
    from optiflow.app import format_progress_metrics_text
    from optiflow.optimization.algorithms import ProgressReport

    infeasible_report = ProgressReport(
      algorithm="GA", algorithm_index=0, algorithm_count=1,
      iteration=1, max_iterations=10, best_fitness=-0.72,
      potency=0.5, operativeness=0.5, resource_saving=0.5, overall_fraction=0.1,
    )
    text = format_progress_metrics_text(infeasible_report)
    self.assertIn("F = 0.0000", text)
    self.assertIn("-0.7200", text)

    feasible_report = ProgressReport(
      algorithm="GA", algorithm_index=0, algorithm_count=1,
      iteration=1, max_iterations=10, best_fitness=0.42,
      potency=0.5, operativeness=0.5, resource_saving=0.5, overall_fraction=0.1,
    )
    text2 = format_progress_metrics_text(feasible_report)
    self.assertIn("F = 0.4200", text2)
    self.assertNotIn("ниж. предел", text2)


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
