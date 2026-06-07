from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from optiflow.models.scoring import DataType, FieldSpec, FunctionRegistry
from optiflow.optimization.algorithms import (
  DecisionSpace,
  ObjectiveEvaluator,
  TargetProfile,
  aco,
  brute_force,
  brute_force_search_space_size,
  classic_genetic_algorithm,
  compute_total_efficiency,
  greedy,
  nsga2,
  profile_constraints_satisfied,
  profile_from_slider_values,
  pso,
  simulated_annealing,
  tabu_search,
)

logger = logging.getLogger(__name__)

BENCHMARK_ALGORITHMS: Tuple[str, ...] = (
  "BruteForce",
  "NSGA-II",
  "GA",
  "PSO",
  "Greedy",
  "SA",
  "Tabu",
  "ACO",
)


def ensure_allowed_controls(fields: List[FieldSpec]) -> None:
  """Attach ``allowed_controls()`` to FieldSpec instances (mirrors app.py)."""
  from optiflow.models.scoring import ControlType

  def allowed_for(field: FieldSpec) -> List[ControlType]:
    if field.data_type == DataType.BOOLEAN:
      return [ControlType.CHECKBOX]
    if field.data_type == DataType.UNSIGNED:
      return [ControlType.SPINNER, ControlType.SLIDER]
    return [ControlType.TEXTBOX, ControlType.DROPDOWNLIST]

  for f in fields:
    if hasattr(f, "allowed_controls"):
      continue
    allowed = allowed_for(f)
    setattr(f, "allowed_controls", lambda allowed=allowed: allowed)


def random_benchmark_snapshot(
  rng: random.Random,
  *,
  min_fields: int = 2,
  max_fields: int = 4,
  max_forms_cap: int = 3,
) -> Tuple[List[FieldSpec], int, TargetProfile]:
  """Monte Carlo draw: random fields, wizard depth, and TargetProfile sliders."""
  n_fields = rng.randint(min_fields, max_fields)
  fields: List[FieldSpec] = []
  for i in range(n_fields):
    dtype = rng.choice(list(DataType))
    size = rng.randint(1, 24 if dtype == DataType.TEXT else 6)
    fields.append(FieldSpec(name=f"Field{i + 1}", data_type=dtype, size=size))
  ensure_allowed_controls(fields)
  max_forms = rng.randint(1, min(max_forms_cap, max(1, n_fields)))
  sliders = [rng.randint(0, 100) for _ in range(3)]
  profile = profile_from_slider_values(*sliders)
  return fields, max_forms, profile


def convergence_plateau_iteration(
  history: Sequence[float],
  *,
  epsilon: float = 1e-6,
  patience: int = 3,
) -> int:
  """Index in ``history`` when best score stops improving (convergence speed proxy)."""
  if not history:
    return 0
  best = float(history[0])
  plateau_at = 0
  stale = 0
  for i, value in enumerate(history):
    if value > best + epsilon:
      best = float(value)
      plateau_at = i
      stale = 0
    else:
      stale += 1
      if stale >= patience:
        return plateau_at
  return max(0, len(history) - 1)


def precision_vs_baseline(algorithm_score: float, baseline_score: float) -> Optional[float]:
  if baseline_score <= 0.0:
    return None
  return min(1.0, max(0.0, algorithm_score / baseline_score))


@dataclass
class AlgorithmBenchmarkStats:
  precision_rates: List[float] = field(default_factory=list)
  convergence_iterations: List[int] = field(default_factory=list)
  constraint_passes: int = 0
  constraint_total: int = 0
  baseline_runs: int = 0

  def mean_precision(self) -> Optional[float]:
    if not self.precision_rates:
      return None
    return sum(self.precision_rates) / len(self.precision_rates)

  def mean_convergence(self) -> Optional[float]:
    if not self.convergence_iterations:
      return None
    return sum(self.convergence_iterations) / len(self.convergence_iterations)

  def constraint_pass_rate(self) -> Optional[float]:
    if self.constraint_total == 0:
      return None
    return self.constraint_passes / self.constraint_total


def _run_algorithm(
  name: str,
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  rng: random.Random,
) -> Dict[str, object]:
  seed = rng.randint(0, 2**31 - 1)
  runners: Dict[str, Callable[[], Dict[str, object]]] = {
    "BruteForce": lambda: brute_force(space, evaluator),
    "NSGA-II": lambda: nsga2(space, evaluator, pop_size=24, generations=20, random_seed=seed),
    "GA": lambda: classic_genetic_algorithm(
      space, evaluator, pop_size=24, generations=20, random_seed=seed
    ),
    "PSO": lambda: pso(space, evaluator, swarm_size=20, iterations=30, random_seed=seed),
    "Greedy": lambda: greedy(space, evaluator),
    "SA": lambda: simulated_annealing(space, evaluator, iterations=120, random_seed=seed),
    "Tabu": lambda: tabu_search(space, evaluator, iterations=80, random_seed=seed),
    "ACO": lambda: aco(space, evaluator, ants=12, iterations=25, random_seed=seed),
  }
  return runners[name]()


def format_benchmark_markdown(
  stats_by_algorithm: Dict[str, AlgorithmBenchmarkStats],
  *,
  runs_count: int,
  baseline_computable_runs: int,
) -> str:
  lines = [
    "# OptiFlow Optimization Benchmark",
    "",
    f"Monte Carlo runs: **{runs_count}** | Brute-force baseline computable: **{baseline_computable_runs}**",
    "",
    "| Algorithm | Precision Rate | Convergence (iter) | Constraint Pass Rate | Baseline Samples |",
    "| --- | ---: | ---: | ---: | ---: |",
  ]
  for name in BENCHMARK_ALGORITHMS:
    stats = stats_by_algorithm[name]
    precision = stats.mean_precision()
    convergence = stats.mean_convergence()
    constraint = stats.constraint_pass_rate()
    lines.append(
      "| {name} | {precision} | {convergence} | {constraint} | {baseline} |".format(
        name=name,
        precision=f"{precision:.4f}" if precision is not None else "n/a",
        convergence=f"{convergence:.1f}" if convergence is not None else "n/a",
        constraint=f"{constraint:.2%}" if constraint is not None else "n/a",
        baseline=stats.baseline_runs if name != "BruteForce" else baseline_computable_runs,
      )
    )
  lines.append("")
  return "\n".join(lines)


def run_optimization_benchmark(
  runs_count: int = 100,
  *,
  random_seed: int = 42,
  output_dir: Path | str | None = None,
  log_markdown: bool = True,
) -> Dict[str, AlgorithmBenchmarkStats]:
  """
  Monte Carlo quality validation across all metaheuristics on identical snapshots.

  Writes ``benchmark_report.md`` next to ``wizard_output.html`` when ``output_dir`` is set.
  """
  rng = random.Random(random_seed)
  registry = FunctionRegistry()
  stats_by_algorithm: Dict[str, AlgorithmBenchmarkStats] = {
    name: AlgorithmBenchmarkStats() for name in BENCHMARK_ALGORITHMS
  }
  baseline_computable_runs = 0

  for run_idx in range(runs_count):
    fields, max_forms, profile = random_benchmark_snapshot(rng)
    space = DecisionSpace(fields, max_forms=max_forms)
    evaluator = ObjectiveEvaluator(registry, profile)
    search_size = brute_force_search_space_size(space)
    baseline_score: Optional[float] = None
    bf_ran = False

    if search_size <= 50_000:
      try:
        bf_result = _run_algorithm("BruteForce", space, evaluator, rng)
        baseline_score = float(bf_result["best_score"])
        baseline_computable_runs += 1
        bf_ran = True
        stats_by_algorithm["BruteForce"].precision_rates.append(1.0)
        stats_by_algorithm["BruteForce"].convergence_iterations.append(
          convergence_plateau_iteration(bf_result["history"])
        )
        bf_layout = bf_result.get("best_layout")
        if bf_layout is not None:
          triple = compute_total_efficiency(bf_layout, registry)
          stats_by_algorithm["BruteForce"].constraint_total += 1
          if profile_constraints_satisfied(triple, profile):
            stats_by_algorithm["BruteForce"].constraint_passes += 1
      except ValueError:
        baseline_score = None

    for name in BENCHMARK_ALGORITHMS:
      if name == "BruteForce" and bf_ran:
        continue

      result = _run_algorithm(name, space, evaluator, rng)
      score = float(result["best_score"])
      history = result.get("history") or [score]
      stats_by_algorithm[name].convergence_iterations.append(convergence_plateau_iteration(history))

      layout = result.get("best_layout")
      if layout is not None:
        triple = compute_total_efficiency(layout, registry)
        stats_by_algorithm[name].constraint_total += 1
        if profile_constraints_satisfied(triple, profile):
          stats_by_algorithm[name].constraint_passes += 1

      if baseline_score is not None and baseline_score > 0.0:
        prec = precision_vs_baseline(score, baseline_score)
        if prec is not None:
          stats_by_algorithm[name].precision_rates.append(prec)
          stats_by_algorithm[name].baseline_runs += 1

    if (run_idx + 1) % max(1, runs_count // 10) == 0:
      logger.info("Benchmark progress: %s / %s runs", run_idx + 1, runs_count)

  markdown = format_benchmark_markdown(
    stats_by_algorithm,
    runs_count=runs_count,
    baseline_computable_runs=baseline_computable_runs,
  )
  if log_markdown:
    logger.info("\n%s", markdown)

  if output_dir is not None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote benchmark report -> %s", report_path.resolve())

  return stats_by_algorithm
