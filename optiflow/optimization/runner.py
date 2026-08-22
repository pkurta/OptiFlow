from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from optiflow.optimization.algorithms import (
  DecisionSpace,
  ObjectiveEvaluator,
  OptimizationControl,
  aco,
  brute_force,
  classic_genetic_algorithm,
  greedy,
  hill_climb,
  nsga2,
  pso,
  random_search,
  simulated_annealing,
  tabu_search,
)

SUITE_STEPS: Tuple[Tuple[str, str], ...] = (
  ("NSGA-II", "Многокритериальный (NSGA-II)"),
  ("BruteForce", "Полный перебор (Brute Force)"),
  ("GA", "Классический генетический алгоритм (GA)"),
  ("HillClimb", "Локальный поиск (Hill Climb)"),
  ("PSO", "Алгоритм роя частиц (PSO)"),
  ("Random", "Случайный поиск"),
  ("Greedy", "Жадный (Greedy)"),
  ("SA", "Имитация отжига (SA)"),
  ("Tabu", "Поиск с запретами (Tabu)"),
  ("ACO", "Муравьиный алгоритм (ACO)"),
)


def run_optimization_suite(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  params_by_label: Dict[str, Dict[str, float]],
  control: Optional[OptimizationControl] = None,
) -> Dict[str, object]:
  """Run the full algorithm suite. Safe to call from a worker thread."""
  results: Dict[str, Dict[str, object]] = {}
  warning: Optional[str] = None
  cancelled = False
  total = len(SUITE_STEPS)

  for index, (key, label) in enumerate(SUITE_STEPS):
    if control is not None and control.cancelled:
      cancelled = True
      break
    if control is not None:
      control.begin_algorithm(label, index, total)
    params = params_by_label.get(label, {})
    if key == "NSGA-II":
      results[key] = nsga2(
        space,
        evaluator,
        pop_size=int(params.get("pop_size", 40)),
        generations=int(params.get("generations", 40)),
        crossover_prob=float(params.get("crossover_prob", 0.9)),
        mutation_prob=float(params.get("mutation_prob", 0.2)),
        mutation_sigma=float(params.get("mutation_sigma", 0.5)),
        control=control,
      )
    elif key == "BruteForce":
      try:
        results[key] = brute_force(space, evaluator, control=control)
      except ValueError as exc:
        results[key] = {"best_score": 0.0, "history": [], "best_layout": None}
        warning = str(exc)
    elif key == "GA":
      results[key] = classic_genetic_algorithm(
        space,
        evaluator,
        pop_size=int(params.get("pop_size", 30)),
        generations=int(params.get("generations", 30)),
        control=control,
      )
    elif key == "HillClimb":
      results[key] = hill_climb(
        space, evaluator, iterations=int(params.get("iterations", 200)), control=control
      )
    elif key == "PSO":
      results[key] = pso(
        space,
        evaluator,
        swarm_size=int(params.get("swarm_size", 30)),
        iterations=int(params.get("iterations", 50)),
        inertia=float(params.get("inertia", 0.7)),
        cognitive=float(params.get("cognitive", 1.5)),
        social=float(params.get("social", 1.5)),
        control=control,
      )
    elif key == "Random":
      results[key] = random_search(
        space, evaluator, iterations=int(params.get("iterations", 200)), control=control
      )
    elif key == "Greedy":
      results[key] = greedy(space, evaluator, control=control)
    elif key == "SA":
      results[key] = simulated_annealing(
        space,
        evaluator,
        iterations=int(params.get("iterations", 300)),
        initial_temp=float(params.get("initial_temp", 5.0)),
        cooling=float(params.get("cooling", 0.97)),
        control=control,
      )
    elif key == "Tabu":
      results[key] = tabu_search(
        space,
        evaluator,
        iterations=int(params.get("iterations", 200)),
        tabu_tenure=int(params.get("tabu_tenure", 7)),
        control=control,
      )
    elif key == "ACO":
      results[key] = aco(
        space,
        evaluator,
        ants=int(params.get("ants", 20)),
        iterations=int(params.get("iterations", 40)),
        alpha=float(params.get("alpha", 1.0)),
        beta=float(params.get("beta", 2.0)),
        evaporation=float(params.get("evaporation", 0.1)),
        deposit_weight=float(params.get("deposit_weight", 1.0)),
        control=control,
      )
    if control is not None and control.cancelled:
      cancelled = True
      break

  return {"results": results, "cancelled": cancelled, "warning": warning}


def histories_from_results(
  results: Dict[str, Dict[str, object]],
) -> List[Tuple[str, List[float]]]:
  order = (
    "NSGA-II",
    "BruteForce",
    "GA",
    "HillClimb",
    "PSO",
    "Random",
    "Greedy",
    "SA",
    "Tabu",
    "ACO",
  )
  histories: List[Tuple[str, List[float]]] = []
  for key in order:
    payload = results.get(key)
    if not payload:
      continue
    history = payload.get("history") or []
    if key == "BruteForce" and not history:
      continue
    histories.append((key, history))  # type: ignore[arg-type]
  return histories
