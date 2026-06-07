from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from optiflow.models.scoring import (
  ControlType,
  EfficiencyTriple,
  FieldSpec,
  FunctionRegistry,
  InterfaceLayout,
  build_interface_layout,
  build_interface_layout_from_partition,
  evaluate_form,
  partition_counts_to_form_indices,
)
from optiflow.optimization.corrections import (
  apply_element_position_correction,
  apply_form_step_correction,
)


def clamp(value: float, low: float, high: float) -> float:
  return max(low, min(high, value))


class TargetMode(Enum):
  Certain = auto()
  Max = auto()
  Any = auto()


@dataclass
class ComponentTarget:
  mode: TargetMode = TargetMode.Any
  value: float = 0.0


@dataclass
class TargetProfile:
  potency: ComponentTarget = field(default_factory=ComponentTarget)
  operativeness: ComponentTarget = field(default_factory=ComponentTarget)
  resource_saving: ComponentTarget = field(default_factory=ComponentTarget)

  @classmethod
  def balanced(cls) -> TargetProfile:
    return cls(
      potency=ComponentTarget(TargetMode.Any),
      operativeness=ComponentTarget(TargetMode.Any),
      resource_saving=ComponentTarget(TargetMode.Any),
    )

  @classmethod
  def maximize_potency(cls) -> TargetProfile:
    return cls(
      potency=ComponentTarget(TargetMode.Max),
      operativeness=ComponentTarget(TargetMode.Any),
      resource_saving=ComponentTarget(TargetMode.Any),
    )


def compute_total_efficiency(layout: InterfaceLayout, registry: FunctionRegistry) -> EfficiencyTriple:
  """
  Equation (1): Total = ∏(corrected forms) × ∏(double-corrected elements).
  """
  total = EfficiencyTriple.identity()

  for form in layout.forms:
    i = form.form_index
    form_atomic = evaluate_form(len(form.elements))
    form_corrected = apply_form_step_correction(form_atomic, i)
    total = total * form_corrected

    for element in form.elements:
      field = layout.fields[element.field_index]
      atomic = registry.evaluate_atomic(element.control, field.data_type, field.size)
      with_position = apply_element_position_correction(atomic, element.position_index)
      double_corrected = apply_form_step_correction(with_position, i)
      total = total * double_corrected

  return total


def triple_to_objective_tuple(triple: EfficiencyTriple) -> Tuple[float, float, float]:
  return triple.as_tuple()


def select_from_pareto(
  objectives: List[EfficiencyTriple],
  profile: TargetProfile,
) -> int:
  if not objectives:
    return 0

  indexes = list(range(len(objectives)))

  def filter_by_component(
    candidates: List[int],
    getter: Callable[[EfficiencyTriple], float],
    target: ComponentTarget,
  ) -> List[int]:
    if target.mode == TargetMode.Any or not candidates:
      return candidates
    if target.mode == TargetMode.Max:
      best_val = max(getter(objectives[i]) for i in candidates)
      return [i for i in candidates if getter(objectives[i]) >= best_val - 1e-12]
    desired = clamp(target.value, 0.0, 1.0)
    return [min(candidates, key=lambda i: abs(getter(objectives[i]) - desired))]

  indexes = filter_by_component(indexes, lambda t: t.potency, profile.potency)
  indexes = filter_by_component(indexes, lambda t: t.operativeness, profile.operativeness)
  indexes = filter_by_component(indexes, lambda t: t.resource_saving, profile.resource_saving)

  if len(indexes) == 1:
    return indexes[0]

  return max(
    indexes,
    key=lambda i: scalar_fitness_from_triple(objectives[i], profile),
  )


def profile_from_slider_values(v1: int, v2: int, v3: int) -> TargetProfile:
  """Map raw slider positions (0–100) to TargetProfile modes; 100 → Max for that dimension."""

  def component(raw: int) -> ComponentTarget:
    v = int(clamp(float(raw), 0.0, 100.0))
    if v >= 100:
      return ComponentTarget(TargetMode.Max, 1.0)
    if v > 0:
      return ComponentTarget(TargetMode.Certain, v / 100.0)
    return ComponentTarget(TargetMode.Any, 0.0)

  return TargetProfile(
    potency=component(v1),
    operativeness=component(v2),
    resource_saving=component(v3),
  )


def profile_from_slider_weights(w1: float, w2: float, w3: float) -> TargetProfile:
  """Backward-compatible wrapper: normalized weights mapped to 0–100 slider semantics."""
  return profile_from_slider_values(
    int(round(clamp(w1, 0.0, 1.0) * 100)),
    int(round(clamp(w2, 0.0, 1.0) * 100)),
    int(round(clamp(w3, 0.0, 1.0) * 100)),
  )


def scalar_fitness_from_triple(
  triple: EfficiencyTriple,
  profile: TargetProfile,
  weights: Optional[Sequence[float]] = None,
) -> float:
  """
  Kurta-Izrailov scalar fitness for single-objective metaheuristics.

  Max mode: strict per-component isolation (never the triple product).
  Certain / Any mode: weighted linear projection; optional ``weights`` override
  profile-derived coefficients as (w_P, w_O, w_R).
  """
  max_values: List[float] = []
  if profile.potency.mode == TargetMode.Max:
    max_values.append(triple.potency)
  if profile.operativeness.mode == TargetMode.Max:
    max_values.append(triple.operativeness)
  if profile.resource_saving.mode == TargetMode.Max:
    max_values.append(triple.resource_saving)

  if len(max_values) == 1:
    return max_values[0]
  if len(max_values) > 1:
    return min(max_values)

  if weights is not None and len(weights) >= 3:
    w_p, w_o, w_r = (max(0.0, float(weights[0])), max(0.0, float(weights[1])), max(0.0, float(weights[2])))
    total_w = w_p + w_o + w_r
    if total_w <= 0.0:
      return (triple.potency + triple.operativeness + triple.resource_saving) / 3.0
    return (
      w_p * triple.potency + w_o * triple.operativeness + w_r * triple.resource_saving
    ) / total_w

  coeff: List[float] = []
  values: List[float] = []
  for target, value in (
    (profile.potency, triple.potency),
    (profile.operativeness, triple.operativeness),
    (profile.resource_saving, triple.resource_saving),
  ):
    if target.mode == TargetMode.Certain:
      coeff.append(max(1e-9, clamp(target.value, 0.0, 1.0)))
      values.append(value)
    elif target.mode == TargetMode.Any:
      coeff.append(1.0)
      values.append(value)

  if not coeff:
    return (triple.potency + triple.operativeness + triple.resource_saving) / 3.0

  total_w = sum(coeff)
  return sum(w * v for w, v in zip(coeff, values)) / total_w


def profile_constraints_satisfied(triple: EfficiencyTriple, profile: TargetProfile, tol: float = 1e-6) -> bool:
  """Return True when Certain-mode targets are met within tolerance."""
  if profile.potency.mode == TargetMode.Certain:
    if abs(triple.potency - clamp(profile.potency.value, 0.0, 1.0)) > tol:
      return False
  if profile.operativeness.mode == TargetMode.Certain:
    if abs(triple.operativeness - clamp(profile.operativeness.value, 0.0, 1.0)) > tol:
      return False
  if profile.resource_saving.mode == TargetMode.Certain:
    if abs(triple.resource_saving - clamp(profile.resource_saving.value, 0.0, 1.0)) > tol:
      return False
  return True


@dataclass
class DecisionSpace:
  """
  Genome structure (Kurta-Izrailov parametric synthesis):
    Part 1 — length D: control type index per field (preserves field order).
    Part 2 — length N: partition weights → field counts per sequential wizard form.
  """
  fields: List[FieldSpec]
  max_forms: int = 5

  def control_dim(self) -> int:
    return len(self.fields)

  def partition_dim(self) -> int:
    return max(1, self.max_forms)

  def dimension(self) -> int:
    return self.control_dim() + self.partition_dim()

  def cardinalities(self) -> List[int]:
    return [len(f.allowed_controls()) for f in self.fields]

  def _partition_counts_from_genome(self, x: np.ndarray) -> List[int]:
    """Decode Part 2 into integer counts per form that sum to D."""
    d = self.control_dim()
    n = self.partition_dim()
    if n <= 1:
      return [d]
    start = d
    weights = np.array(
      [max(0.0, float(x[start + i])) for i in range(n)],
      dtype=float,
    )
    if float(weights.sum()) <= 0.0:
      weights = np.ones(n, dtype=float)
    exact = weights / weights.sum() * d
    counts = [int(math.floor(v)) for v in exact]
    remainder = d - sum(counts)
    if remainder > 0:
      fractional = sorted(
        ((exact[i] - counts[i], i) for i in range(n)),
        reverse=True,
      )
      for k in range(remainder):
        counts[fractional[k % n][1]] += 1
    return counts

  def counts_to_form_indices(self, counts: List[int]) -> List[int]:
    return partition_counts_to_form_indices(counts)

  def round_to_valid(self, x: np.ndarray) -> Tuple[List[int], List[int]]:
    control_rounded: List[int] = []
    for i, f in enumerate(self.fields):
      k = len(f.allowed_controls())
      xi = int(round(clamp(float(x[i]), 0, max(0, k - 1))))
      control_rounded.append(xi)
    partition_counts = self._partition_counts_from_genome(x)
    return control_rounded, partition_counts

  def random_vector(self) -> np.ndarray:
    d = self.control_dim()
    controls = [random.uniform(0, len(f.allowed_controls()) - 1) for f in self.fields]
    partition = [random.uniform(0.1, 1.0) for _ in range(self.partition_dim())]
    return np.array(controls + partition, dtype=float)

  def decode_layout(self, x: np.ndarray, registry: FunctionRegistry) -> InterfaceLayout:
    del registry  # layout decode is registry-independent; kept for a uniform API
    control_idx, partition_counts = self.round_to_valid(x)
    controls = self.all_controls(control_idx)
    return build_interface_layout_from_partition(self.fields, controls, partition_counts)

  def all_controls(self, discrete_indexes: List[int]) -> List[ControlType]:
    result: List[ControlType] = []
    for idx, f in zip(discrete_indexes, self.fields):
      allowed = f.allowed_controls()
      result.append(allowed[int(idx)])
    return result


class ObjectiveEvaluator:
  def __init__(self, registry: FunctionRegistry, profile: TargetProfile) -> None:
    self.registry = registry
    self.profile = profile

  def evaluate_layout(self, layout: InterfaceLayout) -> EfficiencyTriple:
    return compute_total_efficiency(layout, self.registry)

  def multi_objective(self, layout: InterfaceLayout) -> Tuple[float, float, float]:
    return triple_to_objective_tuple(self.evaluate_layout(layout))

  def scalar_fitness(self, layout: InterfaceLayout) -> float:
    triple = self.evaluate_layout(layout)
    return scalar_fitness_from_triple(triple, self.profile)

  def evaluate_vector(self, x: np.ndarray, space: DecisionSpace) -> EfficiencyTriple:
    layout = space.decode_layout(x, self.registry)
    return self.evaluate_layout(layout)


def non_dominated_sort(points: List[Tuple[float, float, float]]) -> List[List[int]]:
  n_pop = len(points)
  s_sets = [set() for _ in range(n_pop)]
  domination_count = [0] * n_pop
  ranks: List[Optional[int]] = [None] * n_pop
  fronts: List[List[int]] = []

  def dominates(p: Tuple[float, float, float], q: Tuple[float, float, float]) -> bool:
    return all(p[i] >= q[i] for i in range(3)) and any(p[i] > q[i] for i in range(3))

  for p in range(n_pop):
    for q in range(n_pop):
      if p == q:
        continue
      if dominates(points[p], points[q]):
        s_sets[p].add(q)
      elif dominates(points[q], points[p]):
        domination_count[p] += 1
    if domination_count[p] == 0:
      ranks[p] = 0
  front0 = [i for i in range(n_pop) if ranks[i] == 0]
  if front0:
    fronts.append(front0)

  i = 0
  while i < len(fronts):
    next_front: List[int] = []
    for p in fronts[i]:
      for q in s_sets[p]:
        domination_count[q] -= 1
        if domination_count[q] == 0:
          ranks[q] = i + 1
          next_front.append(q)
    if next_front:
      fronts.append(next_front)
    i += 1
  return fronts


def crowding_distance(
  front_points: List[Tuple[float, float, float]],
  front_indexes: List[int],
) -> Dict[int, float]:
  m = 3
  length = len(front_indexes)
  if length == 0:
    return {}
  distance: Dict[int, float] = {idx: 0.0 for idx in front_indexes}
  f_arr = np.array(front_points, dtype=float)
  for dim in range(m):
    order = np.argsort(f_arr[:, dim])
    min_m = f_arr[order[0], dim]
    max_m = f_arr[order[-1], dim]
    distance[front_indexes[order[0]]] = float("inf")
    distance[front_indexes[order[-1]]] = float("inf")
    span = max(1e-9, max_m - min_m)
    for k in range(1, length - 1):
      prev_val = f_arr[order[k - 1], dim]
      next_val = f_arr[order[k + 1], dim]
      distance[front_indexes[order[k]]] += (next_val - prev_val) / span
  return distance


def nsga2(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  pop_size: int = 40,
  generations: int = 40,
  crossover_prob: float = 0.9,
  mutation_prob: float = 0.2,
  mutation_sigma: float = 0.5,
  random_seed: int | None = None,
):
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  d = space.dimension()

  def evaluate_population(pop: List[np.ndarray]) -> List[Tuple[float, float, float]]:
    return [evaluator.multi_objective(space.decode_layout(x, evaluator.registry)) for x in pop]

  population = [space.random_vector() for _ in range(pop_size)]
  objectives = evaluate_population(population)
  history_best_scalar: List[float] = []
  history_best_triple: List[EfficiencyTriple] = []
  best_layout: Optional[InterfaceLayout] = None

  for _g in range(generations):
    offspring: List[np.ndarray] = []
    while len(offspring) < pop_size:

      def tournament_select(k: int = 2) -> np.ndarray:
        candidates = random.sample(range(pop_size), k)
        fronts = non_dominated_sort([objectives[i] for i in candidates])
        first_front = fronts[0]
        if len(first_front) == 1:
          return population[candidates[first_front[0]]].copy()
        cd = crowding_distance(
          [objectives[candidates[i]] for i in first_front],
          list(first_front),
        )
        best_index = max(first_front, key=lambda i: cd.get(i, 0.0))
        return population[candidates[best_index]].copy()

      p1 = tournament_select()
      p2 = tournament_select()
      c1 = p1.copy()
      c2 = p2.copy()
      if random.random() < crossover_prob:
        alpha = np.random.uniform(0.0, 1.0, size=d)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
      if random.random() < mutation_prob:
        c1 = c1 + np.random.normal(0, mutation_sigma, size=d)
      if random.random() < mutation_prob:
        c2 = c2 + np.random.normal(0, mutation_sigma, size=d)
      offspring.append(c1)
      if len(offspring) < pop_size:
        offspring.append(c2)

    combined = population + offspring
    combined_objectives = evaluate_population(combined)
    fronts = non_dominated_sort(combined_objectives)
    new_population: List[np.ndarray] = []
    new_objectives: List[Tuple[float, float, float]] = []
    for front in fronts:
      if len(new_population) + len(front) <= pop_size:
        for i in front:
          new_population.append(combined[i])
          new_objectives.append(combined_objectives[i])
      else:
        front_points = [combined_objectives[i] for i in front]
        cd = crowding_distance(front_points, front)
        order = sorted(front, key=lambda i: cd.get(i, 0.0), reverse=True)
        slots = pop_size - len(new_population)
        for i in order[:slots]:
          new_population.append(combined[i])
          new_objectives.append(combined_objectives[i])
        break
    population = new_population
    objectives = new_objectives

    front0 = non_dominated_sort(objectives)[0] if objectives else [0]
    triples = [
      EfficiencyTriple(*objectives[i])
      for i in front0
    ]
    pick = select_from_pareto(triples, evaluator.profile)
    chosen_index = front0[pick]
    best_layout = space.decode_layout(population[chosen_index], evaluator.registry)
    best_triple = EfficiencyTriple(*objectives[chosen_index])
    history_best_scalar.append(
      scalar_fitness_from_triple(best_triple, evaluator.profile)
    )
    history_best_triple.append(best_triple)

  if best_layout is None and population:
    best_layout = space.decode_layout(population[0], evaluator.registry)

  return {
    "best_score": history_best_scalar[-1] if history_best_scalar else 0.0,
    "history": history_best_scalar,
    "best_controls": best_layout.controls_flat() if best_layout else None,
    "best_layout": best_layout,
    "best_triple": history_best_triple[-1] if history_best_triple else EfficiencyTriple.identity(),
  }


def _vector_to_layout(space: DecisionSpace, evaluator: ObjectiveEvaluator, x: np.ndarray) -> InterfaceLayout:
  return space.decode_layout(x, evaluator.registry)


BRUTE_FORCE_MAX_COMBINATIONS = 50_000


def brute_force_search_space_size(space: DecisionSpace) -> int:
  """Total discrete layouts: ∏ control choices × compositions of D fields into N forms."""
  control_combos = 1
  for cardinality in space.cardinalities():
    control_combos *= max(1, cardinality)
  d = space.control_dim()
  n = space.partition_dim()
  partition_combos = math.comb(d + n - 1, n - 1) if d >= 0 and n >= 1 else 1
  return control_combos * partition_combos


def _enumerate_field_partitions(d: int, n: int) -> Iterator[List[int]]:
  """All non-negative integer vectors of length n summing to d (wizard step counts)."""
  if n <= 0:
    return
  if n == 1:
    yield [d]
    return
  for head in range(d + 1):
    for tail in _enumerate_field_partitions(d - head, n - 1):
      yield [head] + tail


def brute_force(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  max_combinations: int = BRUTE_FORCE_MAX_COMBINATIONS,
) -> Dict[str, object]:
  """
  Exhaustive search over every control assignment and every valid wizard partition.

  Raises ValueError when the search space exceeds ``max_combinations`` (default 50_000).
  """
  total = brute_force_search_space_size(space)
  if total > max_combinations:
    raise ValueError(
      f"Brute force search space ({total} combinations) exceeds limit ({max_combinations}). "
      "Reduce fields, allowed controls, or max_forms."
    )

  d = space.control_dim()
  n = space.partition_dim()
  cardinalities = space.cardinalities()
  control_ranges = [range(max(1, c)) for c in cardinalities]
  partitions = list(_enumerate_field_partitions(d, n))

  best_score = -float("inf")
  best_layout: Optional[InterfaceLayout] = None
  history: List[float] = []

  for control_combo in itertools.product(*control_ranges):
    controls = space.all_controls(list(control_combo))
    for partition in partitions:
      layout = build_interface_layout_from_partition(space.fields, controls, partition)
      score = evaluator.scalar_fitness(layout)
      if score > best_score:
        best_score = score
        best_layout = layout
      history.append(best_score)

  if best_layout is None:
    raise RuntimeError("Brute force found no valid layout")

  return {
    "best_score": best_score,
    "history": history,
    "best_controls": best_layout.controls_flat(),
    "best_layout": best_layout,
    "evaluations": len(history),
    "search_space_size": total,
  }


def _genome_bounds(space: DecisionSpace) -> Tuple[np.ndarray, np.ndarray]:
  """Per-gene lower/upper bounds for the D + N genome."""
  lows: List[float] = []
  highs: List[float] = []
  for field in space.fields:
    span = max(0, len(field.allowed_controls()) - 1)
    lows.append(0.0)
    highs.append(float(span))
  d = space.control_dim()
  for _ in range(space.partition_dim()):
    lows.append(0.0)
    highs.append(float(max(d, 1)))
  return np.array(lows, dtype=float), np.array(highs, dtype=float)


def _sbx_gene(
  y1: float,
  y2: float,
  lb: float,
  ub: float,
  eta: float = 15.0,
) -> Tuple[float, float]:
  """Simulated binary crossover for one gene (Deb & Agrawal, 1995)."""
  if abs(y1 - y2) <= 1e-14:
    return y1, y2
  if y1 > y2:
    y1, y2 = y2, y1
  span = max(1e-14, y2 - y1)

  rand = random.random()
  beta = 1.0 + (2.0 * (y1 - lb) / span)
  alpha = 2.0 - beta ** (-(eta + 1.0))
  if rand <= 1.0 / alpha:
    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
  else:
    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
  c1 = 0.5 * ((y1 + y2) - betaq * span)

  rand = random.random()
  beta = 1.0 + (2.0 * (ub - y2) / span)
  alpha = 2.0 - beta ** (-(eta + 1.0))
  if rand <= 1.0 / alpha:
    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
  else:
    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
  c2 = 0.5 * ((y1 + y2) + betaq * span)
  return clamp(c1, lb, ub), clamp(c2, lb, ub)


def _sbx_crossover(
  parent1: np.ndarray,
  parent2: np.ndarray,
  bounds_low: np.ndarray,
  bounds_high: np.ndarray,
  crossover_prob: float = 0.9,
  eta: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
  child1 = parent1.copy()
  child2 = parent2.copy()
  for i in range(len(parent1)):
    if random.random() > crossover_prob:
      continue
    c1, c2 = _sbx_gene(
      float(parent1[i]),
      float(parent2[i]),
      float(bounds_low[i]),
      float(bounds_high[i]),
      eta=eta,
    )
    child1[i] = c1
    child2[i] = c2
  return child1, child2


def _gaussian_mutate_genome(
  genome: np.ndarray,
  space: DecisionSpace,
  mutation_prob: float = 0.2,
  mutation_sigma: float = 0.5,
) -> np.ndarray:
  """Gaussian perturbation with bounds adapted to control vs partition segments."""
  mutant = genome.copy()
  bounds_low, bounds_high = _genome_bounds(space)
  control_d = space.control_dim()
  for i in range(len(mutant)):
    if random.random() >= mutation_prob:
      continue
    sigma = mutation_sigma if i < control_d else mutation_sigma * max(1.0, space.control_dim() * 0.15)
    mutant[i] = clamp(float(mutant[i]) + random.gauss(0.0, sigma), bounds_low[i], bounds_high[i])
  return mutant


def classic_genetic_algorithm(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  pop_size: int = 30,
  generations: int = 30,
  crossover_prob: float = 0.9,
  mutation_prob: float = 0.2,
  mutation_sigma: float = 0.5,
  tournament_size: int = 3,
  sbx_eta: float = 15.0,
  random_seed: int | None = None,
) -> Dict[str, object]:
  """
  Single-objective genetic algorithm using scalar_fitness_from_triple().

  Selection: tournament on scalar fitness.
  Variation: SBX crossover + Gaussian mutation on the D + N genome.
  """
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  bounds_low, bounds_high = _genome_bounds(space)
  population = [space.random_vector() for _ in range(pop_size)]
  scores = np.array(
    [evaluator.scalar_fitness(_vector_to_layout(space, evaluator, x)) for x in population]
  )
  best_index = int(np.argmax(scores))
  best_layout = _vector_to_layout(space, evaluator, population[best_index])
  best_score = float(scores[best_index])
  history: List[float] = [best_score]

  def tournament_select() -> np.ndarray:
    candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
    winner = max(candidates, key=lambda i: scores[i])
    return population[winner].copy()

  for _ in range(generations):
    offspring: List[np.ndarray] = []
    while len(offspring) < pop_size:
      p1 = tournament_select()
      p2 = tournament_select()
      c1, c2 = _sbx_crossover(p1, p2, bounds_low, bounds_high, crossover_prob, sbx_eta)
      offspring.append(_gaussian_mutate_genome(c1, space, mutation_prob, mutation_sigma))
      if len(offspring) < pop_size:
        offspring.append(_gaussian_mutate_genome(c2, space, mutation_prob, mutation_sigma))

    offspring_scores = np.array(
      [evaluator.scalar_fitness(_vector_to_layout(space, evaluator, x)) for x in offspring]
    )
    combined_pop = population + offspring
    combined_scores = np.concatenate([scores, offspring_scores])
    order = np.argsort(combined_scores)[::-1]
    population = [combined_pop[i] for i in order[:pop_size]]
    scores = combined_scores[order[:pop_size]]

    if scores[0] > best_score:
      best_score = float(scores[0])
      best_layout = _vector_to_layout(space, evaluator, population[0])
    history.append(best_score)

  return {
    "best_score": best_score,
    "history": history,
    "best_controls": best_layout.controls_flat(),
    "best_layout": best_layout,
  }


def hill_climb(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  iterations: int = 200,
  random_seed: int | None = None,
):
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  current = space.random_vector()
  current_layout = _vector_to_layout(space, evaluator, current)
  current_score = evaluator.scalar_fitness(current_layout)
  history = [current_score]
  best_layout = current_layout
  best_score = current_score

  for _ in range(iterations):
    improved = False
    d = space.control_dim()
    for dim in range(space.dimension()):
      best_local = float(current[dim])
      best_local_score = current_score
      if dim < d:
        span = len(space.fields[dim].allowed_controls()) - 1
        deltas = (-1, 1)
      else:
        span = None
        deltas = (-0.2, 0.2)
      for delta in deltas:
        candidate = current.copy()
        if span is not None:
          candidate[dim] = clamp(candidate[dim] + delta, 0, span)
        else:
          candidate[dim] = max(0.0, float(candidate[dim]) + delta)
        layout = _vector_to_layout(space, evaluator, candidate)
        score = evaluator.scalar_fitness(layout)
        if score > best_local_score:
          best_local_score = score
          best_local = float(candidate[dim])
          improved = True
      current[dim] = best_local
      current_layout = _vector_to_layout(space, evaluator, current)
      current_score = evaluator.scalar_fitness(current_layout)
    history.append(current_score)
    if current_score > best_score:
      best_score = current_score
      best_layout = current_layout
    if not improved:
      break

  return {
    "best_score": best_score,
    "history": history,
    "best_controls": best_layout.controls_flat(),
    "best_layout": best_layout,
  }


def pso(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  swarm_size: int = 30,
  iterations: int = 50,
  inertia: float = 0.7,
  cognitive: float = 1.5,
  social: float = 1.5,
  random_seed: int | None = None,
):
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  d = space.dimension()
  positions = np.array([space.random_vector() for _ in range(swarm_size)])
  velocities = np.zeros_like(positions)
  personal_best_positions = positions.copy()
  personal_best_scores = np.array(
    [
      evaluator.scalar_fitness(_vector_to_layout(space, evaluator, p))
      for p in positions
    ]
  )
  global_best_index = int(np.argmax(personal_best_scores))
  global_best_position = personal_best_positions[global_best_index].copy()
  global_best_score = float(personal_best_scores[global_best_index])
  history = [global_best_score]

  for _ in range(iterations):
    r1 = np.random.rand(swarm_size, d)
    r2 = np.random.rand(swarm_size, d)
    velocities = (
      inertia * velocities
      + cognitive * r1 * (personal_best_positions - positions)
      + social * r2 * (global_best_position - positions)
    )
    positions = positions + velocities
    for i in range(swarm_size):
      layout = _vector_to_layout(space, evaluator, positions[i])
      score = evaluator.scalar_fitness(layout)
      if score > personal_best_scores[i]:
        personal_best_scores[i] = score
        personal_best_positions[i] = positions[i].copy()
        if score > global_best_score:
          global_best_score = score
          global_best_position = positions[i].copy()
    history.append(global_best_score)

  best_layout = _vector_to_layout(space, evaluator, global_best_position)
  return {
    "best_score": global_best_score,
    "history": history,
    "best_controls": best_layout.controls_flat(),
    "best_layout": best_layout,
  }


def random_search(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  iterations: int = 200,
  random_seed: int | None = None,
):
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  best_score = -1.0
  best_layout: Optional[InterfaceLayout] = None
  history: List[float] = []
  for _ in range(iterations):
    candidate = space.random_vector()
    layout = _vector_to_layout(space, evaluator, candidate)
    score = evaluator.scalar_fitness(layout)
    if score > best_score:
      best_score = score
      best_layout = layout
    history.append(best_score)
  return {
    "best_score": best_score,
    "history": history,
    "best_controls": best_layout.controls_flat() if best_layout else None,
    "best_layout": best_layout,
  }


def greedy(space: DecisionSpace, evaluator: ObjectiveEvaluator) -> Dict[str, object]:
  control_idx = [0] * len(space.fields)
  partition_counts = [len(space.fields)] + [0] * (space.partition_dim() - 1)
  controls = space.all_controls(control_idx)
  form_idx = space.counts_to_form_indices(partition_counts)
  layout = build_interface_layout(space.fields, controls, form_idx)
  best_score = evaluator.scalar_fitness(layout)

  for field_i, field in enumerate(space.fields):
    best_control = control_idx[field_i]
    best_local = best_score
    for idx in range(len(field.allowed_controls())):
      trial_controls = list(control_idx)
      trial_controls[field_i] = idx
      trial = space.all_controls(trial_controls)
      trial_layout = build_interface_layout(space.fields, trial, form_idx)
      score = evaluator.scalar_fitness(trial_layout)
      if score > best_local:
        best_local = score
        best_control = idx
    control_idx[field_i] = best_control
    controls = space.all_controls(control_idx)
    layout = build_interface_layout(space.fields, controls, form_idx)
    best_score = evaluator.scalar_fitness(layout)

  d = len(space.fields)
  improved_partition = True
  while improved_partition:
    improved_partition = False
    for from_form in range(space.partition_dim()):
      for to_form in range(space.partition_dim()):
        if from_form == to_form or partition_counts[from_form] <= 0:
          continue
        trial_counts = list(partition_counts)
        trial_counts[from_form] -= 1
        trial_counts[to_form] += 1
        if sum(trial_counts) != d:
          continue
        trial_form_idx = space.counts_to_form_indices(trial_counts)
        trial_layout = build_interface_layout(space.fields, controls, trial_form_idx)
        score = evaluator.scalar_fitness(trial_layout)
        if score > best_score:
          best_score = score
          partition_counts = trial_counts
          form_idx = trial_form_idx
          layout = trial_layout
          improved_partition = True

  return {
    "best_score": best_score,
    "history": [best_score],
    "best_controls": layout.controls_flat(),
    "best_layout": layout,
  }


def simulated_annealing(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  iterations: int = 300,
  initial_temp: float = 5.0,
  cooling: float = 0.97,
  random_seed: int | None = None,
):
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  current = space.random_vector()
  current_layout = _vector_to_layout(space, evaluator, current)
  current_score = evaluator.scalar_fitness(current_layout)
  best_layout = current_layout
  best_score = current_score
  history = [best_score]
  temp = initial_temp
  d = space.dimension()

  control_d = space.control_dim()
  for _ in range(iterations):
    dim = random.randrange(d)
    candidate = current.copy()
    if dim < control_d:
      span = len(space.fields[dim].allowed_controls()) - 1
      candidate[dim] = clamp(candidate[dim] + random.choice([-1.0, 1.0]), 0, span)
    else:
      candidate[dim] = max(0.0, float(candidate[dim]) + random.uniform(-0.25, 0.25))
    candidate_layout = _vector_to_layout(space, evaluator, candidate)
    candidate_score = evaluator.scalar_fitness(candidate_layout)
    delta = candidate_score - current_score
    if delta > 0 or random.random() < math.exp(delta / max(1e-9, temp)):
      current = candidate
      current_layout = candidate_layout
      current_score = candidate_score
    if current_score > best_score:
      best_score = current_score
      best_layout = current_layout
    history.append(best_score)
    temp *= cooling

  return {
    "best_score": best_score,
    "history": history,
    "best_controls": best_layout.controls_flat(),
    "best_layout": best_layout,
  }


def tabu_search(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  iterations: int = 200,
  tabu_tenure: int = 7,
  random_seed: int | None = None,
):
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  current = space.random_vector()
  current_layout = _vector_to_layout(space, evaluator, current)
  current_score = evaluator.scalar_fitness(current_layout)
  best_layout = current_layout
  best_score = current_score
  history = [best_score]
  tabu_list: List[Tuple[int, int]] = []
  control_idx, partition_counts = space.round_to_valid(current)
  form_idx = space.counts_to_form_indices(partition_counts)

  for _ in range(iterations):
    best_neighbor: Tuple[List[int], List[int], float] | None = None
    move_used: Tuple[int, int] | None = None
    for i, field in enumerate(space.fields):
      for idx in range(len(field.allowed_controls())):
        if idx == control_idx[i]:
          continue
        move = (i, idx)
        trial_controls = list(control_idx)
        trial_controls[i] = idx
        trial = space.all_controls(trial_controls)
        layout = build_interface_layout(space.fields, trial, form_idx)
        score = evaluator.scalar_fitness(layout)
        if move in tabu_list and score <= best_score:
          continue
        if best_neighbor is None or score > best_neighbor[2]:
          best_neighbor = (trial_controls, partition_counts, score)
          move_used = move
    for from_form in range(space.partition_dim()):
      for to_form in range(space.partition_dim()):
        if from_form == to_form or partition_counts[from_form] <= 0:
          continue
        move = (1000 + from_form, to_form)
        trial_counts = list(partition_counts)
        trial_counts[from_form] -= 1
        trial_counts[to_form] += 1
        trial_form_idx = space.counts_to_form_indices(trial_counts)
        layout = build_interface_layout(
          space.fields,
          space.all_controls(control_idx),
          trial_form_idx,
        )
        score = evaluator.scalar_fitness(layout)
        if move in tabu_list and score <= best_score:
          continue
        if best_neighbor is None or score > best_neighbor[2]:
          best_neighbor = (control_idx, trial_counts, score)
          move_used = move
    if best_neighbor is None:
      break
    control_idx, partition_counts = best_neighbor[0], best_neighbor[1]
    form_idx = space.counts_to_form_indices(partition_counts)
    current_score = best_neighbor[2]
    current_layout = build_interface_layout(space.fields, space.all_controls(control_idx), form_idx)
    if move_used:
      tabu_list.append(move_used)
      if len(tabu_list) > tabu_tenure:
        tabu_list.pop(0)
    if current_score > best_score:
      best_score = current_score
      best_layout = current_layout
    history.append(best_score)

  return {
    "best_score": best_score,
    "history": history,
    "best_controls": best_layout.controls_flat(),
    "best_layout": best_layout,
  }


def aco(
  space: DecisionSpace,
  evaluator: ObjectiveEvaluator,
  ants: int = 20,
  iterations: int = 40,
  alpha: float = 1.0,
  beta: float = 2.0,
  evaporation: float = 0.1,
  deposit_weight: float = 1.0,
  random_seed: int | None = None,
):
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  num_fields = len(space.fields)
  allowed = [field.allowed_controls() for field in space.fields]
  max_controls = max(len(c) for c in allowed) if allowed else 0
  pheromone = np.ones((num_fields, max_controls), dtype=float)
  heuristic = np.zeros_like(pheromone)
  for i, field in enumerate(space.fields):
    for j, ctrl in enumerate(allowed[i]):
      trial = [0] * num_fields
      trial[i] = j
      counts = [num_fields] + [0] * (space.partition_dim() - 1)
      layout = build_interface_layout(
        space.fields,
        space.all_controls(trial),
        space.counts_to_form_indices(counts),
      )
      heuristic[i, j] = evaluator.scalar_fitness(layout)

  best_layout: Optional[InterfaceLayout] = None
  best_score = -1.0
  history: List[float] = []

  for _ in range(iterations):
    iteration_best = -1.0
    iteration_best_layout: Optional[InterfaceLayout] = None
    for _ant in range(ants):
      chosen_indexes: List[int] = []
      for i in range(num_fields):
        options = len(allowed[i])
        tau = pheromone[i, :options] ** alpha
        eta = heuristic[i, :options] ** beta
        probs = tau * eta
        if np.sum(probs) <= 0:
          probs = np.ones_like(probs)
        probs = probs / np.sum(probs)
        idx = int(np.random.choice(np.arange(options), p=probs))
        chosen_indexes.append(idx)
      partition_vec = np.array(
        [random.uniform(0.1, 1.0) for _ in range(space.partition_dim())],
        dtype=float,
      )
      genome = np.concatenate(
        [
          np.array(chosen_indexes, dtype=float),
          partition_vec,
        ]
      )
      layout = space.decode_layout(genome, evaluator.registry)
      score = evaluator.scalar_fitness(layout)
      if score > iteration_best:
        iteration_best = score
        iteration_best_layout = layout
    pheromone *= 1.0 - evaporation
    if iteration_best_layout is not None:
      for i, ctrl in enumerate(iteration_best_layout.controls_flat()):
        idx = allowed[i].index(ctrl)
        pheromone[i, idx] += deposit_weight * iteration_best
      if iteration_best > best_score:
        best_score = iteration_best
        best_layout = iteration_best_layout
    history.append(best_score if best_score >= 0 else iteration_best)

  return {
    "best_score": best_score,
    "history": history,
    "best_controls": best_layout.controls_flat() if best_layout else None,
    "best_layout": best_layout,
  }
