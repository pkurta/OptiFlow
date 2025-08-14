from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from optiflow.models.scoring import ControlType, FieldSpec, FunctionRegistry, ScoreTriple


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def weighted_score(scores: List[ScoreTriple], weights: Tuple[float, float, float]) -> float:
    w = np.array(weights, dtype=float)
    s = np.array(scores, dtype=float)
    return float(np.mean(s @ w))


def sum_objectives(scores: List[ScoreTriple]) -> ScoreTriple:
    s = np.array(scores, dtype=float)
    totals = np.mean(s, axis=0)
    return float(totals[0]), float(totals[1]), float(totals[2])


@dataclass
class DecisionSpace:
    fields: List[FieldSpec]

    def cardinalities(self) -> List[int]:
        return [len(f.allowed_controls()) for f in self.fields]

    def round_to_valid(self, x: np.ndarray) -> List[int]:
        # x is real vector; map to valid integer index per field
        rounded: List[int] = []
        for i, f in enumerate(self.fields):
            k = len(f.allowed_controls())
            xi = int(round(clamp(x[i], 0, k - 1)))
            rounded.append(xi)
        return rounded

    def random_vector(self) -> np.ndarray:
        return np.array([random.uniform(0, len(f.allowed_controls()) - 1) for f in self.fields], dtype=float)

    def all_controls(self, discrete_indexes: List[int]) -> List[ControlType]:
        result: List[ControlType] = []
        for idx, f in zip(discrete_indexes, self.fields):
            allowed = f.allowed_controls()
            result.append(allowed[int(idx)])
        return result


class ObjectiveEvaluator:
    def __init__(self, registry: FunctionRegistry, weights: Tuple[float, float, float]) -> None:
        self.registry = registry
        self.weights = weights

    def evaluate_controls(self, controls: List[ControlType], fields: List[FieldSpec]) -> List[ScoreTriple]:
        triples: List[ScoreTriple] = []
        for control, field in zip(controls, fields):
            triples.append(self.registry.evaluate(control, field.data_type, field.size))
        return triples

    def scalar_fitness(self, controls: List[ControlType], fields: List[FieldSpec]) -> float:
        triples = self.evaluate_controls(controls, fields)
        return weighted_score(triples, self.weights)

    def multi_objective(self, controls: List[ControlType], fields: List[FieldSpec]) -> ScoreTriple:
        triples = self.evaluate_controls(controls, fields)
        return sum_objectives(triples)


def non_dominated_sort(points: List[Tuple[float, float, float]]) -> List[List[int]]:
    # Higher is better (maximize). NSGA-II typically assumes minimization; invert if needed
    N = len(points)
    S = [set() for _ in range(N)]
    n = [0] * N
    ranks = [None] * N
    fronts: List[List[int]] = []

    def dominates(p, q) -> bool:
        return all(p[i] >= q[i] for i in range(3)) and any(p[i] > q[i] for i in range(3))

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(points[p], points[q]):
                S[p].add(q)
            elif dominates(points[q], points[p]):
                n[p] += 1
        if n[p] == 0:
            ranks[p] = 0
    front0 = [i for i in range(N) if ranks[i] == 0]
    if front0:
        fronts.append(front0)

    i = 0
    while i < len(fronts):
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def crowding_distance(front_points: List[Tuple[float, float, float]], front_indexes: List[int]) -> Dict[int, float]:
    # Higher is better
    M = 3
    l = len(front_indexes)
    if l == 0:
        return {}
    distance: Dict[int, float] = {idx: 0.0 for idx in front_indexes}
    F = np.array(front_points, dtype=float)
    for m in range(M):
        order = np.argsort(F[:, m])
        min_m = F[order[0], m]
        max_m = F[order[-1], m]
        distance[front_indexes[order[0]]] = float("inf")
        distance[front_indexes[order[-1]]] = float("inf")
        span = max(1e-9, max_m - min_m)
        for i in range(1, l - 1):
            prev_val = F[order[i - 1], m]
            next_val = F[order[i + 1], m]
            distance[front_indexes[order[i]]] += (next_val - prev_val) / span
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

    D = len(space.fields)

    def evaluate_population(pop: List[np.ndarray]) -> List[Tuple[float, float, float]]:
        objs: List[Tuple[float, float, float]] = []
        for x in pop:
            discrete = space.round_to_valid(x)
            controls = space.all_controls(discrete)
            objs.append(evaluator.multi_objective(controls, space.fields))
        return objs

    # Initialize population uniformly at random
    population = [space.random_vector() for _ in range(pop_size)]
    objectives = evaluate_population(population)
    history_best_scalar: List[float] = []
    history_best_mapping: List[int] | None = None
    history_best_controls: List[ControlType] | None = None

    for g in range(generations):
        # Create offspring via tournament selection, simulated binary crossover, gaussian mutation
        offspring: List[np.ndarray] = []
        while len(offspring) < pop_size:
            def tournament_select(k: int = 2) -> np.ndarray:
                candidates = random.sample(range(pop_size), k)
                # selection based on rank and crowding distance
                fronts = non_dominated_sort([objectives[i] for i in candidates])
                first_front = fronts[0]
                if len(first_front) == 1:
                    return population[candidates[first_front[0]]].copy()
                # crowding among selected
                cd = crowding_distance([objectives[candidates[i]] for i in first_front], [i for i in first_front])
                best_index = max(first_front, key=lambda i: cd.get(i, 0.0))
                return population[candidates[best_index]].copy()

            p1 = tournament_select()
            p2 = tournament_select()
            c1 = p1.copy()
            c2 = p2.copy()
            if random.random() < crossover_prob:
                alpha = np.random.uniform(0.0, 1.0, size=D)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            if random.random() < mutation_prob:
                c1 = c1 + np.random.normal(0, mutation_sigma, size=D)
            if random.random() < mutation_prob:
                c2 = c2 + np.random.normal(0, mutation_sigma, size=D)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # Combine and select next generation using non-dominated sorting and crowding
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
                # fill remainder by crowding distance
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

        # Track best by weighted scalar for comparison
        best_scalar = -1.0
        best_controls: List[ControlType] | None = None
        for x in population:
            discrete = space.round_to_valid(x)
            controls = space.all_controls(discrete)
            scalar = evaluator.scalar_fitness(controls, space.fields)
            if scalar > best_scalar:
                best_scalar = scalar
                best_controls = controls
        history_best_scalar.append(best_scalar)
        if best_controls is not None:
            history_best_controls = best_controls
            history_best_mapping = space.round_to_valid(population[0])

    return {
        "best_score": history_best_scalar[-1] if history_best_scalar else 0.0,
        "history": history_best_scalar,
        "best_controls": history_best_controls,
    }


def hill_climb(space: DecisionSpace, evaluator: ObjectiveEvaluator, iterations: int = 200, random_seed: int | None = None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Start from random discrete solution
    current = space.round_to_valid(space.random_vector())
    current_controls = space.all_controls(current)
    current_score = evaluator.scalar_fitness(current_controls, space.fields)
    history = [current_score]

    for _ in range(iterations):
        improved = False
        for i, field in enumerate(space.fields):
            best_local = current[i]
            best_local_score = current_score
            for idx in range(len(field.allowed_controls())):
                if idx == current[i]:
                    continue
                candidate = list(current)
                candidate[i] = idx
                candidate_controls = space.all_controls(candidate)
                score = evaluator.scalar_fitness(candidate_controls, space.fields)
                if score > best_local_score:
                    best_local_score = score
                    best_local = idx
                    improved = True
            current[i] = best_local
            current_controls = space.all_controls(current)
            current_score = evaluator.scalar_fitness(current_controls, space.fields)
        history.append(current_score)
        if not improved:
            break

    return {"best_score": current_score, "history": history, "best_controls": current_controls}


def pso(space: DecisionSpace, evaluator: ObjectiveEvaluator, swarm_size: int = 30, iterations: int = 50, inertia: float = 0.7, cognitive: float = 1.5, social: float = 1.5, random_seed: int | None = None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    D = len(space.fields)
    positions = np.array([space.random_vector() for _ in range(swarm_size)])
    velocities = np.zeros_like(positions)
    personal_best_positions = positions.copy()
    personal_best_scores = np.array([evaluator.scalar_fitness(space.all_controls(space.round_to_valid(p)), space.fields) for p in positions])
    global_best_index = int(np.argmax(personal_best_scores))
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_score = float(personal_best_scores[global_best_index])

    history = [global_best_score]

    for _ in range(iterations):
        r1 = np.random.rand(swarm_size, D)
        r2 = np.random.rand(swarm_size, D)
        velocities = inertia * velocities + cognitive * r1 * (personal_best_positions - positions) + social * r2 * (global_best_position - positions)
        positions = positions + velocities
        # Evaluate
        for i in range(swarm_size):
            score = evaluator.scalar_fitness(space.all_controls(space.round_to_valid(positions[i])), space.fields)
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i].copy()
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()
        history.append(global_best_score)

    best_controls = space.all_controls(space.round_to_valid(global_best_position))
    return {"best_score": global_best_score, "history": history, "best_controls": best_controls}


def random_search(space: DecisionSpace, evaluator: ObjectiveEvaluator, iterations: int = 200, random_seed: int | None = None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    best_score = -1.0
    best_controls: List[ControlType] | None = None
    history: List[float] = []
    for _ in range(iterations):
        candidate = space.round_to_valid(space.random_vector())
        controls = space.all_controls(candidate)
        score = evaluator.scalar_fitness(controls, space.fields)
        if score > best_score:
            best_score = score
            best_controls = controls
        history.append(best_score)
    return {"best_score": best_score, "history": history, "best_controls": best_controls}


