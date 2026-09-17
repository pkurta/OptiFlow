#!/usr/bin/env python3
"""Raw measurement script for the T_cpu(M) / DeltaE(M) / constraint_pass_rate
experiments referenced in docs/dissertation/audit_glava_5_programmnii_prototip_optiflow.md
(chapters 4-5). Produces raw per-run data and a summary table. Does NOT write
any dissertation prose -- see the printed report and the output files for the
raw facts only.

Usage:
  python scripts/run_dissertation_experiments.py [--repeats N] [--quick]

--quick runs 2 repeats per point instead of the full count, for a fast sanity
check of the pipeline before committing to the full (slow) run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import numpy as np

from optiflow.benchmarks import ensure_allowed_controls
from optiflow.models.scoring import DataType, FieldSpec, FunctionRegistry, InterfaceLayout
from optiflow.optimization.algorithms import (
  CriterionWeights,
  DecisionSpace,
  ObjectiveEvaluator,
  brute_force_search_space_size,
  calculate_fitness,
  clip_fitness_for_display,
  compute_total_efficiency,
)
from optiflow.optimization.corrections import MILLER_HARD_LIMIT, compute_miller_penalty
from optiflow.optimization.runner import SUITE_STEPS, run_optimization_suite

OUT_DIR = ROOT / "docs" / "dissertation" / "experiments"
RAW_CSV = OUT_DIR / "raw_runs.csv"
SUMMARY_MD = OUT_DIR / "summary.md"
SUMMARY_CSV = OUT_DIR / "summary.csv"
ENV_JSON = OUT_DIR / "environment.json"

WEIGHTS = CriterionWeights.balanced()

# --- Experiment matrix -------------------------------------------------
# N = ceil(M / MILLER_HARD_LIMIT): the smallest N for which D=M is Inv_3-
# feasible (D <= 9N). This is the same recipe RealisticMillerConvergenceTests
# uses (D=17, N=2 -> capacity 18, margin 1); the resulting margin varies with
# M since capacity only grows in steps of 9 (see printed table below).
FEASIBLE_POINTS: List[Tuple[int, int]] = [
  (m, math.ceil(m / MILLER_HARD_LIMIT)) for m in (5, 10, 20, 50, 100)
]
# One deliberately Miller-infeasible point (D > 9N), separate from the main
# table, solely to get a non-trivial constraint_pass_rate reading.
INFEASIBLE_POINTS: List[Tuple[int, int]] = [(50, 5)]  # capacity 45 < 50, deficit 5

BRUTE_FORCE_LIMIT = 50_000

FIELD_SEED_OFFSET = 10_000  # field composition seed = FIELD_SEED_OFFSET + M (feasible) or +1 for infeasible
S0_SEED = 999_999  # fixed, out-of-band seed for the pre-optimization baseline layout


def make_fields(m: int, field_seed: int) -> List[FieldSpec]:
  """Mixed BOOLEAN/UNSIGNED/TEXT fields, same generation style as
  optiflow.benchmarks.random_benchmark_snapshot, but with a fixed field count."""
  rng = random.Random(field_seed)
  fields: List[FieldSpec] = []
  for i in range(m):
    dtype = rng.choice(list(DataType))
    size = rng.randint(1, 24 if dtype == DataType.TEXT else 6)
    fields.append(FieldSpec(name=f"Field{i + 1}", data_type=dtype, size=size))
  ensure_allowed_controls(fields)
  return fields


def max_form_size(layout: Optional[InterfaceLayout]) -> Optional[int]:
  if layout is None:
    return None
  return max((len(form.elements) for form in layout.forms), default=0)


@dataclass
class RunRecord:
  series: str  # "feasible" or "infeasible"
  m: int
  n: int
  algorithm: str
  seed: int
  elapsed_s: Optional[float]
  raw_fitness: Optional[float]
  fitness_display: Optional[float]
  potency: Optional[float]
  operativeness: Optional[float]
  resource_saving: Optional[float]
  max_k: Optional[int]
  constraint_satisfied: Optional[bool]
  has_layout: bool
  raw_fitness_s0: float
  potency_s0: float
  operativeness_s0: float
  resource_saving_s0: float
  delta_f_vs_s0: Optional[float]
  delta_p_vs_s0: Optional[float]
  delta_o_vs_s0: Optional[float]
  delta_r_vs_s0: Optional[float]
  brute_force_available: bool
  omega_size: Optional[int]
  delta_f_vs_brute_force: Optional[float]


RUN_RECORD_FIELDS = [f for f in RunRecord.__dataclass_fields__.keys()]


def run_point(series: str, m: int, n: int, field_seed: int, repeats: int) -> List[RunRecord]:
  fields = make_fields(m, field_seed)
  space = DecisionSpace(fields, max_forms=n)
  registry = FunctionRegistry()
  evaluator = ObjectiveEvaluator(registry, WEIGHTS)

  omega_size = brute_force_search_space_size(space)
  brute_force_available = omega_size <= BRUTE_FORCE_LIMIT

  # S0: fixed-seed, undecided-by-any-optimizer baseline layout, decoded by the
  # same DecisionSpace.decode_layout used everywhere else. See report for why
  # this specific definition was chosen (no S0 convention exists in the
  # codebase prior to this experiment).
  random.seed(S0_SEED)
  np.random.seed(S0_SEED)
  s0_vector = space.random_vector()
  s0_layout = space.decode_layout(s0_vector, registry)
  s0_triple = compute_total_efficiency(s0_layout, registry)
  s0_penalty = compute_miller_penalty(s0_layout, evaluator.penalty_weight)
  s0_raw_fitness = calculate_fitness(s0_triple, WEIGHTS, penalties=s0_penalty)

  records: List[RunRecord] = []
  for seed in range(repeats):
    random.seed(seed)
    np.random.seed(seed)
    payload = run_optimization_suite(space, evaluator, {})
    results = payload["results"]

    brute_force_raw: Optional[float] = None
    bf_result = results.get("BruteForce")
    if bf_result is not None and bf_result.get("best_layout") is not None:
      bf_layout = bf_result["best_layout"]
      bf_triple = compute_total_efficiency(bf_layout, registry)
      bf_penalty = compute_miller_penalty(bf_layout, evaluator.penalty_weight)
      brute_force_raw = calculate_fitness(bf_triple, WEIGHTS, penalties=bf_penalty)

    for key, _label in SUITE_STEPS:
      result = results.get(key) or {}
      layout = result.get("best_layout")
      elapsed = result.get("elapsed_s")
      if layout is None:
        records.append(RunRecord(
          series=series, m=m, n=n, algorithm=key, seed=seed,
          elapsed_s=elapsed, raw_fitness=None, fitness_display=None,
          potency=None, operativeness=None, resource_saving=None,
          max_k=None, constraint_satisfied=None, has_layout=False,
          raw_fitness_s0=s0_raw_fitness,
          potency_s0=s0_triple.potency, operativeness_s0=s0_triple.operativeness,
          resource_saving_s0=s0_triple.resource_saving,
          delta_f_vs_s0=None, delta_p_vs_s0=None, delta_o_vs_s0=None, delta_r_vs_s0=None,
          brute_force_available=brute_force_available, omega_size=omega_size,
          delta_f_vs_brute_force=None,
        ))
        continue
      triple = compute_total_efficiency(layout, registry)
      penalty = compute_miller_penalty(layout, evaluator.penalty_weight)
      raw_fitness = calculate_fitness(triple, WEIGHTS, penalties=penalty)
      mk = max_form_size(layout)
      records.append(RunRecord(
        series=series, m=m, n=n, algorithm=key, seed=seed,
        elapsed_s=elapsed,
        raw_fitness=raw_fitness,
        fitness_display=clip_fitness_for_display(raw_fitness),
        potency=triple.potency, operativeness=triple.operativeness,
        resource_saving=triple.resource_saving,
        max_k=mk,
        constraint_satisfied=(mk is not None and mk <= MILLER_HARD_LIMIT),
        has_layout=True,
        raw_fitness_s0=s0_raw_fitness,
        potency_s0=s0_triple.potency, operativeness_s0=s0_triple.operativeness,
        resource_saving_s0=s0_triple.resource_saving,
        delta_f_vs_s0=raw_fitness - s0_raw_fitness,
        delta_p_vs_s0=triple.potency - s0_triple.potency,
        delta_o_vs_s0=triple.operativeness - s0_triple.operativeness,
        delta_r_vs_s0=triple.resource_saving - s0_triple.resource_saving,
        brute_force_available=brute_force_available, omega_size=omega_size,
        delta_f_vs_brute_force=(raw_fitness - brute_force_raw) if brute_force_raw is not None else None,
      ))
  return records


def write_environment_info() -> None:
  info = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "machine": platform.machine(),
    "numpy_version": np.__version__,
  }
  try:
    import subprocess
    info["cpu_brand_macos"] = subprocess.run(
      ["sysctl", "-n", "machdep.cpu.brand_string"],
      capture_output=True, text=True, timeout=5,
    ).stdout.strip()
  except Exception:
    pass
  ENV_JSON.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repeats", type=int, default=10)
  parser.add_argument("--quick", action="store_true", help="2 repeats per point, for a pipeline smoke test")
  args = parser.parse_args()
  repeats = 2 if args.quick else args.repeats

  OUT_DIR.mkdir(parents=True, exist_ok=True)
  write_environment_info()

  all_records: List[RunRecord] = []
  wall_start = time.perf_counter()

  print(f"Feasible points (M, N): {FEASIBLE_POINTS}", flush=True)
  print(f"Infeasible points (M, N): {INFEASIBLE_POINTS}", flush=True)
  print(f"Repeats per point: {repeats}", flush=True)

  for m, n in FEASIBLE_POINTS:
    field_seed = FIELD_SEED_OFFSET + m
    omega = brute_force_search_space_size(DecisionSpace(make_fields(m, field_seed), max_forms=n))
    print(f"\n=== feasible M={m} N={n} capacity={9*n} margin={9*n-m} |Omega|={omega} ===", flush=True)
    point_start = time.perf_counter()
    records = run_point("feasible", m, n, field_seed, repeats)
    all_records.extend(records)
    print(f"    done in {time.perf_counter() - point_start:.1f}s", flush=True)
    with RAW_CSV.open("w", newline="", encoding="utf-8") as fh:
      writer = csv.DictWriter(fh, fieldnames=RUN_RECORD_FIELDS)
      writer.writeheader()
      for r in all_records:
        writer.writerow(r.__dict__)

  for m, n in INFEASIBLE_POINTS:
    field_seed = FIELD_SEED_OFFSET + m + 1  # distinct from the feasible-series field draw at the same M
    omega = brute_force_search_space_size(DecisionSpace(make_fields(m, field_seed), max_forms=n))
    print(f"\n=== infeasible M={m} N={n} capacity={9*n} deficit={m-9*n} |Omega|={omega} ===", flush=True)
    point_start = time.perf_counter()
    records = run_point("infeasible", m, n, field_seed, repeats)
    all_records.extend(records)
    print(f"    done in {time.perf_counter() - point_start:.1f}s", flush=True)
    with RAW_CSV.open("w", newline="", encoding="utf-8") as fh:
      writer = csv.DictWriter(fh, fieldnames=RUN_RECORD_FIELDS)
      writer.writeheader()
      for r in all_records:
        writer.writerow(r.__dict__)

  total_elapsed = time.perf_counter() - wall_start
  print(f"\nTotal wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)

  build_summary(all_records, total_elapsed, repeats)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
  if not values:
    return None, None
  mean = sum(values) / len(values)
  if len(values) < 2:
    return mean, 0.0
  var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
  return mean, math.sqrt(var)


def build_summary(records: List[RunRecord], total_elapsed_s: float, repeats: int) -> None:
  points = sorted({(r.series, r.m, r.n) for r in records})
  algorithms = [key for key, _label in SUITE_STEPS]

  summary_rows: List[Dict[str, object]] = []
  for series, m, n in points:
    for algo in algorithms:
      subset = [r for r in records if r.series == series and r.m == m and r.n == n and r.algorithm == algo]
      if not subset:
        continue
      elapsed_vals = [r.elapsed_s for r in subset if r.elapsed_s is not None]
      delta_f_vals = [r.delta_f_vs_s0 for r in subset if r.delta_f_vs_s0 is not None]
      raw_f_vals = [r.raw_fitness for r in subset if r.raw_fitness is not None]
      bf_delta_vals = [r.delta_f_vs_brute_force for r in subset if r.delta_f_vs_brute_force is not None]
      n_ok = sum(1 for r in subset if r.constraint_satisfied)
      n_with_layout = sum(1 for r in subset if r.has_layout)
      t_mean, t_std = _mean_std(elapsed_vals)
      f_mean, f_std = _mean_std(raw_f_vals)
      d_mean, d_std = _mean_std(delta_f_vals)
      bf_mean, bf_std = _mean_std(bf_delta_vals)
      summary_rows.append({
        "series": series, "M": m, "N": n, "algorithm": algo,
        "repeats": len(subset), "runs_with_layout": n_with_layout,
        "t_cpu_mean_s": t_mean, "t_cpu_std_s": t_std,
        "raw_fitness_mean": f_mean, "raw_fitness_std": f_std,
        "delta_f_vs_s0_mean": d_mean, "delta_f_vs_s0_std": d_std,
        "delta_f_vs_brute_force_mean": bf_mean, "delta_f_vs_brute_force_std": bf_std,
        "brute_force_available": subset[0].brute_force_available,
        "omega_size": subset[0].omega_size,
        "constraint_pass_rate": (n_ok / n_with_layout) if n_with_layout else None,
      })

  with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
    writer.writeheader()
    for row in summary_rows:
      writer.writerow(row)

  lines = [
    "# OptiFlow dissertation experiment: raw summary (script output, not manuscript text)",
    "",
    f"Repeats per (M, N, algorithm): {repeats}",
    f"Total wall time: {total_elapsed_s:.1f} s ({total_elapsed_s/60:.1f} min)",
    "",
    "| series | M | N | algorithm | repeats | T_cpu mean (s) | T_cpu std (s) | "
    "raw F mean | delta_F vs S0 mean | delta_F vs S0 std | delta_F vs BF mean | "
    "constraint_pass_rate | |Omega| | BF available |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
  ]
  for row in summary_rows:
    def fmt(v: object, nd: int = 4) -> str:
      return f"{v:.{nd}f}" if isinstance(v, float) else "n/a"
    lines.append(
      f"| {row['series']} | {row['M']} | {row['N']} | {row['algorithm']} | {row['repeats']} | "
      f"{fmt(row['t_cpu_mean_s'], 3)} | {fmt(row['t_cpu_std_s'], 3)} | "
      f"{fmt(row['raw_fitness_mean'])} | {fmt(row['delta_f_vs_s0_mean'])} | {fmt(row['delta_f_vs_s0_std'])} | "
      f"{fmt(row['delta_f_vs_brute_force_mean'])} | {fmt(row['constraint_pass_rate'])} | "
      f"{row['omega_size']} | {row['brute_force_available']} |"
    )
  SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

  print(f"\nWrote raw data: {RAW_CSV}")
  print(f"Wrote summary table: {SUMMARY_MD}")
  print(f"Wrote summary csv: {SUMMARY_CSV}")
  print(f"Wrote environment info: {ENV_JSON}")


if __name__ == "__main__":
  main()
