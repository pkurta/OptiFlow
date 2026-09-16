from __future__ import annotations

import unittest

from optiflow.models.scoring import (
  ControlType,
  DataType,
  EfficiencyTriple,
  FieldSpec,
  build_interface_layout_from_partition,
)
from optiflow.optimization.algorithms import CriterionWeights
from optiflow.optimization.runner import SUITE_STEPS
from optiflow.ui.gemini_interpretation import build_interpretation_prompt
from optiflow.ui.run_results import AlgorithmRunSummary


def _fake_summary(
  key: str,
  raw_fitness: float,
  *,
  fitness: float | None = None,
) -> AlgorithmRunSummary:
  layout = build_interface_layout_from_partition(
    [FieldSpec("A", DataType.BOOLEAN, 1)], [ControlType.CHECKBOX], [1],
  )
  return AlgorithmRunSummary(
    key=key,
    label=f"label-{key}",
    layout=layout,
    triple=EfficiencyTriple(0.5, 0.5, 0.5),
    fitness=fitness if fitness is not None else max(0.0, min(1.0, raw_fitness)),
    raw_fitness=raw_fitness,
    form_count=1,
    history_steps=1,
    algo_best_score=raw_fitness,
    elapsed_s=0.1,
    ran=True,
  )


class LeaderSelectionTests(unittest.TestCase):
  def test_leader_is_the_actual_best_raw_fitness_not_the_first_in_suite_order(self) -> None:
    # SUITE_STEPS order is fixed (NSGA-II, BruteForce, GA, ...); pick two keys
    # from it and give the FIRST one in that order the WORSE raw_fitness, so a
    # naive "first summary with a layout" pick would get it wrong.
    first_key, _ = SUITE_STEPS[0]
    second_key, _ = SUITE_STEPS[1]
    summaries = [
      _fake_summary(first_key, raw_fitness=0.1),
      _fake_summary(second_key, raw_fitness=0.9),
    ]

    prompt = build_interpretation_prompt(
      summaries=summaries,
      report_text="",
      fields=[FieldSpec("A", DataType.BOOLEAN, 1)],
      max_forms=1,
      weights=CriterionWeights.balanced(),
      cancelled=False,
      warning=None,
      optiflow_version="test",
    )

    self.assertIn(f"Лидер по F: label-{second_key}", prompt)
    self.assertNotIn(f"Лидер по F: label-{first_key}", prompt)

  def test_leader_block_annotates_raw_fitness_when_clipped(self) -> None:
    key, _ = SUITE_STEPS[0]
    summaries = [_fake_summary(key, raw_fitness=-0.72, fitness=0.0)]

    prompt = build_interpretation_prompt(
      summaries=summaries,
      report_text="",
      fields=[FieldSpec("A", DataType.BOOLEAN, 1)],
      max_forms=1,
      weights=CriterionWeights.balanced(),
      cancelled=False,
      warning=None,
      optiflow_version="test",
    )

    self.assertIn("F=0.0000", prompt)
    self.assertIn("-0.7200", prompt)


if __name__ == "__main__":
  unittest.main()
