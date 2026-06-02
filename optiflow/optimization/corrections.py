from __future__ import annotations

from optiflow.models.scoring import EfficiencyTriple


def apply_element_position_correction(atomic: EfficiencyTriple, j: int) -> EfficiencyTriple:
  """ПЭН: sequential vertical position j on the current form (1-based)."""
  decay = 0.998 ** max(0, int(j) - 1)
  return EfficiencyTriple(
    atomic.potency * decay,
    atomic.operativeness * decay,
    atomic.resource_saving * decay,
  )


def apply_form_step_correction(value: EfficiencyTriple, i: int) -> EfficiencyTriple:
  """Wizard step fatigue: chronological form index i (1-based)."""
  decay = 0.995 ** max(0, int(i) - 1)
  return EfficiencyTriple(
    value.potency * decay,
    value.operativeness * decay,
    value.resource_saving * decay,
  )
