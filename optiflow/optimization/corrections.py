from __future__ import annotations

from optiflow.models.scoring import EfficiencyTriple, InterfaceLayout

# Inv_3 (когнитивный предел Миллера): "∑ c_elem ≤ 7 ± 2" — мягкая граница 5 из
# рукописи не используется как ограничение (см. audit_glava_5_..., 1.5), только
# жёсткая: k_i <= MILLER_SOFT_LIMIT + MILLER_TOLERANCE.
MILLER_SOFT_LIMIT = 7
MILLER_TOLERANCE = 2
MILLER_HARD_LIMIT = MILLER_SOFT_LIMIT + MILLER_TOLERANCE  # 9

# Штраф за 1 «лишний» элемент на экране, при excess=1 (k_i=10).
DEFAULT_MILLER_PENALTY_WEIGHT = 0.01


def compute_miller_penalty(
  layout: InterfaceLayout,
  penalty_weight: float = DEFAULT_MILLER_PENALTY_WEIGHT,
  hard_limit: int = MILLER_HARD_LIMIT,
) -> float:
  """Inv_3: штраф за экраны, где число элементов k_i превышает hard_limit.

  Рост штрафа выбран квадратичным по excess = k_i - hard_limit, а не линейным:
  экран с одним «лишним» полем — небольшая эргономическая проблема, а экран,
  на котором элементов в разы больше нормы, качественно непригоден для
  использования. Линейный штраф (weight * excess) одинаково наказывает «+1» и
  «+10» в пересчёте на единицу превышения и слабо подавляет сильные нарушения;
  квадратичный (weight * excess**2) растёт быстрее самого нарушения и надёжнее
  отталкивает метаэвристики от таких решений, оставаясь равным 0 при excess<=0.
  """
  total_excess_sq = 0.0
  for form in layout.forms:
    excess = len(form.elements) - int(hard_limit)
    if excess > 0:
      total_excess_sq += excess * excess
  return max(0.0, float(penalty_weight)) * total_excess_sq


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
