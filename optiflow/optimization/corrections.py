from __future__ import annotations

import math
from typing import Optional

from optiflow.models.scoring import EfficiencyTriple, InterfaceLayout

# Inv_3 (когнитивный предел Миллера): "∑ c_elem ≤ 7 ± 2" — мягкая граница 5 из
# рукописи не используется как ограничение (см. audit_glava_5_..., 1.5), только
# жёсткая: k_i <= MILLER_SOFT_LIMIT + MILLER_TOLERANCE.
MILLER_SOFT_LIMIT = 7
MILLER_TOLERANCE = 2
MILLER_HARD_LIMIT = MILLER_SOFT_LIMIT + MILLER_TOLERANCE  # 9

# Штраф за 1 «лишний» элемент на экране, при excess=1 (k_i=10).
DEFAULT_MILLER_PENALTY_WEIGHT = 0.01


def is_miller_feasible(d: int, n: int, hard_limit: int = MILLER_HARD_LIMIT) -> bool:
  """True, если суммарная ёмкость N экранов (hard_limit * N) вмещает D полей.

  Граница включительна: D == hard_limit * N считается допустимым (существует
  разбиение, где каждый экран получает ровно hard_limit полей).
  """
  return int(d) <= int(hard_limit) * int(n)


def miller_feasibility_margin(d: int, n: int, hard_limit: int = MILLER_HARD_LIMIT) -> int:
  """hard_limit*N - D: запас ёмкости. Отрицательное значение — дефицит полей,
  на который суммарная ёмкость экранов меньше D (Inv_3 структурно невыполним)."""
  return int(hard_limit) * int(n) - int(d)


def miller_feasibility_warning(
  d: int,
  n: int,
  hard_limit: int = MILLER_HARD_LIMIT,
) -> Optional[str]:
  """None, если Inv_3 выполним при данных (D, N); иначе — готовый текст
  предупреждения с конкретными числами, для GUI и headless-путей.

  Не блокирует синтез: возвращаемая строка предназначена для показа
  предупреждением (как переполнение |Ω| у brute_force), а не для исключения —
  штраф compute_miller_penalty всё равно минимизирует превышение, только не
  может свести его к нулю при D > hard_limit * N.
  """
  if is_miller_feasible(d, n, hard_limit):
    return None
  capacity = int(hard_limit) * int(n)
  min_forms_needed = math.ceil(d / hard_limit) if hard_limit > 0 else d
  return (
    f"При D={d} полях и N={n} экранах предел Миллера (k_i<={hard_limit}) структурно "
    f"недостижим: суммарная ёмкость экранов {capacity} < D={d}. Требуется минимум "
    f"{min_forms_needed} экранов, чтобы Inv_3 стал выполним; штраф всё равно "
    f"минимизирует превышение ёмкости, но не может свести его к нулю."
  )


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
