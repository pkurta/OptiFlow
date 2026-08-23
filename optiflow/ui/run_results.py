from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from optiflow.models.scoring import EfficiencyTriple, FieldSpec, FunctionRegistry, InterfaceLayout
from optiflow.optimization.algorithms import CriterionWeights, calculate_fitness, compute_total_efficiency
from optiflow.optimization.runner import SUITE_STEPS

# key → (идея алгоритма, как ищет решение)
ALGORITHM_META: Dict[str, Tuple[str, str]] = {
  "NSGA-II": (
    "Многокритериальный генетический алгоритм: одновременно оптимизирует P, O и R, "
    "сохраняя набор недоминируемых решений (фронт Парето).",
    "Эволюционирует популяцию хромосом; лучшее по F(w) из первого фронта → layout.",
  ),
  "BruteForce": (
    "Полный перебор всех комбинаций контролов и разбиений полей по экранам.",
    "Перебирает каждый вариант; гарантирует глобальный максимум F, если пространство "
    "не превышает лимит (~50 000 комбинаций).",
  ),
  "GA": (
    "Классический однокритериальный GA: максимизирует скалярную пригодность F = w₁P + w₂O + w₃R.",
    "Популяция + отбор + кроссовер + мутация над хромосомой длины D + N.",
  ),
  "HillClimb": (
    "Локальный поиск: улучшает текущее решение малыми шагами по каждой координате.",
    "Стартует со случайного layout; принимает изменение, если F растёт; застревает в локальном максимуме.",
  ),
  "PSO": (
    "Рой частиц: «частицы» движутся в пространстве решений с учётом личного и глобального лучшего.",
    "Итеративно обновляет позиции вектора D + N; лучшая позиция → layout.",
  ),
  "Random": (
    "Случайный поиск: на каждом шаге пробует новую случайную комбинацию.",
    "Запоминает лучший из всех просмотренных layout за заданное число итераций.",
  ),
  "Greedy": (
    "Жадный подбор: для каждого поля по очереди выбирает контрол с максимальным F при фиксированных остальных.",
    "Быстро, но не гарантирует глобальный оптимум — локально оптимальный выбор на каждом шаге.",
  ),
  "SA": (
    "Имитация отжига: иногда принимает ухудшения, чтобы выйти из локальных максимумов.",
    "Температура постепенно падает; лучший layout сохраняется за весь прогон.",
  ),
  "Tabu": (
    "Поиск с запретами: не возвращается к недавно посещённым решениям (табу-список).",
    "На каждой итерации — лучший допустимый сосед; табу-tenure задаёт «память».",
  ),
  "ACO": (
    "Муравьиный алгоритм: «феромоны» на привлекательных комбинациях контролов и разбиений.",
    "Муравьи строят layout; удачные варианты получают больше феромона.",
  ),
}


@dataclass(frozen=True)
class AlgorithmRunSummary:
  key: str
  label: str
  layout: Optional[InterfaceLayout]
  triple: Optional[EfficiencyTriple]
  fitness: float
  form_count: int
  history_steps: int
  algo_best_score: float
  elapsed_s: Optional[float] = None
  ran: bool = True


def _format_duration(seconds: Optional[float]) -> str:
  if seconds is None:
    return "— (не запускался)"
  if seconds < 1.0:
    return f"{seconds * 1000:.0f} мс"
  if seconds < 60.0:
    return f"{seconds:.2f} с"
  minutes = int(seconds // 60)
  secs = seconds - minutes * 60
  return f"{minutes} мин {secs:.1f} с"


def describe_solution_layout(layout: InterfaceLayout) -> str:
  lines: List[str] = []
  for form in layout.forms:
    parts: List[str] = []
    for element in form.elements:
      field = layout.fields[element.field_index]
      parts.append(f"{field.name} → {element.control.name}")
    lines.append(f"    Экран {form.form_index}: {', '.join(parts)}")
  return "\n".join(lines) if lines else "    (пустой layout)"


def _plain_scoring_explanation(*, field_count: int, max_forms: int, w1: float, w2: float, w3: float) -> List[str]:
  return [
    "Как устроено решение и оценка (на пальцах)",
    "-" * 52,
    "  1. Решение — это «рецепт интерфейса»:",
    f"     • для каждого из {field_count} полей выбран тип контрола (TEXTBOX, SLIDER, CHECKBOX…);",
    f"     • поля распределены по 1…{max_forms} экранам мастера (wizard) в заданном порядке.",
    "  2. Хромосома алгоритма = D чисел (контрол на поле) + N весов (сколько полей на каждом экране).",
    "  3. Для каждого контрола считается «атомарная» тройка (P, O, R) по формулам из «Настройки задачи»",
    "     с учётом типа данных и размера поля.",
    "  4. Позиция на экране и номер экрана умножают вклад (штраф «усталости»): чем ниже поле",
    "     и чем дальше шаг мастера — тем хуже P, O, R.",
    "  5. Итоговая эффективность интерфейса = произведение всех поправленных P·O·R по формам",
    "     и элементам (мультипликативная модель когнитивной нагрузки).",
    (
      f"  6. Для сравнения алгоритмов всё сворачивается в одно число F = "
      f"{w1:.2f}·P + {w2:.2f}·O + {w3:.2f}·R — чем больше F, тем лучше при ваших весах."
    ),
    "",
    "Как представляется лучшее решение алгоритма",
    "-" * 52,
    "  Объект InterfaceLayout: список экранов (FormLayout), на каждом — элементы с привязкой",
    "  к полю, выбранному контролу и позиции j. Его можно просмотреть на вкладке «Визуализация»",
    "  как готовую HTML-страницу мастера.",
    "",
  ]


def build_algorithm_summaries(
  results: Dict[str, Dict[str, object]],
  registry: FunctionRegistry,
  weights: CriterionWeights,
) -> List[AlgorithmRunSummary]:
  summaries: List[AlgorithmRunSummary] = []
  for key, label in SUITE_STEPS:
    payload = results.get(key)
    if payload is None:
      summaries.append(
        AlgorithmRunSummary(
          key=key,
          label=label,
          layout=None,
          triple=None,
          fitness=0.0,
          form_count=0,
          history_steps=0,
          algo_best_score=0.0,
          elapsed_s=None,
          ran=False,
        )
      )
      continue
    layout = payload.get("best_layout")
    if layout is not None and not isinstance(layout, InterfaceLayout):
      layout = None
    history = payload.get("history") or []
    history_steps = len(history) if isinstance(history, Sequence) else 0
    algo_best = float(payload.get("best_score", 0.0) or 0.0)
    elapsed_raw = payload.get("elapsed_s")
    elapsed_s = float(elapsed_raw) if elapsed_raw is not None else None
    triple: Optional[EfficiencyTriple] = None
    fitness = 0.0
    form_count = 0
    if layout is not None:
      triple = compute_total_efficiency(layout, registry)
      fitness = calculate_fitness(triple, weights)
      form_count = layout.form_count
    summaries.append(
      AlgorithmRunSummary(
        key=key,
        label=label,
        layout=layout,
        triple=triple,
        fitness=fitness,
        form_count=form_count,
        history_steps=history_steps,
        algo_best_score=algo_best,
        elapsed_s=elapsed_s,
        ran=True,
      )
    )
  return summaries


def summaries_by_key(summaries: Sequence[AlgorithmRunSummary]) -> Dict[str, AlgorithmRunSummary]:
  return {item.key: item for item in summaries}


def summaries_with_layouts(
  summaries: Sequence[AlgorithmRunSummary],
) -> List[AlgorithmRunSummary]:
  return [item for item in summaries if item.layout is not None]


def format_optimization_report(
  summaries: Sequence[AlgorithmRunSummary],
  *,
  weights: CriterionWeights,
  field_count: int,
  max_forms: int,
  cancelled: bool,
  warning: Optional[str],
  optiflow_version: str,
  total_elapsed_s: Optional[float] = None,
  fields: Optional[Sequence[FieldSpec]] = None,
) -> str:
  w1, w2, w3 = weights.as_tuple()
  lines: List[str] = [
    "OptiFlow — отчёт по прогону оптимизации",
    "=" * 52,
    f"Версия приложения: {optiflow_version}",
    "",
    "Постановка задачи",
    "-" * 52,
    f"  Полей в форме: {field_count}",
    f"  Макс. число экранов мастера (N): {max_forms}",
    (
      f"  Веса свёртки F = w₁P + w₂O + w₃R: "
      f"w₁={w1:.3f}, w₂={w2:.3f}, w₃={w3:.3f} (Σ={w1 + w2 + w3:.3f})"
    ),
  ]
  if fields:
    lines.append("  Поля:")
    for field in fields:
      lines.append(f"    • {field.name}: {field.data_type.name}, размер={field.size}")
  lines.extend(["", "Статус прогона", "-" * 52])
  if cancelled:
    lines.append("  Прервано пользователем: да (сохранены частичные результаты).")
  else:
    lines.append("  Прервано пользователем: нет.")
  if warning:
    lines.append(f"  Предупреждение: {warning}")
  if total_elapsed_s is not None:
    lines.append(f"  Общее время набора алгоритмов: {_format_duration(total_elapsed_s)}")

  lines.append("")
  lines.extend(_plain_scoring_explanation(field_count=field_count, max_forms=max_forms, w1=w1, w2=w2, w3=w3))

  by_key = summaries_by_key(summaries)
  best_fitness = max(
    (item.fitness for item in summaries if item.layout is not None),
    default=0.0,
  )

  lines.extend(["Алгоритмы (порядок запуска)", "=" * 52])

  for step, (key, label) in enumerate(SUITE_STEPS, start=1):
    item = by_key.get(key)
    if item is None:
      continue
    idea, search = ALGORITHM_META.get(key, ("—", "—"))
    lines.extend(["", f"{step}. {label} [{key}]", "-" * 52])
    lines.append(f"  Идея: {idea}")
    lines.append(f"  Как ищет: {search}")
    lines.append(f"  Время работы: {_format_duration(item.elapsed_s)}")
    if not item.ran:
      lines.append("  Статус: не запускался (прогон прерван до этого шага).")
      continue
    if item.layout is None:
      lines.append("  Результат: layout не получен.")
      if item.history_steps:
        lines.append(f"  Шагов в истории: {item.history_steps}")
      continue
    assert item.triple is not None
    lines.extend(
      [
        f"  F = {item.fitness:.4f}  |  P = {item.triple.potency:.4f}  |  "
        f"O = {item.triple.operativeness:.4f}  |  R = {item.triple.resource_saving:.4f}",
        f"  Экранов мастера: {item.form_count}  |  Шагов истории: {item.history_steps}",
        f"  Контролы: {', '.join(c.name for c in item.layout.controls_flat())}",
        "  Структура решения:",
        describe_solution_layout(item.layout),
      ]
    )
    if best_fitness > 0:
      if item.fitness >= best_fitness - 1e-9:
        lines.append("  Сравнение: лидер по F среди успешных алгоритмов.")
      else:
        pct = item.fitness / best_fitness * 100.0
        lines.append(f"  Сравнение: {pct:.1f}% от лучшего F ({best_fitness:.4f}).")

  ranked = sorted(
    [item for item in summaries if item.ran and item.layout is not None],
    key=lambda x: x.fitness,
    reverse=True,
  )
  if ranked:
    leader = ranked[0]
    lines.extend(
      [
        "",
        "Итог",
        "=" * 52,
        f"  Лучший F: {leader.label} — F={leader.fitness:.4f}, "
        f"время {_format_duration(leader.elapsed_s)}, экранов={leader.form_count}.",
        "  Подробная визуализация — вкладка «Визуализация», интерпретация — «Интерпретация».",
      ]
    )
  return "\n".join(lines)
