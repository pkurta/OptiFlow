from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Tuple

from optiflow.models.scoring import DataType, FieldSpec, InterfaceLayout
from optiflow.optimization.algorithms import CriterionWeights
from optiflow.optimization.runner import SUITE_STEPS
from optiflow.ui.run_results import (
  ALGORITHM_META,
  AlgorithmRunSummary,
  _format_duration,
  summaries_by_key,
  summaries_with_layouts,
)

# (подпись в UI, model id для API)
GEMINI_MODEL_CHOICES: Tuple[Tuple[str, str], ...] = (
  ("Gemini Flash 3.7", "gemini-3.7-flash"),
  ("Gemini Flash-Lite 3.5", "gemini-3.5-flash-lite"),
)

DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
GEMINI_API_URL = (
  "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def _ssl_context() -> ssl.SSLContext:
  """CA bundle for macOS/python.org builds without system certs in SSL."""
  try:
    import certifi

    return ssl.create_default_context(cafile=certifi.where())
  except ImportError:
    return ssl.create_default_context()


def gemini_model_label(model_id: str) -> str:
  for label, mid in GEMINI_MODEL_CHOICES:
    if mid == model_id:
      return label
  return model_id


def describe_layout(layout: InterfaceLayout) -> str:
  lines: List[str] = []
  for form in layout.forms:
    parts: List[str] = []
    for element in form.elements:
      field = layout.fields[element.field_index]
      parts.append(
        f"{field.name} (тип {field.data_type.name}, размер {field.size}) "
        f"→ контрол {element.control.name}, позиция j={element.position_index}"
      )
    lines.append(f"Экран {form.form_index}/{layout.form_count}: " + "; ".join(parts))
  return "\n".join(lines) if lines else "  (пустой layout)"


def _allowed_controls_hint(field: FieldSpec) -> str:
  if field.data_type == DataType.BOOLEAN:
    return "CHECKBOX"
  if field.data_type == DataType.UNSIGNED:
    return "SPINNER | SLIDER"
  return "TEXTBOX | DROPDOWNLIST"


def _build_algorithm_search_blocks(summaries: Sequence[AlgorithmRunSummary]) -> str:
  by_key = summaries_by_key(summaries)
  blocks: List[str] = []
  for step, (key, label) in enumerate(SUITE_STEPS, start=1):
    item = by_key.get(key)
    if item is None:
      continue
    idea, search = ALGORITHM_META.get(key, ("—", "—"))
    block = [
      f"### {step}. {label} [{key}]",
      f"Идея алгоритма: {idea}",
      f"Как ищет решение (из документации OptiFlow): {search}",
      f"Запускался: {'да' if item.ran else 'нет (прогон прерван до этого шага)'}",
      f"Время работы: {_format_duration(item.elapsed_s)}",
      f"Шагов в истории сходимости: {item.history_steps}",
      f"Best score внутри алгоритма: {item.algo_best_score:.4f}",
    ]
    if item.layout is None:
      block.append("Итоговый layout: не получен.")
    else:
      assert item.triple is not None
      block.extend(
        [
          (
            f"Итоговые метрики layout: F={item.fitness:.4f}, "
            f"P={item.triple.potency:.4f}, O={item.triple.operativeness:.4f}, "
            f"R={item.triple.resource_saving:.4f}"
          ),
          f"Экранов мастера: {item.form_count}",
          f"Выбранные контролы (порядок полей): {', '.join(c.name for c in item.layout.controls_flat())}",
          "Структура найденного решения (экран → поле → контрол → позиция j):",
          describe_layout(item.layout),
        ]
      )
    blocks.append("\n".join(block))
  return "\n\n".join(blocks) if blocks else "  Нет данных по алгоритмам."


def build_interpretation_prompt(
  *,
  summaries: Sequence[AlgorithmRunSummary],
  report_text: str,
  fields: Sequence[FieldSpec],
  max_forms: int,
  weights: CriterionWeights,
  cancelled: bool,
  warning: Optional[str],
  optiflow_version: str,
) -> str:
  w1, w2, w3 = weights.as_tuple()
  field_lines = [
    (
      f"  - {field.name}: тип={field.data_type.name}, размер={field.size}, "
      f"допустимые контролы: {_allowed_controls_hint(field)}"
    )
    for field in fields
  ]
  ranked = summaries_with_layouts(list(summaries))
  leader_block = ""
  if ranked:
    leader = ranked[0]
    assert leader.layout is not None and leader.triple is not None
    leader_block = (
      f"Лидер по F: {leader.label} [{leader.key}] — F={leader.fitness:.4f}, "
      f"время {_format_duration(leader.elapsed_s)}, экранов={leader.form_count}."
    )

  failed = [item for item in summaries if item.ran and item.layout is None]
  failed_lines = [
    f"  - {item.label} [{item.key}]: layout отсутствует; "
    f"время {_format_duration(item.elapsed_s)}; шагов истории={item.history_steps}"
    for item in failed
  ]

  search_blocks = _build_algorithm_search_blocks(summaries)

  return f"""Ты — эксперт по UX, эргономике интерфейсов, эволюционным и метаэвристическим методам оптимизации.
Проанализируй результаты прогона OptiFlow (версия {optiflow_version}) и подготовь **развёрнутую интерпретацию на русском языке**
(ориентир: не менее 1500–2500 слов, если данных достаточно).

Твоя главная задача — **объяснить человеку без глубокого ML-бэкграунда**, как каждый алгоритм **искал** решение,
что именно перебирал/эволюционировал, почему пришёл к своему layout, и чем стратегии поиска отличаются друг от друга.

## Контекст модели OptiFlow
- **Постановка:** для каждого поля выбрать UI-контрол и распределить поля по 1…N экранам мастера (wizard).
- **Хромосома решения:** D генов (индекс контрола для каждого поля) + N весов (сколько полей на каждом экране).
- **Оценка:** для каждого контрола считаются P, O, R; применяются поправки на позицию j на экране и номер экрана i;
  итог — **мультипликативное** произведение (модель накопления когнитивной нагрузки).
- **Сравнение алгоритмов:** скаляр F = w₁·P + w₂·O + w₃·R при заданных весах.
- **Важно:** одинаковый F у разных алгоритмов может означать сходимость к эквивалентному или близкому layout
  в малом пространстве поиска — различай «одинаковый результат» и «одинаковый *путь* поиска».

## Постановка задачи
Поля ({len(fields)} шт.):
{chr(10).join(field_lines) if field_lines else "  (нет полей)"}

Макс. число экранов мастера (N): {max_forms}
Веса свёртки: w₁={w1:.3f} (результативность), w₂={w2:.3f} (оперативность), w₃={w3:.3f} (ресурсоэкономность)

Статус прогона: {"прерван пользователем" if cancelled else "завершён полностью"}
{f"Предупреждение системы: {warning}" if warning else ""}
{leader_block}

## Сводный детерминированный отчёт OptiFlow
{report_text}

## Карточка каждого алгоритма (данные для интерпретации процесса поиска)
{search_blocks}

## Алгоритмы без итогового layout
{chr(10).join(failed_lines) if failed_lines else "  У всех запущенных алгоритмов есть layout."}

---

## Структура ответа (обязательно соблюдай все разделы)

### 1. Резюме для заказчика (5–7 предложений)
Кто победил по F, насколько результаты близки, стоит ли доверять одному алгоритму.

### 2. Как устроено решение «на пальцах»
Объясни хромосому D+N, что такое layout, как из выбора контролов получается HTML-мастер.
Приведи **конкретный пример** по лидирующему алгоритму (какие контролы на каких экранах).

### 3. Как считается качество «на пальцах»
Цепочка: контрол → (P,O,R) → поправки позиции/экрана → произведение → F.
Свяжи с весами w₁/w₂/w₃ текущей задачи.

### 4. **Подробно: как искалось решение — по каждому алгоритму**
Для **каждого** алгоритма из карточек выше (в порядке запуска) напиши подраздел:

#### 4.X. [Название алгоритма]
- **Механизм поиска:** опиши простым языком, как алгоритм двигается по пространству решений
  (случайные пробы, популяция, градиентный шаг, феромоны, табу-список и т.д.).
- **Ход поиска (реконструкция):** по числу шагов истории, времени работы и метрикам **восстанови**
  правдоподобный narrative: старт → исследование → сходимость; был ли это быстрый жадный выбор
  или долгая эволюция; мог ли алгоритм застрять в локальном оптимуме.
- **Найденное решение:** расшифруй layout — почему выбраны именно эти контролы для типов данных
  (TEXT/UNSIGNED/BOOLEAN) и размеров полей.
- **Сильные/слабые стороны именно в этом прогоне:** скорость, F, число экранов, устойчивость.
- **Сравнение с лидером:** если F совпадает или близко — объясни, совпали ли **решения** или только **оценка**.

(Минимум 4–6 содержательных предложений на алгоритм; для топ-3 по F — не менее 8 предложений каждый.)

### 5. Сравнительная таблица стратегий поиска (текстом)
Сгруппируй алгоритмы: точный перебор / популяционные / локальные / стохастические.
Кому доверять как ground truth, кто даёт быстрый черновик, кто — баланс.

### 6. UX-интерпретация лучших layout
Уместность контролов, одно- vs многоэкранный мастер, риски для пользователя.

### 7. Практические рекомендации
Что экспортировать, что изменить в постановке (поля, N, веса, лимиты итераций/времени).

### 8. Ограничения и честные оговорки
Что нельзя утверждать без юзабилити-тестов; где ты реконструируешь процесс поиска, а не наблюдаешь его напрямую.

---

**Правила:**
- Пиши **только на русском**, структурированно, с подзаголовками и списками.
- **Не выдумывай числа** — все F, P, O, R, времена и шаги бери из данных выше.
- Если процесс поиска не наблюдается напрямую, явно помечай: «реконструкция по косвенным признакам».
- Особое внимание: объясни, **почему** при одинаковом F разные алгоритмы могли прийти разными путями
  или к визуально похожему интерфейсу.
"""


def request_gemini_flash(
  api_key: str,
  prompt: str,
  *,
  model: str = DEFAULT_GEMINI_MODEL,
  timeout_s: float = 180.0,
) -> str:
  key = api_key.strip()
  if not key:
    raise ValueError("API-ключ Gemini не задан.")

  url = GEMINI_API_URL.format(model=model) + f"?key={key}"
  generation_config: dict = {
    "temperature": 0.4,
    "maxOutputTokens": 16384,
  }

  payload = {
    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    "generationConfig": generation_config,
  }
  data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
  request = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
  )
  try:
    with urllib.request.urlopen(
      request, timeout=timeout_s, context=_ssl_context()
    ) as response:
      body = json.loads(response.read().decode("utf-8"))
  except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="replace")
    try:
      err_json = json.loads(detail)
      message = err_json.get("error", {}).get("message", detail)
    except json.JSONDecodeError:
      message = detail
    raise RuntimeError(f"Gemini API HTTP {exc.code}: {message}") from exc
  except urllib.error.URLError as exc:
    reason = str(exc.reason)
    if "CERTIFICATE_VERIFY_FAILED" in reason:
      raise RuntimeError(
        "Ошибка проверки SSL-сертификата. Выполните: pip install certifi "
        "и перезапустите OptiFlow (или запустите Install Certificates.command "
        "из каталога Python на macOS)."
      ) from exc
    raise RuntimeError(f"Сеть недоступна: {exc.reason}") from exc

  candidates = body.get("candidates") or []
  if not candidates:
    block_reason = (body.get("promptFeedback") or {}).get("blockReason")
    if block_reason:
      raise RuntimeError(f"Запрос заблокирован Gemini: {block_reason}")
    raise RuntimeError("Gemini вернул пустой ответ.")

  parts = (candidates[0].get("content") or {}).get("parts") or []
  texts = [str(part.get("text", "")) for part in parts if part.get("text")]
  result = "\n".join(texts).strip()
  if not result:
    raise RuntimeError("Gemini не вернул текст интерпретации.")
  return result
