from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Tuple

from optiflow.models.scoring import FieldSpec, InterfaceLayout
from optiflow.optimization.algorithms import CriterionWeights
from optiflow.ui.run_results import AlgorithmRunSummary, summaries_with_layouts

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
    f"  - {field.name}: тип={field.data_type.name}, размер={field.size}"
    for field in fields
  ]
  ranked = summaries_with_layouts(list(summaries))
  layout_blocks: List[str] = []
  for rank, item in enumerate(ranked, start=1):
    assert item.layout is not None and item.triple is not None
    layout_blocks.append(
      "\n".join(
        [
          f"{rank}. {item.label} [{item.key}]",
          f"   F={item.fitness:.4f}, P={item.triple.potency:.4f}, "
          f"O={item.triple.operativeness:.4f}, R={item.triple.resource_saving:.4f}",
          f"   Экранов мастера: {item.form_count}, шагов истории: {item.history_steps}",
          "   Структура интерфейса:",
          describe_layout(item.layout),
        ]
      )
    )

  failed = [item for item in summaries if item.layout is None]
  failed_lines = [
    f"  - {item.label} [{item.key}]: layout отсутствует, шагов истории={item.history_steps}"
    for item in failed
  ]

  return f"""Ты — эксперт по UX, эргономике интерфейсов и метаэвристической оптимизации.
Проанализируй результаты прогона OptiFlow (версия {optiflow_version}) и дай **детальную интерпретацию на русском языке**.

## Контекст модели OptiFlow
- Задача: подобрать тип UI-контрола для каждого поля и разбить поля на N экранов мастера (wizard).
- Скалярная пригодность: F = w₁·P + w₂·O + w₃·R, где P — результативность, O — оперативность, R — ресурсоэкономность.
- Общая эффективность интерфейса считается **мультипликативно** по формам и элементам (накопление когнитивной нагрузки).
- Алгоритмы сравниваются на одной постановке; лучше ориентироваться на F при заданных весах.

## Постановка задачи
Поля ({len(fields)} шт.):
{chr(10).join(field_lines) if field_lines else "  (нет полей)"}

Макс. число экранов мастера (N): {max_forms}
Веса: w₁={w1:.3f}, w₂={w2:.3f}, w₃={w3:.3f}

Статус прогона: {"прерван пользователем" if cancelled else "завершён полностью"}
{f"Предупреждение: {warning}" if warning else ""}

## Сводный отчёт (детерминированный)
{report_text}

## Топ алгоритмов с layout (по F, убывание)
{chr(10).join(layout_blocks) if layout_blocks else "  Нет успешных layout."}

## Алгоритмы без layout
{chr(10).join(failed_lines) if failed_lines else "  Все алгоритмы вернули layout."}

## Что нужно в ответе
1. **Краткое резюме** (3–5 предложений): какой алгоритм лидирует и почему при текущих весах.
2. **Сравнение алгоритмов**: сильные и слабые стороны топ-3 по F; кто даёт больше экранов vs компактность.
3. **Анализ UX решений**: какие контролы выбраны для полей, уместность для типов данных, эффект многошагового мастера.
4. **Интерпретация P, O, R**: что означают полученные значения для пользователя в контексте весов w₁/w₂/w₃.
5. **Практическая рекомендация**: какой результат экспортировать в продакшен и нужно ли менять постановку (поля, N, веса).
6. **Ограничения анализа**: что нельзя вывести из чисел без пользовательского тестирования.

Пиши структурированно с подзаголовками. Не выдумывай числа — используй только данные выше. Если данных недостаточно, явно укажи это.
"""


def request_gemini_flash(
  api_key: str,
  prompt: str,
  *,
  model: str = DEFAULT_GEMINI_MODEL,
  timeout_s: float = 120.0,
) -> str:
  key = api_key.strip()
  if not key:
    raise ValueError("API-ключ Gemini не задан.")

  url = GEMINI_API_URL.format(model=model) + f"?key={key}"
  generation_config: dict = {
    "temperature": 0.35,
    "maxOutputTokens": 8192,
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
