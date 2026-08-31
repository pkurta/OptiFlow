"""JSON snapshot of a synthesized wizard InterfaceLayout for repeatable experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from optiflow.models.scoring import (
  ControlType,
  DataType,
  FieldSpec,
  FormLayout,
  InterfaceLayout,
  LayoutElement,
)

INTERFACE_LAYOUT_FORMAT = "optiflow-interface-layout"
INTERFACE_LAYOUT_VERSION = 1
LOADED_LAYOUT_KEY = "Loaded"


def fields_to_json(fields: List[FieldSpec]) -> List[Dict[str, Any]]:
  return [
    {
      "name": field.name,
      "data_type": field.data_type.name,
      "size": int(field.size),
    }
    for field in fields
  ]


def fields_from_json(items: Any) -> List[FieldSpec]:
  if not isinstance(items, list) or not items:
    raise ValueError("В файле отсутствует непустой массив fields.")
  fields: List[FieldSpec] = []
  for index, item in enumerate(items):
    if not isinstance(item, dict):
      raise ValueError(f"Поле #{index + 1} должно быть объектом.")
    name = str(item.get("name", "")).strip() or f"Field{index + 1}"
    try:
      dtype = DataType[str(item.get("data_type", "TEXT"))]
    except KeyError as exc:
      raise ValueError(f"Неизвестный тип данных у поля {name!r}: {item.get('data_type')!r}.") from exc
    size = max(1, int(item.get("size", 1)))
    if dtype == DataType.BOOLEAN:
      size = 1
    fields.append(FieldSpec(name=name, data_type=dtype, size=size))
  return fields


def interface_layout_to_payload(
  layout: InterfaceLayout,
  *,
  optiflow_version: str,
  algorithm_key: str = "",
  algorithm_label: str = "",
  weight_preset: str = "",
  weights: Optional[Tuple[float, float, float]] = None,
  max_forms: Optional[int] = None,
  potency: Optional[float] = None,
  operativeness: Optional[float] = None,
  resource_saving: Optional[float] = None,
  fitness: Optional[float] = None,
) -> Dict[str, Any]:
  payload: Dict[str, Any] = {
    "format": INTERFACE_LAYOUT_FORMAT,
    "version": INTERFACE_LAYOUT_VERSION,
    "optiflow_version": optiflow_version,
    "algorithm": algorithm_key,
    "algorithm_label": algorithm_label,
    "max_forms": int(max_forms if max_forms is not None else layout.form_count),
    "fields": fields_to_json(layout.fields),
    "forms": [
      {
        "form_index": int(form.form_index),
        "elements": [
          {
            "field_index": int(element.field_index),
            "control": element.control.name,
            "position_index": int(element.position_index),
          }
          for element in form.elements
        ],
      }
      for form in layout.forms
    ],
  }
  if weight_preset:
    payload["weight_preset"] = weight_preset
  if weights is not None:
    payload["weights"] = {
      "potency": float(weights[0]),
      "operativeness": float(weights[1]),
      "resource_saving": float(weights[2]),
    }
  if any(value is not None for value in (potency, operativeness, resource_saving, fitness)):
    payload["metrics"] = {
      "potency": potency,
      "operativeness": operativeness,
      "resource_saving": resource_saving,
      "fitness": fitness,
    }
  return payload


def interface_layout_from_payload(data: Dict[str, Any]) -> InterfaceLayout:
  if data.get("format") != INTERFACE_LAYOUT_FORMAT:
    raise ValueError(
      f"Неизвестный формат файла: {data.get('format')!r}. "
      f"Ожидается {INTERFACE_LAYOUT_FORMAT!r}."
    )
  version = int(data.get("version", 0))
  if version != INTERFACE_LAYOUT_VERSION:
    raise ValueError(
      f"Неподдерживаемая версия интерфейса: {version}. "
      f"Ожидается {INTERFACE_LAYOUT_VERSION}."
    )
  fields = fields_from_json(data.get("fields"))
  forms_raw = data.get("forms")
  if not isinstance(forms_raw, list) or not forms_raw:
    raise ValueError("В файле отсутствует непустой массив forms.")
  seen_fields: set[int] = set()
  forms: List[FormLayout] = []
  for form_pos, form_item in enumerate(forms_raw):
    if not isinstance(form_item, dict):
      raise ValueError(f"Экран #{form_pos + 1} должен быть объектом.")
    elements_raw = form_item.get("elements")
    if not isinstance(elements_raw, list):
      raise ValueError(f"У экрана #{form_pos + 1} отсутствует массив elements.")
    elements: List[LayoutElement] = []
    for el_pos, el_item in enumerate(elements_raw):
      if not isinstance(el_item, dict):
        raise ValueError(f"Элемент #{el_pos + 1} экрана #{form_pos + 1} должен быть объектом.")
      field_index = int(el_item.get("field_index", -1))
      if field_index < 0 or field_index >= len(fields):
        raise ValueError(
          f"field_index={field_index} выходит за пределы списка полей (0…{len(fields) - 1})."
        )
      if field_index in seen_fields:
        raise ValueError(f"Поле с индексом {field_index} встречается в layout дважды.")
      seen_fields.add(field_index)
      try:
        control = ControlType[str(el_item.get("control", ""))]
      except KeyError as exc:
        raise ValueError(f"Неизвестный тип контрола: {el_item.get('control')!r}.") from exc
      position_index = max(1, int(el_item.get("position_index", el_pos + 1)))
      elements.append(
        LayoutElement(
          field_index=field_index,
          control=control,
          position_index=position_index,
        )
      )
    form_index = int(form_item.get("form_index", form_pos + 1))
    forms.append(FormLayout(form_index=form_index, elements=elements))
  if len(seen_fields) != len(fields):
    missing = sorted(set(range(len(fields))) - seen_fields)
    raise ValueError(f"Не все поля размещены на экранах мастера: пропущены индексы {missing}.")
  forms.sort(key=lambda form: form.form_index)
  for index, form in enumerate(forms, start=1):
    form.form_index = index
    for position, element in enumerate(form.elements, start=1):
      element.position_index = position
  return InterfaceLayout(forms=forms, fields=fields)


def optional_weights_from_payload(data: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
  raw = data.get("weights")
  if not isinstance(raw, dict):
    return None
  return (
    float(raw.get("potency", 0.0)),
    float(raw.get("operativeness", 0.0)),
    float(raw.get("resource_saving", 0.0)),
  )
