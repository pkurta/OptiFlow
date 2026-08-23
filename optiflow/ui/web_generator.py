from __future__ import annotations

from typing import List, Optional

from optiflow.models.scoring import ControlType, DataType, FieldSpec, InterfaceLayout


def _safe_name(field: FieldSpec) -> str:
  return field.name.replace('"', "&quot;")


def _dropdown_options(field: FieldSpec) -> str:
  """Placeholder options: count derived from field size when reasonable."""
  count = max(2, min(12, int(field.size) if field.size > 0 else 3))
  if field.data_type == DataType.TEXT and field.size <= 7:
    count = max(2, min(count, field.size if field.size >= 2 else 2))
  options = "".join(f'<option value="{i}">Вариант {i}</option>' for i in range(1, count + 1))
  return options


def _spinner_bounds(field: FieldSpec) -> tuple[int, int, int]:
  """min, max, step for number spinner based on UNSIGNED size (digits)."""
  digits = max(1, min(9, int(field.size) if field.size > 0 else 3))
  max_value = 10**digits - 1
  return 0, max_value, 1


def _slider_bounds(field: FieldSpec) -> tuple[int, int]:
  digits = max(1, min(6, int(field.size) if field.size > 0 else 2))
  return 0, 10**digits - 1


def control_to_html(control: ControlType, field: FieldSpec) -> str:
  name = _safe_name(field)
  ctrl = control.name if isinstance(control, ControlType) else str(control)

  # Core OptiFlow controls used by the optimizer.
  if control == ControlType.TEXTBOX or ctrl in ("INPUT", "TEXTBOX"):
    maxlen = max(1, int(field.size)) if field.size else 255
    return (
      f'<label class="field-label">{name}'
      f'<input type="text" name="{name}" maxlength="{maxlen}" '
      f'placeholder="до {maxlen} симв." /></label>'
    )
  if control == ControlType.TEXTBOX_RO:
    maxlen = max(1, int(field.size)) if field.size else 255
    return (
      f'<label class="field-label">{name}'
      f'<input type="text" name="{name}" maxlength="{maxlen}" '
      f'value="(только чтение)" readonly /></label>'
    )
  if control == ControlType.DROPDOWNLIST or ctrl in ("SELECT", "DROPDOWNLIST"):
    return (
      f'<label class="field-label">{name}'
      f'<select name="{name}">{_dropdown_options(field)}</select></label>'
    )
  if control == ControlType.CHECKBOX or ctrl in ("CHECKBOX", "TOGGLE"):
    return f'<label class="field-check"><input type="checkbox" name="{name}" /> {name}</label>'
  if control == ControlType.SPINNER or ctrl == "SPINNER":
    lo, hi, step = _spinner_bounds(field)
    return (
      f'<label class="field-label">{name}'
      f'<input type="number" name="{name}" min="{lo}" max="{hi}" step="{step}" '
      f'value="{lo}" /></label>'
    )
  if control == ControlType.SLIDER or ctrl == "SLIDER":
    lo, hi = _slider_bounds(field)
    mid = (lo + hi) // 2
    return (
      f'<label class="field-label">{name}'
      f'<input type="range" name="{name}" min="{lo}" max="{hi}" value="{mid}" '
      f'oninput="this.nextElementSibling.value=this.value" />'
      f'<output>{mid}</output></label>'
    )

  # Extended / alias controls (kept for compatibility with older HTML helpers).
  if ctrl == "TEXTAREA":
    return f'<label class="field-label">{name}<br/><textarea name="{name}" rows="4" cols="50"></textarea></label>'
  if ctrl == "PASSWORD":
    return f'<label class="field-label">{name}<input type="password" name="{name}" /></label>'
  if ctrl == "SEARCH":
    return f'<label class="field-label">{name}<input type="search" name="{name}" /></label>'
  if ctrl == "MASKED_INPUT":
    return (
      f'<label class="field-label">{name}'
      f'<input type="text" name="{name}" pattern="[0-9A-Za-z\\-]+" /></label>'
    )
  if ctrl == "RADIO":
    return (
      f'<fieldset><legend>{name}</legend>'
      f'<label><input type="radio" name="{name}" value="A" /> A</label> '
      f'<label><input type="radio" name="{name}" value="B" /> B</label></fieldset>'
    )
  if ctrl == "COMBOBOX":
    return (
      f'<label class="field-label">{name}'
      f'<input list="{name}_list" name="{name}" />'
      f'<datalist id="{name}_list">{_dropdown_options(field)}</datalist></label>'
    )
  if ctrl == "DATETIME_PICKER":
    return f'<label class="field-label">{name}<input type="datetime-local" name="{name}" /></label>'
  if ctrl == "COLOR_PICKER":
    return f'<label class="field-label">{name}<input type="color" name="{name}" /></label>'
  if ctrl in ("BUTTON", "TOGGLE_BUTTON"):
    return f'<button type="button">{name}</button>'
  if ctrl == "ICON_BUTTON":
    return f'<button type="button" title="{name}">●</button>'
  if ctrl == "FAB":
    return f'<button type="button" class="fab">{name}</button>'
  if ctrl == "TABLE":
    return (
      f'<h3>{name}</h3><table border="1">'
      f'<tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>'
    )
  if ctrl == "LIST":
    return f'<h3>{name}</h3><ul><li>Элемент 1</li><li>Элемент 2</li></ul>'
  if ctrl == "TREE_VIEW":
    return f'<h3>{name}</h3><ul><li>Узел 1<ul><li>Дочерний</li></ul></li><li>Узел 2</li></ul>'
  if ctrl == "GRID":
    return (
      f'<h3>{name}</h3>'
      f'<div class="grid"><div>1</div><div>2</div><div>3</div></div>'
    )
  if ctrl == "CHART":
    return f'<h3>{name}</h3><div class="placeholder">График</div>'
  if ctrl == "CAROUSEL":
    return f'<h3>{name}</h3><div class="placeholder">Карусель</div>'
  if ctrl == "RICH_TEXT_EDITOR":
    return f'<label class="field-label">{name}<br/><textarea name="{name}" rows="6" cols="60"></textarea></label>'

  # Unknown control: still show an interactive field so nothing "disappears".
  return (
    f'<label class="field-label">{name}'
    f'<input type="text" name="{name}" '
    f'data-control="{ctrl}" placeholder="[{ctrl}]" /></label>'
  )


def _render_form_step(form_index: int, total_forms: int, layout: InterfaceLayout) -> str:
  form = layout.forms[form_index]
  blocks: List[str] = []
  for element in form.elements:
    field = layout.fields[element.field_index]
    html = control_to_html(element.control, field)
    badge = f'<span class="ctrl-badge">{element.control.name}</span>'
    blocks.append(f'<div class="control">{badge}{html}</div>')
  body = "\n".join(blocks)
  is_first = form_index == 0
  is_last = form_index == total_forms - 1
  nav = '<div class="wizard-nav">'
  if not is_first:
    nav += '<button type="button" class="wizard-back">Назад</button>'
  if not is_last:
    nav += '<button type="button" class="wizard-next">Далее</button>'
  else:
    nav += '<button type="submit">Готово</button>'
  nav += "</div>"
  hidden = "" if form_index == 0 else ' style="display:none;"'
  return (
    f'<section class="wizard-step" data-step="{form_index}"{hidden}>'
    f'<h2>Шаг {form.form_index} из {total_forms}</h2>'
    f"{body}"
    f"{nav}"
    f"</section>"
  )


def generate_html_from_layout(layout: InterfaceLayout) -> str:
  total = layout.form_count
  steps = "\n".join(_render_form_step(i, total, layout) for i in range(total))
  html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OptiFlow — сгенерированный интерфейс</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #1a1a1a; }}
    h1 {{ font-size: 1.25rem; margin-bottom: 8px; }}
    .control {{ margin: 14px 0; padding: 10px 12px; background: #f7f8fa; border-radius: 8px; }}
    .ctrl-badge {{
      display: inline-block; font-size: 11px; color: #5b6b7c; background: #e8eef4;
      padding: 2px 8px; border-radius: 999px; margin-bottom: 8px;
    }}
    .field-label {{ display: flex; flex-direction: column; align-items: stretch; gap: 6px; font-weight: 500; }}
    .field-check {{ display: inline-flex; align-items: center; gap: 8px; }}
    input[type="text"], input[type="number"], input[type="password"], input[type="search"],
    input[type="datetime-local"], select, textarea {{
      padding: 8px 10px; border: 1px solid #c5ced8; border-radius: 6px; font-size: 14px;
    }}
    input[type="range"] {{ width: 100%; }}
    output {{ font-variant-numeric: tabular-nums; color: #445; }}
    .wizard-nav {{ margin-top: 20px; display: flex; gap: 12px; }}
    .wizard-step h2 {{ margin-bottom: 16px; color: #333; font-size: 1.05rem; }}
    button {{ padding: 8px 16px; cursor: pointer; border-radius: 6px; border: 1px solid #b0b8c4; background: #fff; }}
    button[type="submit"] {{ background: #2f6fed; color: #fff; border-color: #2f6fed; }}
    .fab {{ border-radius: 24px; padding: 12px 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    .placeholder {{ padding: 16px; background: #eee; border-radius: 6px; color: #666; }}
  </style>
</head>
<body>
  <h1>Сгенерированный UI (мастер из {total} форм)</h1>
  <form id="optiflow-wizard">
    {steps}
  </form>
  <script>
    (function () {{
      const steps = Array.from(document.querySelectorAll('.wizard-step'));
      let current = 0;
      function show(idx) {{
        steps.forEach((s, i) => {{ s.style.display = i === idx ? '' : 'none'; }});
        current = idx;
      }}
      document.querySelectorAll('.wizard-next').forEach(btn => {{
        btn.addEventListener('click', () => {{
          if (current < steps.length - 1) show(current + 1);
        }});
      }});
      document.querySelectorAll('.wizard-back').forEach(btn => {{
        btn.addEventListener('click', () => {{
          if (current > 0) show(current - 1);
        }});
      }});
    }})()
  </script>
</body>
</html>
"""
  return html


def generate_html(
  fields: List[FieldSpec],
  controls: List[ControlType],
  form_indices: Optional[List[int]] = None,
) -> str:
  from optiflow.models.scoring import build_interface_layout

  if form_indices is None:
    form_indices = [0] * len(fields)
  layout = build_interface_layout(fields, controls, form_indices)
  return generate_html_from_layout(layout)
