from __future__ import annotations

from typing import List

from optiflow.models.scoring import ControlType, DataType, FieldSpec


def control_to_html(control: ControlType, field: FieldSpec) -> str:
    name = field.name
    if control == ControlType.INPUT:
        return f'<label>{name}: <input type="text" name="{name}" /></label>'
    if control == ControlType.TEXTAREA:
        return f'<label>{name}:<br/><textarea name="{name}" rows="4" cols="50"></textarea></label>'
    if control == ControlType.PASSWORD:
        return f'<label>{name}: <input type="password" name="{name}" /></label>'
    if control == ControlType.SEARCH:
        return f'<label>{name}: <input type="search" name="{name}" /></label>'
    if control == ControlType.MASKED_INPUT:
        return f'<label>{name}: <input type="text" name="{name}" pattern="[0-9A-Za-z\-]+" /></label>'
    if control == ControlType.CHECKBOX:
        return f'<label><input type="checkbox" name="{name}" /> {name}</label>'
    if control == ControlType.RADIO:
        return f'<fieldset><legend>{name}</legend><label><input type="radio" name="{name}" /> A</label> <label><input type="radio" name="{name}" /> B</label></fieldset>'
    if control == ControlType.TOGGLE:
        return f'<label>{name}: <input type="checkbox" name="{name}" /></label>'
    if control == ControlType.SELECT:
        return f'<label>{name}: <select name="{name}"><option>Option 1</option><option>Option 2</option></select></label>'
    if control == ControlType.COMBOBOX:
        return f'<label>{name}: <input list="{name}_list" name="{name}" /><datalist id="{name}_list"><option>Option 1</option><option>Option 2</option></datalist></label>'
    if control == ControlType.SLIDER:
        return f'<label>{name}: <input type="range" name="{name}" min="0" max="100" /></label>'
    if control == ControlType.DATETIME_PICKER:
        return f'<label>{name}: <input type="datetime-local" name="{name}" /></label>'
    if control == ControlType.COLOR_PICKER:
        return f'<label>{name}: <input type="color" name="{name}" /></label>'
    if control == ControlType.BUTTON:
        return f'<button>{name}</button>'
    if control == ControlType.ICON_BUTTON:
        return f'<button title="{name}">🔘</button>'
    if control == ControlType.FAB:
        return f'<button style="border-radius: 24px; padding: 12px 18px;">{name}</button>'
    if control == ControlType.TOGGLE_BUTTON:
        return f'<button aria-pressed="false">{name}</button>'
    if control == ControlType.TABLE:
        return f'<h3>{name}</h3><table border="1"><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>'
    if control == ControlType.LIST:
        return f'<h3>{name}</h3><ul><li>Item 1</li><li>Item 2</li></ul>'
    if control == ControlType.TREE_VIEW:
        return f'<h3>{name}</h3><ul><li>Node 1<ul><li>Child</li></ul></li><li>Node 2</li></ul>'
    if control == ControlType.GRID:
        return f'<h3>{name}</h3><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;"><div>1</div><div>2</div><div>3</div></div>'
    if control == ControlType.CHART:
        return f'<h3>{name}</h3><div>Chart placeholder</div>'
    if control == ControlType.CAROUSEL:
        return f'<h3>{name}</h3><div>Carousel placeholder</div>'
    if control == ControlType.RICH_TEXT_EDITOR:
        return f'<label>{name}:<br/><textarea name="{name}" rows="6" cols="60"></textarea></label>'
    return f'<div>{name}</div>'


def generate_html(fields: List[FieldSpec], controls: List[ControlType]) -> str:
    blocks = [control_to_html(ctrl, field) for ctrl, field in zip(controls, fields)]
    body = "\n".join(f'<div class="control">{b}</div>' for b in blocks)
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>OptiFlow Generated UI</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    .control {{ margin: 12px 0; }}
    label {{ display: inline-flex; align-items: center; gap: 8px; }}
  </style>
  </head>
  <body>
    <h1>Generated UI</h1>
    {body}
  </body>
  </html>
"""
    return html


