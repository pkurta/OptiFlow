from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict

class DataType(Enum):
    BOOLEAN = "BOOLEAN"
    UNSIGNED = "UNSIGNED"
    TEXT = "TEXT"

class ControlType(Enum):
    TEXTBOX = "TEXTBOX"
    DROPDOWNLIST = "DROPDOWNLIST"
    CHECKBOX = "CHECKBOX"
    SPINNER = "SPINNER"
    SLIDER = "SLIDER"
    TEXTBOX_RO = "TEXTBOX_RO"

def _clip(val: float) -> float:
    return max(0.0, min(1.0, float(val)))

@dataclass(frozen=True)
class EfficiencyTriple:
    potency: float
    operativeness: float
    resource_saving: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "potency", _clip(self.potency))
        object.__setattr__(self, "operativeness", _clip(self.operativeness))
        object.__setattr__(self, "resource_saving", _clip(self.resource_saving))

    def __mul__(self, other: EfficiencyTriple) -> EfficiencyTriple:
        if not isinstance(other, EfficiencyTriple):
            return NotImplemented
        return EfficiencyTriple(
            self.potency * other.potency,
            self.operativeness * other.operativeness,
            self.resource_saving * other.resource_saving,
        )

    def __rmul__(self, other: EfficiencyTriple) -> EfficiencyTriple:
        return self.__mul__(other)

    @classmethod
    def identity(cls) -> EfficiencyTriple:
        return cls(1.0, 1.0, 1.0)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.potency, self.operativeness, self.resource_saving)

@dataclass
class FieldSpec:
    name: str
    data_type: DataType
    size: int

@dataclass
class LayoutElement:
    field_index: int
    control: ControlType
    position_index: int  # 1-based index j

@dataclass
class FormLayout:
    form_index: int  # 1-based index i
    elements: List[LayoutElement]

@dataclass
class InterfaceLayout:
    forms: List[FormLayout]
    fields: List[FieldSpec]
    
    @property
    def form_count(self) -> int:
        return len(self.forms)

    def controls_flat(self) -> List[ControlType]:
        result: List[ControlType] = []
        for form in self.forms:
            for element in form.elements:
                result.append(element.control)
        return result

def evaluate_form(element_count: int) -> EfficiencyTriple:
    count = max(0, int(element_count))
    return EfficiencyTriple(
        potency=0.999**count,
        operativeness=0.999**count,
        resource_saving=0.995**count,
    )

def partition_counts_to_form_indices(field_counts_per_form: List[int]) -> List[int]:
    form_indices: List[int] = []
    for form_i, count in enumerate(field_counts_per_form):
        form_indices.extend([form_i] * int(count))
    return form_indices

def build_interface_layout_from_partition(
    fields: List[FieldSpec],
    controls: List[ControlType],
    field_counts_per_form: List[int],
) -> InterfaceLayout:
    if len(controls) != len(fields):
        raise ValueError("controls must match fields length")
    d = len(fields)
    counts = [max(0, int(c)) for c in field_counts_per_form]
    if sum(counts) != d:
        raise ValueError(f"partition must sum to {d} fields, got {sum(counts)}")
    form_indices = partition_counts_to_form_indices(counts)
    return build_interface_layout(fields, controls, form_indices)

def build_interface_layout(
    fields: List[FieldSpec],
    controls: List[ControlType],
    form_indices: List[int],
) -> InterfaceLayout:
    forms_dict: Dict[int, List[Tuple[int, ControlType]]] = {}
    for field_idx, form_idx in enumerate(form_indices):
        if form_idx not in forms_dict:
            forms_dict[form_idx] = []
        forms_dict[form_idx].append((field_idx, controls[field_idx]))
    
    sorted_form_keys = sorted(forms_dict.keys())
    layout_forms: List[FormLayout] = []
    
    for dynamic_i, form_key in enumerate(sorted_form_keys, start=1):
        elements_list: List[LayoutElement] = []
        for dynamic_j, (f_idx, ctrl) in enumerate(forms_dict[form_key], start=1):
            elements_list.append(LayoutElement(
                field_index=f_idx,
                control=ctrl,
                position_index=dynamic_j  # Strict 1-based index j
            ))
        layout_forms.append(FormLayout(form_index=dynamic_i, elements=elements_list)) # Strict 1-based index i
        
    return InterfaceLayout(forms=layout_forms, fields=fields)

class FunctionRegistry:
    DEFAULT_FUNCTIONS: Dict[ControlType, str] = {
        ControlType.TEXTBOX: (
            "if dtype == DataType.TEXT:\n"
            "    if size < 6:\n"
            "        p, o, r = 0.92, 0.90, 0.85\n"
            "    else:\n"
            "        p, o, r = max(0.4, 0.92 - 0.05 * (size - 5)), 0.75, 0.80\n"
            "else:\n"
            "    p, o, r = 0.50, 0.50, 0.50\n"
        ),
        ControlType.DROPDOWNLIST: (
            "if dtype == DataType.TEXT:\n"
            "    if size > 7:\n"
            "        p, o, r = 0.88, 0.82, 0.78\n"
            "    else:\n"
            "        p, o, r = 0.65, 0.70, 0.85\n"
            "else:\n"
            "    p, o, r = 0.50, 0.50, 0.50\n"
        ),
        ControlType.CHECKBOX: (
            "if dtype == DataType.BOOLEAN:\n"
            "    p, o, r = 0.95, 0.95, 0.95\n"
            "else:\n"
            "    p, o, r = 0.50, 0.50, 0.50\n"
        ),
        ControlType.SPINNER: (
            "if dtype == DataType.UNSIGNED:\n"
            "    if size <= 3:\n"
            "        p, o, r = 0.85, 0.88, 0.82\n"
            "    else:\n"
            "        p, o, r = 0.72, 0.68, 0.80\n"
            "else:\n"
            "    p, o, r = 0.50, 0.50, 0.50\n"
        ),
        ControlType.SLIDER: (
            "if dtype == DataType.UNSIGNED:\n"
            "    if size <= 3:\n"
            "        p, o, r = 0.85, 0.88, 0.82\n"
            "    else:\n"
            "        p, o, r = 0.72, 0.68, 0.80\n"
            "else:\n"
            "    p, o, r = 0.50, 0.50, 0.50\n"
        ),
        ControlType.TEXTBOX_RO: (
            "if dtype == DataType.TEXT:\n"
            "    if size < 6:\n"
            "        p, o, r = 0.92, 0.90, 0.85\n"
            "    else:\n"
            "        p, o, r = max(0.4, 0.92 - 0.05 * (size - 5)), 0.75, 0.80\n"
            "else:\n"
            "    p, o, r = 0.50, 0.50, 0.50\n"
        ),
    }

    def __init__(self) -> None:
        self._code_by_control: Dict[ControlType, str] = {
            ctrl: self.DEFAULT_FUNCTIONS.get(ctrl, "p, o, r = 0.50, 0.50, 0.50")
            for ctrl in ControlType
        }

    def get_code_for_control(self, ctrl: ControlType) -> str:
        return self._code_by_control.get(ctrl, "p, o, r = 0.50, 0.50, 0.50")

    def update_code_for_control(self, ctrl: ControlType, code_str: str) -> bool:
        compile(code_str, f"<FunctionRegistry:{ctrl.name}>", "exec")
        self._code_by_control[ctrl] = code_str
        return True

    # Backward-compatible alias used by FunctionEditor in app.py.
    def set_code_for_control(self, ctrl: ControlType, code_str: str) -> bool:
        return self.update_code_for_control(ctrl, code_str)

    def _evaluate_empirical_fallback(self, cls: ControlType, dtype: DataType, size: int) -> EfficiencyTriple:
        if cls == ControlType.CHECKBOX and dtype == DataType.BOOLEAN:
            return EfficiencyTriple(0.95, 0.95, 0.95)
        if cls == ControlType.TEXTBOX and dtype == DataType.TEXT:
            if size < 6:
                return EfficiencyTriple(0.92, 0.90, 0.85)
            return EfficiencyTriple(max(0.4, 0.92 - 0.05 * (size - 5)), 0.75, 0.80)
        if cls == ControlType.DROPDOWNLIST and dtype == DataType.TEXT:
            if size > 7:
                return EfficiencyTriple(0.88, 0.82, 0.78)
            return EfficiencyTriple(0.65, 0.70, 0.85)
        if cls in (ControlType.SPINNER, ControlType.SLIDER) and dtype == DataType.UNSIGNED:
            if size <= 3:
                return EfficiencyTriple(0.85, 0.88, 0.82)
            return EfficiencyTriple(0.72, 0.68, 0.80)
        return EfficiencyTriple(0.50, 0.50, 0.50)

    def evaluate_atomic(self, cls: ControlType, dtype: DataType, size: int) -> EfficiencyTriple:
        code_str = self._code_by_control.get(cls)
        if code_str:
            local_ctx: Dict[str, object] = {
                "dtype": dtype,
                "size": int(size),
                "ControlType": ControlType,
                "DataType": DataType,
                "EfficiencyTriple": EfficiencyTriple,
                "max": max,
                "min": min,
            }
            try:
                exec(code_str, {"__builtins__": {}}, local_ctx)
                p = float(local_ctx["p"])  # type: ignore[arg-type]
                o = float(local_ctx["o"])  # type: ignore[arg-type]
                r = float(local_ctx["r"])  # type: ignore[arg-type]
                return EfficiencyTriple(p, o, r)
            except Exception:
                pass
        return self._evaluate_empirical_fallback(cls, dtype, size)