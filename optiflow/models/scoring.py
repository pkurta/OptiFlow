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

# Эмпирические функции P/O/R по Курта П.А., Труды учебных заведений связи, 2024, Т.10, №6, С.99-110.
KURTA_2024_CONTROL_FUNCTIONS: Dict[str, str] = {
    "TEXTBOX": (
        "if dtype == DataType.TEXT:\n"
        "    # Рис. 9a — длина строки L_D = size (1…10)\n"
        "    tbl_p = {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00, 5: 0.83, 6: 0.59, 7: 0.51, 8: 0.33, 9: 0.31, 10: 0.25}\n"
        "    tbl_o = {1: 1.00, 2: 0.67, 3: 0.50, 4: 0.36, 5: 0.33, 6: 0.29, 7: 0.19, 8: 0.17, 9: 0.11, 10: 0.01}\n"
        "    tbl_r = {1: 1.00, 2: 0.94, 3: 0.89, 4: 0.82, 5: 0.75, 6: 0.66, 7: 0.57, 8: 0.47, 9: 0.36, 10: 0.25}\n"
        "    sz = max(1, min(10, int(size)))\n"
        "    if size <= 10:\n"
        "        p, o, r = tbl_p[sz], tbl_o[sz], tbl_r[sz]\n"
        "    else:\n"
        "        p = max(0.05, 0.25 - 0.02 * (size - 10))\n"
        "        o = 0.01\n"
        "        r = max(0.05, 0.25 - 0.02 * (size - 10))\n"
        "else:\n"
        "    p, o, r = 0.50, 0.50, 0.50\n"
    ),
    "DROPDOWNLIST": (
        "if dtype in (DataType.TEXT, DataType.UNSIGNED):\n"
        "    # Рис. 9b — число элементов M = size (монотонное убывание P, O, R)\n"
        "    tbl_p = {1: 1.00, 2: 1.00, 3: 0.83, 4: 0.77, 5: 0.72, 6: 0.56, 7: 0.50, 8: 0.40, 9: 0.36, 10: 0.33}\n"
        "    tbl_o = {1: 1.00, 2: 0.50, 3: 0.33, 4: 0.25, 5: 0.25, 6: 0.25, 7: 0.22, 8: 0.20, 9: 0.13, 10: 0.07}\n"
        "    tbl_r = {1: 1.00, 2: 0.91, 3: 0.83, 4: 0.73, 5: 0.63, 6: 0.52, 7: 0.39, 8: 0.27, 9: 0.13, 10: 0.01}\n"
        "    sz = max(1, min(10, int(size)))\n"
        "    if size <= 10:\n"
        "        p, o, r = tbl_p[sz], tbl_o[sz], tbl_r[sz]\n"
        "    else:\n"
        "        p = max(0.10, 0.33 - 0.02 * (size - 10))\n"
        "        o = max(0.02, 0.07 - 0.005 * (size - 10))\n"
        "        r = 0.01\n"
        "else:\n"
        "    p, o, r = 0.50, 0.50, 0.50\n"
    ),
    "CHECKBOX": (
        "if dtype == DataType.BOOLEAN:\n"
        "    # Таблица 4.4 — бинарный выбор\n"
        "    p, o, r = 0.99, 0.97, 0.96\n"
        "else:\n"
        "    p, o, r = 0.50, 0.50, 0.50\n"
    ),
    "SPINNER": (
        "if dtype == DataType.UNSIGNED:\n"
        "    # Двунаправленный счётчик: высокая эффективность при малом диапазоне Δ\n"
        "    if size <= 3:\n"
        "        p, o, r = 0.96, 0.90, 0.91\n"
        "    elif size <= 7:\n"
        "        p, o, r = 0.88, 0.65, 0.72\n"
        "    else:\n"
        "        p, o, r = 0.75, 0.35, 0.50\n"
        "else:\n"
        "    p, o, r = 0.50, 0.50, 0.50\n"
    ),
    "SLIDER": (
        "if dtype == DataType.UNSIGNED:\n"
        "    # Ползунок: стабильно высокая оперативность на средних и больших диапазонах\n"
        "    if size <= 5:\n"
        "        p, o, r = 0.92, 0.93, 0.90\n"
        "    elif size <= 15:\n"
        "        p, o, r = 0.88, 0.90, 0.82\n"
        "    else:\n"
        "        p, o, r = 0.84, 0.85, 0.75\n"
        "else:\n"
        "    p, o, r = 0.50, 0.50, 0.50\n"
    ),
    "TEXTBOX_RO": (
        "if dtype == DataType.TEXT:\n"
        "    # Статический вывод (восприятие информации)\n"
        "    p, o, r = 0.99, 0.95, 0.95\n"
        "else:\n"
        "    p, o, r = 0.50, 0.50, 0.50\n"
    ),
}

KURTA_2024_META_SOURCE = (
    "Курта П.А. Система статистического измерения атомарной эффективности "
    "графических элементов интерфейсов // Труды учебных заведений связи. "
    "2024. Т. 10. № 6. С. 99-110."
)


class FunctionRegistry:
    DEFAULT_FUNCTIONS: Dict[ControlType, str] = {
        ControlType[name]: code for name, code in KURTA_2024_CONTROL_FUNCTIONS.items()
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

    def export_functions(self) -> Dict[str, str]:
        return {ctrl.name: self.get_code_for_control(ctrl) for ctrl in ControlType}

    def import_functions(self, data: Dict[str, str]) -> None:
        for name, code in data.items():
            try:
                ctrl = ControlType[name]
            except KeyError:
                continue
            self.update_code_for_control(ctrl, str(code))

    @staticmethod
    def _exec_control_code(code_str: str, dtype: DataType, size: int) -> EfficiencyTriple | None:
        local_ctx: Dict[str, object] = {
            "dtype": dtype,
            "size": int(size),
            "ControlType": ControlType,
            "DataType": DataType,
            "EfficiencyTriple": EfficiencyTriple,
            "int": int,
            "float": float,
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
            return None

    def _evaluate_empirical_fallback(self, cls: ControlType, dtype: DataType, size: int) -> EfficiencyTriple:
        default_code = self.DEFAULT_FUNCTIONS.get(cls)
        if default_code:
            triple = self._exec_control_code(default_code, dtype, size)
            if triple is not None:
                return triple
        return EfficiencyTriple(0.50, 0.50, 0.50)

    def evaluate_atomic(self, cls: ControlType, dtype: DataType, size: int) -> EfficiencyTriple:
        code_str = self._code_by_control.get(cls)
        if code_str:
            triple = self._exec_control_code(code_str, dtype, size)
            if triple is not None:
                return triple
        return self._evaluate_empirical_fallback(cls, dtype, size)