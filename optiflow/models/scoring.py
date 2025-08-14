from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Tuple


class DataType(Enum):
    NUMBER = auto()
    SHORT_TEXT = auto()
    LONG_TEXT = auto()
    BOOLEAN = auto()
    DATE_TIME = auto()
    COLOR = auto()
    CATEGORY = auto()
    MULTI_CATEGORY = auto()
    RICH_TEXT = auto()
    LIST = auto()
    TREE = auto()
    GRID = auto()
    FILE = auto()
    IMAGE = auto()


class ControlType(Enum):
    # Text inputs
    INPUT = auto()
    TEXTAREA = auto()
    PASSWORD = auto()
    SEARCH = auto()
    MASKED_INPUT = auto()

    # Selection
    CHECKBOX = auto()
    RADIO = auto()
    TOGGLE = auto()
    SELECT = auto()
    COMBOBOX = auto()
    SLIDER = auto()
    DATETIME_PICKER = auto()
    COLOR_PICKER = auto()

    # Buttons
    BUTTON = auto()
    ICON_BUTTON = auto()
    FAB = auto()
    TOGGLE_BUTTON = auto()

    # Display
    TABLE = auto()
    LIST = auto()
    TREE_VIEW = auto()
    GRID = auto()
    CHART = auto()
    CAROUSEL = auto()

    # Special
    RICH_TEXT_EDITOR = auto()


ScoreTriple = Tuple[float, float, float]


def _clip(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _normalize(triple: ScoreTriple) -> ScoreTriple:
    return (_clip(triple[0]), _clip(triple[1]), _clip(triple[2]))


DEFAULT_FUNCTIONS: Dict[ControlType, str] = {
    ControlType.INPUT: (
        """
def score(t, n):
    # Simple single-line text input
    if t in ("NUMBER", "SHORT_TEXT", "SEARCH") and n <= 32:
        return 0.85, 0.90, 0.85
    if t == "NUMBER" and n <= 8:
        return 0.80, 0.92, 0.88
    if t == "SHORT_TEXT" and n <= 128:
        return 0.75, 0.85, 0.80
    return 0.40, 0.70, 0.95
        """
    ),
    ControlType.TEXTAREA: (
        """
def score(t, n):
    if t in ("LONG_TEXT", "RICH_TEXT") or n > 128:
        return 0.92, 0.70, 0.65
    if t == "SHORT_TEXT" and 64 <= n <= 256:
        return 0.80, 0.75, 0.70
    return 0.50, 0.65, 0.85
        """
    ),
    ControlType.PASSWORD: (
        """
def score(t, n):
    if t == "SHORT_TEXT" and n <= 64:
        return 0.78, 0.85, 0.82
    return 0.40, 0.70, 0.90
        """
    ),
    ControlType.SEARCH: (
        """
def score(t, n):
    if t in ("SHORT_TEXT", "SEARCH") and n <= 64:
        return 0.85, 0.92, 0.82
    return 0.45, 0.75, 0.90
        """
    ),
    ControlType.MASKED_INPUT: (
        """
def score(t, n):
    if t == "SHORT_TEXT" and n <= 32:
        return 0.82, 0.82, 0.80
    if t == "NUMBER" and n <= 16:
        return 0.80, 0.80, 0.82
    return 0.45, 0.70, 0.90
        """
    ),
    ControlType.CHECKBOX: (
        """
def score(t, n):
    if t == "BOOLEAN":
        return 0.95, 0.95, 0.95
    return 0.30, 0.85, 0.98
        """
    ),
    ControlType.RADIO: (
        """
def score(t, n):
    if t == "CATEGORY" and n <= 6:
        return 0.85, 0.82, 0.70
    return 0.45, 0.75, 0.88
        """
    ),
    ControlType.TOGGLE: (
        """
def score(t, n):
    if t == "BOOLEAN":
        return 0.92, 0.92, 0.92
    if t == "CATEGORY" and n == 2:
        return 0.80, 0.88, 0.85
    return 0.35, 0.80, 0.95
        """
    ),
    ControlType.SELECT: (
        """
def score(t, n):
    if t in ("CATEGORY", "MULTI_CATEGORY") and n >= 5:
        return 0.82, 0.78, 0.78
    return 0.60, 0.72, 0.85
        """
    ),
    ControlType.COMBOBOX: (
        """
def score(t, n):
    if t in ("CATEGORY", "SHORT_TEXT") and n >= 10:
        return 0.80, 0.80, 0.75
    return 0.60, 0.70, 0.85
        """
    ),
    ControlType.SLIDER: (
        """
def score(t, n):
    if t == "NUMBER" and n <= 4:
        return 0.88, 0.90, 0.80
    if t == "NUMBER" and n <= 8:
        return 0.80, 0.85, 0.82
    return 0.50, 0.70, 0.88
        """
    ),
    ControlType.DATETIME_PICKER: (
        """
def score(t, n):
    if t == "DATE_TIME":
        return 0.92, 0.88, 0.78
    return 0.40, 0.70, 0.90
        """
    ),
    ControlType.COLOR_PICKER: (
        """
def score(t, n):
    if t == "COLOR":
        return 0.95, 0.90, 0.80
    return 0.35, 0.70, 0.92
        """
    ),
    ControlType.BUTTON: (
        """
def score(t, n):
    return 0.70, 0.90, 0.85
        """
    ),
    ControlType.ICON_BUTTON: (
        """
def score(t, n):
    return 0.72, 0.92, 0.84
        """
    ),
    ControlType.FAB: (
        """
def score(t, n):
    return 0.68, 0.88, 0.80
        """
    ),
    ControlType.TOGGLE_BUTTON: (
        """
def score(t, n):
    if t == "BOOLEAN":
        return 0.88, 0.90, 0.86
    return 0.55, 0.80, 0.88
        """
    ),
    ControlType.TABLE: (
        """
def score(t, n):
    if t in ("GRID", "LIST") or n >= 50:
        return 0.88, 0.70, 0.60
    return 0.60, 0.65, 0.75
        """
    ),
    ControlType.LIST: (
        """
def score(t, n):
    if t in ("LIST", "CATEGORY") and n <= 100:
        return 0.80, 0.78, 0.70
    return 0.60, 0.70, 0.80
        """
    ),
    ControlType.TREE_VIEW: (
        """
def score(t, n):
    if t == "TREE":
        return 0.90, 0.70, 0.60
    return 0.50, 0.65, 0.78
        """
    ),
    ControlType.GRID: (
        """
def score(t, n):
    if t in ("GRID", "IMAGE"):
        return 0.82, 0.72, 0.68
    return 0.55, 0.68, 0.78
        """
    ),
    ControlType.CHART: (
        """
def score(t, n):
    if t in ("NUMBER", "GRID", "LIST") and n >= 20:
        return 0.85, 0.65, 0.60
    return 0.50, 0.60, 0.75
        """
    ),
    ControlType.CAROUSEL: (
        """
def score(t, n):
    if t in ("IMAGE", "FILE"):
        return 0.78, 0.70, 0.68
    return 0.45, 0.62, 0.80
        """
    ),
    ControlType.RICH_TEXT_EDITOR: (
        """
def score(t, n):
    if t in ("RICH_TEXT", "LONG_TEXT"):
        return 0.90, 0.65, 0.55
    return 0.40, 0.60, 0.75
        """
    ),
}


def data_type_to_allowed_controls(data_type: DataType) -> List[ControlType]:
    mapping: Dict[DataType, List[ControlType]] = {
        DataType.NUMBER: [ControlType.INPUT, ControlType.SLIDER, ControlType.SELECT, ControlType.COMBOBOX, ControlType.CHART],
        DataType.SHORT_TEXT: [ControlType.INPUT, ControlType.SEARCH, ControlType.PASSWORD, ControlType.MASKED_INPUT, ControlType.COMBOBOX],
        DataType.LONG_TEXT: [ControlType.TEXTAREA, ControlType.RICH_TEXT_EDITOR],
        DataType.BOOLEAN: [ControlType.CHECKBOX, ControlType.TOGGLE, ControlType.TOGGLE_BUTTON],
        DataType.DATE_TIME: [ControlType.DATETIME_PICKER, ControlType.INPUT],
        DataType.COLOR: [ControlType.COLOR_PICKER],
        DataType.CATEGORY: [ControlType.SELECT, ControlType.RADIO, ControlType.COMBOBOX, ControlType.LIST],
        DataType.MULTI_CATEGORY: [ControlType.SELECT, ControlType.COMBOBOX, ControlType.LIST],
        DataType.RICH_TEXT: [ControlType.RICH_TEXT_EDITOR, ControlType.TEXTAREA],
        DataType.LIST: [ControlType.LIST, ControlType.TABLE, ControlType.CHART],
        DataType.TREE: [ControlType.TREE_VIEW],
        DataType.GRID: [ControlType.TABLE, ControlType.GRID, ControlType.CHART],
        DataType.FILE: [ControlType.CAROUSEL, ControlType.TABLE],
        DataType.IMAGE: [ControlType.CAROUSEL, ControlType.GRID, ControlType.TABLE],
    }
    return mapping.get(data_type, [ControlType.INPUT])


def _safe_env() -> Dict[str, object]:
    # Safe environment for user functions
    import math

    return {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "round": round,
        "math": math,
    }


class FunctionRegistry:
    def __init__(self) -> None:
        self._code_by_control: Dict[ControlType, str] = dict(DEFAULT_FUNCTIONS)
        self._compiled_by_control: Dict[ControlType, Callable[[str, int], ScoreTriple]] = {}
        self._compile_all()

    def _compile_all(self) -> None:
        for control, code in self._code_by_control.items():
            self._compiled_by_control[control] = self._compile_function(code)

    def _compile_function(self, code: str) -> Callable[[str, int], ScoreTriple]:
        local_env: Dict[str, object] = {}
        global_env = _safe_env()
        exec(code, global_env, local_env)
        fn = local_env.get("score")
        if not callable(fn):
            raise ValueError("Provided function must define callable 'score(t, n)'")

        def wrapper(data_type_name: str, size: int) -> ScoreTriple:
            result = fn(data_type_name, int(size))
            if not isinstance(result, (list, tuple)) or len(result) != 3:
                raise ValueError("Function must return a tuple (result, speed, economy)")
            return _normalize((float(result[0]), float(result[1]), float(result[2])))

        return wrapper

    def set_code_for_control(self, control: ControlType, code: str) -> None:
        self._code_by_control[control] = code
        self._compiled_by_control[control] = self._compile_function(code)

    def get_code_for_control(self, control: ControlType) -> str:
        return self._code_by_control[control]

    def evaluate(self, control: ControlType, data_type: DataType, size: int) -> ScoreTriple:
        func = self._compiled_by_control[control]
        return func(data_type.name, int(size))


@dataclass
class FieldSpec:
    name: str
    data_type: DataType
    size: int

    def allowed_controls(self) -> List[ControlType]:
        return data_type_to_allowed_controls(self.data_type)


