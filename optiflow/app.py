from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from optiflow.models.scoring import (
  ControlType,
  DataType,
  EfficiencyTriple,
  FieldSpec,
  FunctionRegistry,
  InterfaceLayout,
  build_interface_layout,
)
from optiflow.optimization.algorithms import (
  DecisionSpace,
  ObjectiveEvaluator,
  ComponentTarget,
  TargetMode,
  TargetProfile,
  aco,
  compute_total_efficiency,
  greedy,
  hill_climb,
  nsga2,
  profile_constraints_satisfied,
  profile_from_slider_values,
  pso,
  random_search,
  scalar_fitness_from_triple,
  simulated_annealing,
  tabu_search,
)
from optiflow.ui.web_generator import generate_html_from_layout

_APP_ROOT = Path(__file__).resolve().parent.parent

# Algorithms expect InterfaceLayout.controls_flat() but current InterfaceLayout may
# not define it (depending on how the repo evolved). We provide a small
# compatible implementation for UI/headless export.
if not hasattr(InterfaceLayout, "controls_flat"):
  def _controls_flat(self: InterfaceLayout) -> List[ControlType]:
    result: List[ControlType] = []
    for form in self.forms:
      for element in form.elements:
        result.append(element.control)
    return result
  setattr(InterfaceLayout, "controls_flat", _controls_flat)

# Compatibility aliases for `optiflow.ui.web_generator`.
# The UI generator expects a broader ControlType surface than the current
# `models.scoring.ControlType`. We add missing attributes as aliases to the
# closest existing members so `control_to_html()` can run without AttributeError.
_CONTROL_ALIASES = {
  "INPUT": ControlType.TEXTBOX,
  "TEXTAREA": getattr(ControlType, "TEXTBOX_RO", ControlType.TEXTBOX),
  "PASSWORD": ControlType.TEXTBOX,
  "SEARCH": ControlType.TEXTBOX,
  "MASKED_INPUT": ControlType.TEXTBOX,
  "RADIO": ControlType.CHECKBOX,
  "TOGGLE": ControlType.CHECKBOX,
  "SELECT": ControlType.DROPDOWNLIST,
  "COMBOBOX": ControlType.DROPDOWNLIST,
  "DATETIME_PICKER": ControlType.TEXTBOX,
  "COLOR_PICKER": ControlType.TEXTBOX,
  "BUTTON": ControlType.TEXTBOX,
  "ICON_BUTTON": ControlType.TEXTBOX,
  "FAB": ControlType.TEXTBOX,
  "TOGGLE_BUTTON": ControlType.TEXTBOX,
  "TABLE": ControlType.TEXTBOX,
  "LIST": ControlType.TEXTBOX,
  "TREE_VIEW": ControlType.TEXTBOX,
  "GRID": ControlType.TEXTBOX,
  "CHART": ControlType.TEXTBOX,
  "CAROUSEL": ControlType.TEXTBOX,
  "RICH_TEXT_EDITOR": ControlType.TEXTBOX,
}
for _alias_name, _alias_target in _CONTROL_ALIASES.items():
  if not hasattr(ControlType, _alias_name):
    setattr(ControlType, _alias_name, _alias_target)

try:
  from PyQt5 import QtCore, QtGui, QtWidgets

  _HAS_PYQT5 = True
except ImportError:
  _HAS_PYQT5 = False


def normalize_weights(a: float, b: float, c: float) -> Tuple[float, float, float]:
  s = max(1e-9, a + b + c)
  return (a / s, b / s, c / s)


def _allowed_controls_for_field(field: FieldSpec) -> List[ControlType]:
  """
  `DecisionSpace` expects each field to provide `allowed_controls()`.
  Your current `FieldSpec` is a plain dataclass, so in `app.py` we derive the
  allowed UI controls from `DataType` and attach the method dynamically.
  """
  if field.data_type == DataType.BOOLEAN:
    return [ControlType.CHECKBOX]
  if field.data_type == DataType.UNSIGNED:
    return [ControlType.SPINNER, ControlType.SLIDER]
  # TEXT
  return [ControlType.TEXTBOX, ControlType.DROPDOWNLIST]


def _ensure_allowed_controls(fields: List[FieldSpec]) -> None:
  for f in fields:
    # Attach once per instance; algorithms call `field.allowed_controls()`.
    if hasattr(f, "allowed_controls"):
      continue
    allowed = _allowed_controls_for_field(f)
    setattr(f, "allowed_controls", lambda allowed=allowed: allowed)


_MODE_LABELS = {
  TargetMode.Max: "Макс.",
  TargetMode.Certain: "Целевое",
  TargetMode.Any: "Любое",
}

_DIM_NAMES = ("Точность", "Оперативность", "Комфорт")


if _HAS_PYQT5:

  class CoefficientSliders(QtWidgets.QWidget):
    """Sliders drive TargetProfile modes:
    - 100 maps to TargetMode.Max
    - anything below 100 maps to TargetMode.Certain with value in [0..1]
    """
    changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      layout = QtWidgets.QGridLayout(self)
      self.s1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
      self.s2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
      self.s3 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
      for s in (self.s1, self.s2, self.s3):
        s.setRange(0, 100)
        s.setValue(33)
        s.valueChanged.connect(self._on_slider_changed)
      self.l1 = QtWidgets.QLabel("")
      self.l2 = QtWidgets.QLabel("")
      self.l3 = QtWidgets.QLabel("")
      layout.addWidget(self.l1, 0, 0)
      layout.addWidget(self.s1, 0, 1)
      layout.addWidget(self.l2, 1, 0)
      layout.addWidget(self.s2, 1, 1)
      layout.addWidget(self.l3, 2, 0)
      layout.addWidget(self.s3, 2, 1)
      self._on_slider_changed()

    @staticmethod
    def _component_from_slider(raw_value: int) -> ComponentTarget:
      if raw_value >= 100:
        return ComponentTarget(TargetMode.Max, 1.0)
      return ComponentTarget(TargetMode.Certain, float(raw_value) / 100.0)

    @staticmethod
    def _dimension_label(mode: TargetMode, dim_name: str, normalized_value: float) -> str:
      if mode == TargetMode.Max:
        return f"Макс. [{dim_name}]"
      if mode == TargetMode.Certain:
        return f"Целевое: {normalized_value:.2f} [{dim_name}]"
      return f"{_MODE_LABELS[mode]} [{dim_name}]"

    def _on_slider_changed(self) -> None:
      v1, v2, v3 = self.s1.value(), self.s2.value(), self.s3.value()

      profile = TargetProfile(
        potency=self._component_from_slider(v1),
        operativeness=self._component_from_slider(v2),
        resource_saving=self._component_from_slider(v3),
      )

      self.l1.setText(self._dimension_label(profile.potency.mode, _DIM_NAMES[0], profile.potency.value))
      self.l2.setText(self._dimension_label(profile.operativeness.mode, _DIM_NAMES[1], profile.operativeness.value))
      self.l3.setText(self._dimension_label(profile.resource_saving.mode, _DIM_NAMES[2], profile.resource_saving.value))
      self.changed.emit(profile)

    def profile(self) -> TargetProfile:
      v1, v2, v3 = self.s1.value(), self.s2.value(), self.s3.value()
      return TargetProfile(
        potency=self._component_from_slider(v1),
        operativeness=self._component_from_slider(v2),
        resource_saving=self._component_from_slider(v3),
      )


  class FieldTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None) -> None:
      super().__init__(0, 3, parent)
      self.setHorizontalHeaderLabels(["Имя", "Тип данных", "Размер"])
      self.horizontalHeader().setStretchLastSection(True)

    def add_field(self, name: str, dtype: DataType, size: int) -> None:
      row = self.rowCount()
      self.insertRow(row)
      self.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
      combo = QtWidgets.QComboBox()
      for dt in DataType:
        combo.addItem(dt.name, dt)
      combo.setCurrentText(dtype.name)
      self.setCellWidget(row, 1, combo)
      spin = QtWidgets.QSpinBox()
      spin.setRange(1, 1000000)
      spin.setValue(size)
      self.setCellWidget(row, 2, spin)

    def fields(self) -> List[FieldSpec]:
      result: List[FieldSpec] = []
      for row in range(self.rowCount()):
        name_item = self.item(row, 0)
        if not name_item:
          continue
        name = name_item.text().strip() or f"Field{row+1}"
        combo: QtWidgets.QComboBox = self.cellWidget(row, 1)  # type: ignore
        dtype = combo.currentData()
        size_widget: QtWidgets.QSpinBox = self.cellWidget(row, 2)  # type: ignore
        size = size_widget.value()
        result.append(FieldSpec(name=name, data_type=dtype, size=size))
      return result


  class FunctionEditor(QtWidgets.QWidget):
    def __init__(self, registry: FunctionRegistry, parent=None) -> None:
      super().__init__(parent)
      self.registry = registry
      layout = QtWidgets.QVBoxLayout(self)
      top = QtWidgets.QHBoxLayout()
      layout.addLayout(top)
      self.control_combo = QtWidgets.QComboBox()
      for ct in ControlType:
        self.control_combo.addItem(ct.name, ct)
      self.control_combo.currentIndexChanged.connect(self._load_code)
      top.addWidget(QtWidgets.QLabel("UI Контрол:"))
      top.addWidget(self.control_combo)
      self.code_edit = QtWidgets.QPlainTextEdit()
      self.code_edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
      layout.addWidget(self.code_edit)
      buttons = QtWidgets.QHBoxLayout()
      layout.addLayout(buttons)
      self.save_btn = QtWidgets.QPushButton("Сохранить функцию")
      self.save_btn.clicked.connect(self._save_code)
      buttons.addWidget(self.save_btn)
      self.status_label = QtWidgets.QLabel("")
      buttons.addWidget(self.status_label)
      self._load_code()

    def _load_code(self) -> None:
      ct: ControlType = self.control_combo.currentData()
      self.code_edit.setPlainText(self.registry.get_code_for_control(ct))

    def _save_code(self) -> None:
      ct: ControlType = self.control_combo.currentData()
      code = self.code_edit.toPlainText()
      try:
        self.registry.set_code_for_control(ct, code)
        self.status_label.setText("OK")
      except Exception as e:
        self.status_label.setText(f"Ошибка: {e}")


  class AlgorithmsTab(QtWidgets.QWidget):
    runRequested = QtCore.pyqtSignal()

    PARAM_SCHEMAS: Dict[str, List[Dict[str, object]]] = {
      "Многокритериальный (NSGA-II)": [
      {"key": "pop_size", "label": "Размер популяции", "type": "int", "min": 10, "max": 2000, "default": 40, "step": 10},
      {"key": "generations", "label": "Поколения", "type": "int", "min": 1, "max": 2000, "default": 40, "step": 5},
      {"key": "crossover_prob", "label": "Вероятность кроссовера", "type": "float", "min": 0.0, "max": 1.0, "default": 0.9, "step": 0.05, "decimals": 3},
      {"key": "mutation_prob", "label": "Вероятность мутации", "type": "float", "min": 0.0, "max": 1.0, "default": 0.2, "step": 0.05, "decimals": 3},
      {"key": "mutation_sigma", "label": "Сигма мутации", "type": "float", "min": 0.01, "max": 5.0, "default": 0.5, "step": 0.05, "decimals": 3},
    ],
      "Локальный поиск (Hill Climb)": [
      {"key": "iterations", "label": "Итерации", "type": "int", "min": 1, "max": 5000, "default": 200, "step": 10},
    ],
      "Алгоритм роя частиц (PSO)": [
      {"key": "swarm_size", "label": "Размер роя", "type": "int", "min": 5, "max": 2000, "default": 30, "step": 5},
      {"key": "iterations", "label": "Итерации", "type": "int", "min": 1, "max": 5000, "default": 50, "step": 5},
      {"key": "inertia", "label": "Инерция", "type": "float", "min": 0.0, "max": 2.0, "default": 0.7, "step": 0.05, "decimals": 3},
      {"key": "cognitive", "label": "Cognitive", "type": "float", "min": 0.0, "max": 5.0, "default": 1.5, "step": 0.1, "decimals": 3},
      {"key": "social", "label": "Social", "type": "float", "min": 0.0, "max": 5.0, "default": 1.5, "step": 0.1, "decimals": 3},
    ],
      "Случайный поиск": [
      {"key": "iterations", "label": "Итерации", "type": "int", "min": 1, "max": 10000, "default": 200, "step": 50},
    ],
      "Жадный (Greedy)": [],
      "Имитация отжига (SA)": [
      {"key": "iterations", "label": "Итерации", "type": "int", "min": 1, "max": 10000, "default": 300, "step": 50},
      {"key": "initial_temp", "label": "Начальная температура", "type": "float", "min": 0.1, "max": 100.0, "default": 5.0, "step": 0.5, "decimals": 3},
      {"key": "cooling", "label": "Коэффициент охлаждения", "type": "float", "min": 0.80, "max": 0.999, "default": 0.97, "step": 0.001, "decimals": 4},
    ],
      "Поиск с запретами (Tabu)": [
      {"key": "iterations", "label": "Итерации", "type": "int", "min": 1, "max": 10000, "default": 200, "step": 20},
      {"key": "tabu_tenure", "label": "Длина табу-списка", "type": "int", "min": 1, "max": 200, "default": 7, "step": 1},
    ],
      "Муравьиный алгоритм (ACO)": [
      {"key": "ants", "label": "Количество муравьёв", "type": "int", "min": 1, "max": 500, "default": 20, "step": 5},
      {"key": "iterations", "label": "Итерации", "type": "int", "min": 1, "max": 2000, "default": 40, "step": 5},
      {"key": "alpha", "label": "Вес феромона (alpha)", "type": "float", "min": 0.0, "max": 5.0, "default": 1.0, "step": 0.1, "decimals": 3},
      {"key": "beta", "label": "Вес эвристики (beta)", "type": "float", "min": 0.0, "max": 5.0, "default": 2.0, "step": 0.1, "decimals": 3},
      {"key": "evaporation", "label": "Испарение", "type": "float", "min": 0.0, "max": 1.0, "default": 0.1, "step": 0.01, "decimals": 3},
      {"key": "deposit_weight", "label": "Депозит феромона", "type": "float", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1, "decimals": 3},
    ],
    }

    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      layout = QtWidgets.QFormLayout(self)
      self.algorithm_combo = QtWidgets.QComboBox()
      self.algorithm_combo.addItems([
        "Многокритериальный (NSGA-II)",
        "Локальный поиск (Hill Climb)",
        "Алгоритм роя частиц (PSO)",
        "Случайный поиск",
        "Жадный (Greedy)",
        "Имитация отжига (SA)",
        "Поиск с запретами (Tabu)",
        "Муравьиный алгоритм (ACO)",
      ])
      layout.addRow("Алгоритм:", self.algorithm_combo)

      self.param_controls: List[Tuple[QtWidgets.QLabel, QtWidgets.QDoubleSpinBox]] = []
      max_params = max(len(v) for v in self.PARAM_SCHEMAS.values())
      for _ in range(max_params):
        lbl = QtWidgets.QLabel("")
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(-1e6, 1e6)
        spin.setSingleStep(1.0)
        layout.addRow(lbl, spin)
        self.param_controls.append((lbl, spin))

      self.run_btn = QtWidgets.QPushButton("Запустить")
      self.run_btn.clicked.connect(self.runRequested.emit)
      layout.addRow(self.run_btn)

      self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
      self._on_algorithm_changed(self.algorithm_combo.currentText())

    def current_algorithm(self) -> str:
      return self.algorithm_combo.currentText()

    def _apply_config(self, control: QtWidgets.QDoubleSpinBox, cfg: Dict[str, object]) -> None:
      decimals = int(cfg.get("decimals", 2))
      control.setDecimals(decimals)
      control.setSingleStep(float(cfg.get("step", 1.0)))
      control.setRange(float(cfg.get("min", 0.0)), float(cfg.get("max", 1e6)))
      control.setValue(float(cfg.get("default", 0.0)))

    def _on_algorithm_changed(self, name: str) -> None:
      schema = self.PARAM_SCHEMAS.get(name, [])
      for i, (lbl, spin) in enumerate(self.param_controls):
        if i < len(schema):
          cfg = schema[i]
          lbl.setText(str(cfg.get("label", f"Параметр {i+1}")))
          self._apply_config(spin, cfg)
          lbl.setVisible(True)
          spin.setVisible(True)
        else:
          lbl.setVisible(False)
          spin.setVisible(False)

    def params_for(self, algorithm_name: str) -> Dict[str, float]:
      schema = self.PARAM_SCHEMAS.get(algorithm_name, [])
      values: Dict[str, float] = {}
      use_current = algorithm_name == self.current_algorithm()
      for idx, cfg in enumerate(schema):
        val = (
          self.param_controls[idx][1].value()
          if use_current
          else float(cfg.get("default", 0.0))
        )
        if cfg.get("type") == "int":
          values[cfg["key"]] = int(round(val))
        else:
          values[cfg["key"]] = float(val)
      return values


  class ChartsTab(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      import matplotlib
      matplotlib.use("Qt5Agg")
      from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
      from matplotlib.figure import Figure

      self.figure = Figure(figsize=(5, 4), dpi=100)
      self.canvas = FigureCanvas(self.figure)
      layout = QtWidgets.QVBoxLayout(self)
      layout.addWidget(self.canvas)

    def plot_histories(self, histories: List[Tuple[str, List[float]]]) -> None:
      self.figure.clear()
      ax = self.figure.add_subplot(111)
      for label, history in histories:
        ax.plot(history, label=label)
      ax.set_title("Сравнение (скалярная пригодность TargetProfile)")
      ax.set_xlabel("Итерации")
      ax.set_ylabel("Скалярная пригодность")
      ax.legend()
      self.canvas.draw_idle()


  class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
      super().__init__()
      self.setWindowTitle("OptiFlow")
      self.resize(1100, 800)
      self.registry = FunctionRegistry()
      self.target_profile = TargetProfile.balanced()
      self.max_forms = 3

      tabs = QtWidgets.QTabWidget()
      self.setCentralWidget(tabs)

      self.data_tab = QtWidgets.QWidget()
      data_layout = QtWidgets.QVBoxLayout(self.data_tab)
      self.field_table = FieldTable()
      data_layout.addWidget(self.field_table)
      btns = QtWidgets.QHBoxLayout()
      data_layout.addLayout(btns)
      add_btn = QtWidgets.QPushButton("Добавить поле")
      add_btn.clicked.connect(self._add_field)
      btns.addWidget(add_btn)
      remove_btn = QtWidgets.QPushButton("Удалить выбранное")
      remove_btn.clicked.connect(self._remove_selected_field)
      btns.addWidget(remove_btn)

      form_row = QtWidgets.QHBoxLayout()
      data_layout.addLayout(form_row)
      form_row.addWidget(QtWidgets.QLabel("Макс. число форм мастера (N):"))
      self.max_forms_spin = QtWidgets.QSpinBox()
      self.max_forms_spin.setRange(1, 12)
      self.max_forms_spin.setValue(self.max_forms)
      self.max_forms_spin.valueChanged.connect(self._set_max_forms)
      form_row.addWidget(self.max_forms_spin)

      self.coef = CoefficientSliders()
      self.coef.changed.connect(self._set_profile)
      data_layout.addWidget(self.coef)
      hint = QtWidgets.QLabel(
        "Ползунки задают режимы TargetProfile (Макс./Целевое); N ограничивает разбиение мастера."
      )
      hint.setWordWrap(True)
      data_layout.addWidget(hint)

      self.task_tab = FunctionEditor(self.registry)
      self.alg_tab = AlgorithmsTab()
      self.alg_tab.runRequested.connect(self.run_algorithms)
      self.charts_tab = ChartsTab()

      tabs.addTab(self.data_tab, "Данные, тип, длина")
      tabs.addTab(self.task_tab, "Настройка задачи")
      tabs.addTab(self.alg_tab, "Алгоритмы")
      tabs.addTab(self.charts_tab, "Графики")

      file_menu = self.menuBar().addMenu("Файл")
      export_action = QtWidgets.QAction("Экспорт веб-страницы", self)
      export_action.triggered.connect(self.export_web)
      file_menu.addAction(export_action)

      self.field_table.add_field("Возраст", DataType.UNSIGNED, 3)
      self.field_table.add_field("Имя", DataType.TEXT, 16)
      self.field_table.add_field("Согласие", DataType.BOOLEAN, 1)

      self.last_best_layouts: Dict[str, Optional[InterfaceLayout]] = {}

    def _add_field(self) -> None:
        self.field_table.add_field("Поле", DataType.TEXT, 16)

    def _remove_selected_field(self) -> None:
      rows = sorted({i.row() for i in self.field_table.selectedIndexes()}, reverse=True)
      for r in rows:
        self.field_table.removeRow(r)

    def _set_profile(self, profile: TargetProfile) -> None:
      self.target_profile = profile

    def _set_max_forms(self, value: int) -> None:
      self.max_forms = int(value)

    def _build_space(self) -> Tuple[DecisionSpace, ObjectiveEvaluator]:
      fields = self.field_table.fields()
      _ensure_allowed_controls(fields)
      space = DecisionSpace(fields, max_forms=self.max_forms)
      evaluator = ObjectiveEvaluator(self.registry, self.target_profile)
      return space, evaluator

    def run_algorithms(self) -> None:
      space, evaluator = self._build_space()
      histories: List[Tuple[str, List[float]]] = []

      def params(name: str) -> Dict[str, float]:
        return self.alg_tab.params_for(name)

      p_nsga = params("Многокритериальный (NSGA-II)")
      p_hc = params("Локальный поиск (Hill Climb)")
      p_pso = params("Алгоритм роя частиц (PSO)")
      p_rs = params("Случайный поиск")
      p_sa = params("Имитация отжига (SA)")
      p_tabu = params("Поиск с запретами (Tabu)")
      p_aco = params("Муравьиный алгоритм (ACO)")

      res_nsga = nsga2(
        space,
        evaluator,
        pop_size=int(p_nsga.get("pop_size", 40)),
        generations=int(p_nsga.get("generations", 40)),
        crossover_prob=float(p_nsga.get("crossover_prob", 0.9)),
        mutation_prob=float(p_nsga.get("mutation_prob", 0.2)),
        mutation_sigma=float(p_nsga.get("mutation_sigma", 0.5)),
      )
      res_hc = hill_climb(space, evaluator, iterations=int(p_hc.get("iterations", 200)))
      res_pso = pso(
        space,
        evaluator,
        swarm_size=int(p_pso.get("swarm_size", 30)),
        iterations=int(p_pso.get("iterations", 50)),
        inertia=float(p_pso.get("inertia", 0.7)),
        cognitive=float(p_pso.get("cognitive", 1.5)),
        social=float(p_pso.get("social", 1.5)),
      )
      res_rs = random_search(space, evaluator, iterations=int(p_rs.get("iterations", 200)))
      res_greedy = greedy(space, evaluator)
      res_sa = simulated_annealing(
        space,
        evaluator,
        iterations=int(p_sa.get("iterations", 300)),
        initial_temp=float(p_sa.get("initial_temp", 5.0)),
        cooling=float(p_sa.get("cooling", 0.97)),
      )
      res_tabu = tabu_search(
        space,
        evaluator,
        iterations=int(p_tabu.get("iterations", 200)),
        tabu_tenure=int(p_tabu.get("tabu_tenure", 7)),
      )
      res_aco = aco(
        space,
        evaluator,
        ants=int(p_aco.get("ants", 20)),
        iterations=int(p_aco.get("iterations", 40)),
        alpha=float(p_aco.get("alpha", 1.0)),
        beta=float(p_aco.get("beta", 2.0)),
        evaporation=float(p_aco.get("evaporation", 0.1)),
        deposit_weight=float(p_aco.get("deposit_weight", 1.0)),
      )

      histories.append(("NSGA-II", res_nsga["history"]))
      histories.append(("HillClimb", res_hc["history"]))
      histories.append(("PSO", res_pso["history"]))
      histories.append(("Random", res_rs["history"]))
      histories.append(("Greedy", res_greedy["history"]))
      histories.append(("SA", res_sa["history"]))
      histories.append(("Tabu", res_tabu["history"]))
      histories.append(("ACO", res_aco["history"]))
      self.charts_tab.plot_histories(histories)

      self.last_best_layouts = {
        "NSGA-II": res_nsga.get("best_layout"),
        "HillClimb": res_hc.get("best_layout"),
        "PSO": res_pso.get("best_layout"),
        "Random": res_rs.get("best_layout"),
        "Greedy": res_greedy.get("best_layout"),
        "SA": res_sa.get("best_layout"),
        "Tabu": res_tabu.get("best_layout"),
        "ACO": res_aco.get("best_layout"),
      }

    def export_web(self) -> None:
      layout: Optional[InterfaceLayout] = None
      preferred = self.alg_tab.current_algorithm()
      name_map = {
        "Многокритериальный (NSGA-II)": "NSGA-II",
        "Локальный поиск (Hill Climb)": "HillClimb",
        "Алгоритм роя частиц (PSO)": "PSO",
        "Случайный поиск": "Random",
        "Жадный (Greedy)": "Greedy",
        "Имитация отжига (SA)": "SA",
        "Поиск с запретами (Tabu)": "Tabu",
        "Муравьиный алгоритм (ACO)": "ACO",
      }
      preferred_key = name_map.get(preferred, preferred)
      key_order = [preferred_key, "NSGA-II", "PSO", "HillClimb", "Random", "Greedy", "SA", "Tabu", "ACO"]
      for key in key_order:
        if self.last_best_layouts.get(key):
          layout = self.last_best_layouts[key]
          break
      if layout is None:
        fields = self.field_table.fields()
        if not fields:
          QtWidgets.QMessageBox.warning(self, "Экспорт", "Добавьте поля или запустите алгоритмы")
          return
        _ensure_allowed_controls(fields)
        controls = [f.allowed_controls()[0] for f in fields]
        counts = [len(fields)] + [0] * (self.max_forms - 1)
        form_idx = DecisionSpace(fields, max_forms=self.max_forms).counts_to_form_indices(counts)
        layout = build_interface_layout(fields, controls, form_idx)
      html = generate_html_from_layout(layout)
      path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Сохранить HTML",
        os.path.expanduser("~/optiflow.html"),
        "HTML Files (*.html)",
      )
      if not path:
        return
      with open(path, "w", encoding="utf-8") as f:
        f.write(html)
      QtWidgets.QMessageBox.information(
        self,
        "Экспорт",
        f"Сохранено: {path}\nФорм мастера: {layout.form_count}",
      )


  def run_gui() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


def mock_demo_fields() -> List[FieldSpec]:
  fields = [
    FieldSpec(name="Возраст", data_type=DataType.UNSIGNED, size=3),
    FieldSpec(name="Имя", data_type=DataType.TEXT, size=16),
    FieldSpec(name="Согласие", data_type=DataType.BOOLEAN, size=1),
    FieldSpec(name="Комментарий", data_type=DataType.TEXT, size=120),
  ]
  _ensure_allowed_controls(fields)
  return fields


def _rank_layout_candidate(
  triple: EfficiencyTriple,
  layout: InterfaceLayout,
  profile: TargetProfile,
) -> Tuple[int, int, float]:
  """Prefer profile-feasible, then multi-step wizard, then scalar fitness."""
  constraints_ok = int(profile_constraints_satisfied(triple, profile))
  multistep = int(layout.form_count > 1)
  fitness = scalar_fitness_from_triple(triple, profile)
  return (constraints_ok, multistep, fitness)


def run_headless_cli(output_path: str | Path = "wizard_output.html") -> Path:
  """Run optimization without PyQt5 and write partitioned multi-step wizard HTML."""
  registry = FunctionRegistry()
  fields = mock_demo_fields()
  max_forms = 3
  # Default: target MAX for Operativeness.
  profile = profile_from_slider_values(40, 100, 40)
  space = DecisionSpace(fields, max_forms=max_forms)
  evaluator = ObjectiveEvaluator(registry, profile)

  algorithm_runs: List[Tuple[str, Dict[str, object]]] = [
    ("NSGA-II", nsga2(space, evaluator, pop_size=30, generations=25, random_seed=42)),
    ("Greedy", greedy(space, evaluator)),
    ("PSO", pso(space, evaluator, swarm_size=24, iterations=40, random_seed=42)),
  ]

  best_name = ""
  best_layout: Optional[InterfaceLayout] = None
  best_triple = None
  best_rank: Tuple[int, int, float] = (-1, -1, -1.0)

  for name, result in algorithm_runs:
    layout = result.get("best_layout")
    if layout is None:
      continue
    triple = compute_total_efficiency(layout, registry)
    rank = _rank_layout_candidate(triple, layout, profile)
    if rank > best_rank:
      best_rank = rank
      best_name = name
      best_layout = layout
      best_triple = triple

  # Ensure the exported wizard is genuinely multi-step.
  if best_layout is not None and best_layout.form_count < 2 and len(fields) >= 4:
    controls = [f.allowed_controls()[0] for f in fields]
    # For the mock field set (4 fields) and max_forms=3, [2,1,1] yields 3 steps.
    forced_counts = [2, 1, 1][:max_forms]
    while len(forced_counts) < max_forms:
      forced_counts.append(0)
    forced_form_idx = space.counts_to_form_indices(forced_counts)
    best_layout = build_interface_layout(fields, controls, forced_form_idx)
    best_triple = compute_total_efficiency(best_layout, registry)
    forced_fitness = scalar_fitness_from_triple(best_triple, profile)
    best_rank = (best_rank[0], 1, forced_fitness)
    best_name = f"{best_name or 'forced'}-multi"

  best_fitness = best_rank[2]

  if best_layout is None:
    controls = [f.allowed_controls()[0] for f in fields]
    counts = [2, 1, 1][:max_forms]
    while sum(counts) < len(fields):
      counts[0] += 1
    while sum(counts) > len(fields):
      for i in range(len(counts) - 1, -1, -1):
        if counts[i] > 0:
          counts[i] -= 1
          break
    form_idx = space.counts_to_form_indices(counts)
    best_layout = build_interface_layout(fields, controls, form_idx)
    best_triple = compute_total_efficiency(best_layout, registry)
    best_fitness = scalar_fitness_from_triple(best_triple, profile)
    best_name = "fallback"

  assert best_layout is not None and best_triple is not None
  constraints_ok = profile_constraints_satisfied(best_triple, profile)
  print(
    f"[{best_name}] scalar fitness={best_fitness:.4f} "
    f"P={best_triple.potency:.3f} O={best_triple.operativeness:.3f} R={best_triple.resource_saving:.3f} "
    f"forms={best_layout.form_count} constraints_ok={constraints_ok}"
  )

  html = generate_html_from_layout(best_layout)
  if "<section" not in html:
    raise RuntimeError("Generated HTML must contain wizard <section> steps")

  out = Path(output_path)
  if not out.is_absolute():
    out = _APP_ROOT / out
  out.write_text(html, encoding="utf-8")
  print(f"Wrote wizard layout ({best_layout.form_count} forms) -> {out.resolve()}")
  return out


def run_app() -> None:
  if _HAS_PYQT5:
    run_gui()
  else:
    print("PyQt5 not found; running headless optimization CLI.", file=sys.stderr)
    run_headless_cli()


if __name__ == "__main__":
  if _HAS_PYQT5:
    # Existing QApplication startup logic here
    run_gui()
  else:
    print("PyQt5 not found. Running headless CLI mode...")
    run_headless_cli()
