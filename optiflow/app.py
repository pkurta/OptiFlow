from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from optiflow import __version__
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
  CUSTOM_WEIGHT_PRESET,
  CriterionWeights,
  DEFAULT_WEIGHT_PRESET,
  DecisionSpace,
  ObjectiveEvaluator,
  OptimizationControl,
  ProgressReport,
  WEIGHT_PRESETS,
  aco,
  brute_force,
  calculate_fitness,
  classic_genetic_algorithm,
  compute_total_efficiency,
  greedy,
  hill_climb,
  nsga2,
  pso,
  random_search,
  redistribute_weight_ticks,
  resolve_weight_preset,
  simulated_annealing,
  tabu_search,
)
from optiflow.optimization.runner import (
  SUITE_STEPS,
  histories_from_results,
  run_optimization_suite,
)
from optiflow.benchmarks import run_optimization_benchmark
from optiflow.ui.gemini_interpretation import (
  DEFAULT_GEMINI_MODEL,
  GEMINI_MODEL_CHOICES,
  build_interpretation_prompt,
  gemini_model_label,
  request_gemini_flash,
)
from optiflow.ui.run_results import (
  AlgorithmRunSummary,
  build_algorithm_summaries,
  format_optimization_report,
  summaries_with_layouts,
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

try:
  from PyQt5.QtWebEngineWidgets import QWebEngineView

  _HAS_WEBENGINE = True
except ImportError:
  _HAS_WEBENGINE = False


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


_DIM_NAMES = ("Результативность", "Оперативность", "Ресурсоэкономность")
_DIM_WEIGHT_LABELS = (
  "Вес результативности",
  "Вес оперативности",
  "Вес ресурсоэкономности",
)
# Integer ticks cannot represent 1/3 + 1/3 + 1/3 = 1. For the Balance preset
# all three handles sit on the same tick so equality is visible; the model
# still uses exact 1/3.
_EQUAL_WEIGHT_DISPLAY_TICK = 33
_BALANCE_SLIDER_TIP = (
  "В сценарии «Баланс» веса равны точно 1/3. "
  "Перемещение ползунка включает «Свой вариант»."
)

# Итерационный параметр и дефолт для вкладки «Настройка задачи».
# None — алгоритм без настраиваемого лимита итераций.
ALGORITHM_LIMIT_DEFAULTS: Dict[str, Tuple[Optional[str], int]] = {
  "Многокритериальный (NSGA-II)": ("generations", 40),
  "Классический генетический алгоритм (GA)": ("generations", 30),
  "Полный перебор (Brute Force)": (None, 0),
  "Локальный поиск (Hill Climb)": ("iterations", 200),
  "Алгоритм роя частиц (PSO)": ("iterations", 50),
  "Случайный поиск": ("iterations", 200),
  "Жадный (Greedy)": (None, 0),
  "Имитация отжига (SA)": ("iterations", 300),
  "Поиск с запретами (Tabu)": ("iterations", 200),
  "Муравьиный алгоритм (ACO)": ("iterations", 40),
}

TASK_SETTINGS_FORMAT = "optiflow-task-settings"
TASK_SETTINGS_VERSION = 2

PROBLEM_DATA_FORMAT = "optiflow-problem-data"
PROBLEM_DATA_VERSION = 1


if _HAS_PYQT5:

  class _CommitSlider(QtWidgets.QSlider):
    """macOS native QSlider fires sliderReleased while dragging; bind commit to real mouse up."""

    gestureStarted = QtCore.pyqtSignal()
    gestureFinished = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
      super().__init__(QtCore.Qt.Horizontal, parent)
      fusion = QtWidgets.QStyleFactory.create("Fusion")
      if fusion is not None:
        self.setStyle(fusion)
      self.setRange(0, 100)
      self.setTracking(True)
      self.setFocusPolicy(QtCore.Qt.StrongFocus)
      self._mouse_down = False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
      if event.button() == QtCore.Qt.LeftButton:
        self._mouse_down = True
        self.gestureStarted.emit()
      super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
      super().mouseReleaseEvent(event)
      if event.button() == QtCore.Qt.LeftButton and self._mouse_down:
        self._mouse_down = False
        self.gestureFinished.emit()

    def is_mouse_down(self) -> bool:
      return self._mouse_down


  class WeightSliders(QtWidgets.QWidget):
    """During a drag only the active slider moves; others snap once on mouse-up or focus-out."""

    changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      self._updating = False
      self._pending_index: Optional[int] = None
      self._frozen_ticks: Optional[List[int]] = None
      layout = QtWidgets.QGridLayout(self)
      layout.setContentsMargins(0, 0, 0, 0)
      layout.setColumnStretch(1, 1)

      self.preset_combo = QtWidgets.QComboBox()
      self.preset_combo.addItems([*WEIGHT_PRESETS.keys(), CUSTOM_WEIGHT_PRESET])
      longest = max([*WEIGHT_PRESETS.keys(), CUSTOM_WEIGHT_PRESET], key=len)
      self.preset_combo.setMinimumContentsLength(len(longest))
      self.preset_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
      layout.addWidget(QtWidgets.QLabel("Сценарий приоритетов:"), 0, 0)
      layout.addWidget(self.preset_combo, 0, 1)
      self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

      self.sliders: List[_CommitSlider] = []
      self.labels: List[QtWidgets.QLabel] = []
      label_width = self.fontMetrics().horizontalAdvance("Вес ресурсоэкономности: 1.00") + 12
      for row, _name in enumerate(_DIM_NAMES, start=1):
        lbl = QtWidgets.QLabel()
        lbl.setMinimumWidth(label_width)
        lbl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        slider = _CommitSlider(self)
        slider.setToolTip(_BALANCE_SLIDER_TIP)
        slider.valueChanged.connect(self._on_slider_changed)
        slider.gestureStarted.connect(self._on_gesture_started)
        slider.gestureFinished.connect(self._on_gesture_finished)
        slider.installEventFilter(self)
        layout.addWidget(lbl, row, 0)
        layout.addWidget(slider, row, 1)
        self.labels.append(lbl)
        self.sliders.append(slider)

      self.sum_label = QtWidgets.QLabel("Сумма весов: 1.00")
      layout.addWidget(self.sum_label, 4, 0, 1, 2)

      self.preset_combo.setCurrentText(DEFAULT_WEIGHT_PRESET)
      self.set_weights(CriterionWeights.balanced(), mark_custom=False)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
      if obj in self.sliders and event.type() == QtCore.QEvent.FocusOut:
        QtCore.QTimer.singleShot(0, self._commit_if_focus_left)
      return super().eventFilter(obj, event)

    def _is_dragging(self) -> bool:
      if QtWidgets.QApplication.mouseButtons() & QtCore.Qt.LeftButton:
        return True
      return any(s.is_mouse_down() for s in self.sliders)

    def _is_balance_preset(self) -> bool:
      return self.preset_combo.currentText() == DEFAULT_WEIGHT_PRESET

    def _equal_illustration_ticks(self, ticks: Sequence[int]) -> bool:
      return (
        len(ticks) == 3
        and ticks[0] == ticks[1] == ticks[2] == _EQUAL_WEIGHT_DISPLAY_TICK
      )

    def _apply_equal_illustration(self) -> None:
      self._updating = True
      for slider in self.sliders:
        slider.setValue(_EQUAL_WEIGHT_DISPLAY_TICK)
      self._updating = False
      self._refresh_slider_tooltips()

    def _refresh_slider_tooltips(self) -> None:
      tip = _BALANCE_SLIDER_TIP if self._is_balance_preset() else ""
      for slider in self.sliders:
        slider.setToolTip(tip)

    def weights(self) -> CriterionWeights:
      if self._is_balance_preset():
        return CriterionWeights.balanced()
      return CriterionWeights.from_ticks(*(s.value() for s in self.sliders))

    def set_weights(self, weights: CriterionWeights, *, mark_custom: bool = True) -> None:
      self._pending_index = None
      self._frozen_ticks = None
      if not mark_custom and self._is_balance_preset():
        ticks = (_EQUAL_WEIGHT_DISPLAY_TICK,) * 3
      else:
        ticks = weights.to_ticks()
      self._updating = True
      for slider, tick in zip(self.sliders, ticks):
        slider.setValue(tick)
      self._updating = False
      if mark_custom:
        self._mark_custom_preset()
      self._refresh_slider_tooltips()
      self._refresh_all_labels()

    def commit(self) -> None:
      if self._updating or self._is_dragging():
        return
      ticks = [s.value() for s in self.sliders]
      idx = self._pending_index
      if self._frozen_ticks is not None and idx is not None:
        ticks = list(self._frozen_ticks)
        ticks[idx] = self.sliders[idx].value()
      # 33+33+33 cannot sit on the 100-tick simplex; keep the equal illustration
      # instead of promoting one handle to 0.34.
      if self._equal_illustration_ticks(ticks):
        self._pending_index = None
        self._frozen_ticks = None
        self._apply_equal_illustration()
        self._refresh_all_labels()
        return
      if idx is None:
        if sum(ticks) == 100:
          self._frozen_ticks = None
          return
        redistributed = CriterionWeights.from_ticks(*ticks).to_ticks()
      else:
        redistributed = redistribute_weight_ticks(ticks, idx, ticks[idx])
      self._pending_index = None
      self._frozen_ticks = None
      if tuple(s.value() for s in self.sliders) != tuple(redistributed):
        self._updating = True
        for slider, tick in zip(self.sliders, redistributed):
          if slider.value() != tick:
            slider.setValue(tick)
        self._updating = False
        self._mark_custom_preset()
      self._refresh_slider_tooltips()
      self._refresh_all_labels()
      self.changed.emit(self.weights())

    def _mark_custom_preset(self) -> None:
      if self.preset_combo.currentText() == CUSTOM_WEIGHT_PRESET:
        return
      self.preset_combo.blockSignals(True)
      self.preset_combo.setCurrentText(CUSTOM_WEIGHT_PRESET)
      self.preset_combo.blockSignals(False)

    def _commit_if_focus_left(self) -> None:
      if self._is_dragging():
        return
      focus = QtWidgets.QApplication.focusWidget()
      if focus in self.sliders or focus is self.preset_combo:
        return
      self.commit()

    def _on_preset_changed(self, name: str) -> None:
      if name == CUSTOM_WEIGHT_PRESET:
        self._refresh_slider_tooltips()
        self._refresh_all_labels()
        return
      if name not in WEIGHT_PRESETS:
        return
      self.set_weights(CriterionWeights.from_raw(*WEIGHT_PRESETS[name]), mark_custom=False)
      self.changed.emit(self.weights())

    def _on_gesture_started(self) -> None:
      sender = self.sender()
      if sender not in self.sliders:
        return
      idx = self.sliders.index(sender)
      if self._pending_index is not None and self._pending_index != idx:
        self.commit()
      self._pending_index = idx
      self._frozen_ticks = [s.value() for s in self.sliders]

    def _on_gesture_finished(self) -> None:
      self.commit()

    def _on_slider_changed(self) -> None:
      if self._updating:
        return
      sender = self.sender()
      if sender not in self.sliders:
        return
      idx = self.sliders.index(sender)
      self._pending_index = idx
      if self._frozen_ticks is not None:
        self._updating = True
        for other, frozen in enumerate(self._frozen_ticks):
          if other != idx and self.sliders[other].value() != frozen:
            self.sliders[other].setValue(frozen)
        self._updating = False
      if self._is_balance_preset() and sender.value() != _EQUAL_WEIGHT_DISPLAY_TICK:
        self._mark_custom_preset()
        self._refresh_slider_tooltips()
        self._refresh_all_labels()
        return
      self._refresh_label(idx)

    def _refresh_label(self, index: int) -> None:
      if self._is_balance_preset():
        self.labels[index].setText(f"{_DIM_WEIGHT_LABELS[index]}: 1/3")
        return
      tick = self.sliders[index].value()
      self.labels[index].setText(f"{_DIM_WEIGHT_LABELS[index]}: {tick / 100.0:.2f}")

    def _refresh_all_labels(self) -> None:
      for index in range(len(self.sliders)):
        self._refresh_label(index)
      self.sum_label.setText("Сумма весов: 1.00")

    def current_preset(self) -> str:
      return self.preset_combo.currentText()

    def apply_preset_or_weights(self, preset: str, weights: CriterionWeights) -> None:
      preset = resolve_weight_preset(preset)
      if preset in WEIGHT_PRESETS:
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(preset)
        self.preset_combo.blockSignals(False)
        self.set_weights(CriterionWeights.from_raw(*WEIGHT_PRESETS[preset]), mark_custom=False)
      else:
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(CUSTOM_WEIGHT_PRESET)
        self.preset_combo.blockSignals(False)
        self.set_weights(weights, mark_custom=False)
      self.changed.emit(self.weights())


  class FieldTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None) -> None:
      super().__init__(0, 3, parent)
      self.setHorizontalHeaderLabels(["Имя", "Тип данных", "Размер"])
      self.horizontalHeader().setStretchLastSection(True)

    def _sync_size_widget(self, row: int) -> None:
      combo: QtWidgets.QComboBox = self.cellWidget(row, 1)  # type: ignore
      spin: QtWidgets.QSpinBox = self.cellWidget(row, 2)  # type: ignore
      if combo is None or spin is None:
        return
      if combo.currentData() == DataType.BOOLEAN:
        spin.setValue(1)
        spin.setReadOnly(True)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
      else:
        spin.setReadOnly(False)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)

    def _on_dtype_changed(self) -> None:
      combo = self.sender()
      if not isinstance(combo, QtWidgets.QComboBox):
        return
      for row in range(self.rowCount()):
        if self.cellWidget(row, 1) is combo:
          self._sync_size_widget(row)
          return

    def add_field(self, name: str, dtype: DataType, size: int) -> None:
      row = self.rowCount()
      self.insertRow(row)
      self.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
      combo = QtWidgets.QComboBox()
      for dt in DataType:
        combo.addItem(dt.name, dt)
      combo.setCurrentText(dtype.name)
      combo.currentIndexChanged.connect(self._on_dtype_changed)
      self.setCellWidget(row, 1, combo)
      spin = QtWidgets.QSpinBox()
      spin.setRange(1, 1000000)
      spin.setValue(1 if dtype == DataType.BOOLEAN else size)
      self.setCellWidget(row, 2, spin)
      self._sync_size_widget(row)

    def next_default_field_name(self) -> str:
      prefix = "Поле "
      used = set()
      for row in range(self.rowCount()):
        item = self.item(row, 0)
        if not item:
          continue
        name = item.text().strip()
        if name.startswith(prefix):
          suffix = name[len(prefix):]
          if suffix.isdigit():
            used.add(int(suffix))
      n = 1
      while n in used:
        n += 1
      return f"{prefix}{n}"

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
        size = 1 if dtype == DataType.BOOLEAN else size_widget.value()
        result.append(FieldSpec(name=name, data_type=dtype, size=size))
      return result

    def export_fields(self) -> List[Dict[str, Any]]:
      return [
        {
          "name": field.name,
          "data_type": field.data_type.name,
          "size": field.size,
        }
        for field in self.fields()
      ]

    def load_fields(self, items: List[Dict[str, Any]]) -> None:
      self.setRowCount(0)
      for item in items:
        if not isinstance(item, dict):
          continue
        name = str(item.get("name", "")).strip() or self.next_default_field_name()
        try:
          dtype = DataType[str(item.get("data_type", "TEXT"))]
        except KeyError:
          dtype = DataType.TEXT
        size = max(1, int(item.get("size", 1)))
        if dtype == DataType.BOOLEAN:
          size = 1
        self.add_field(name, dtype, size)


  class DataTaskTab(QtWidgets.QWidget):
    maxFormsChanged = QtCore.pyqtSignal(int)
    weightsChanged = QtCore.pyqtSignal(object)

    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      layout = QtWidgets.QVBoxLayout(self)

      io_row = QtWidgets.QHBoxLayout()
      self.load_btn = QtWidgets.QPushButton("Загрузить…")
      self.load_btn.clicked.connect(self._load_problem_file)
      io_row.addWidget(self.load_btn)
      self.save_btn = QtWidgets.QPushButton("Сохранить…")
      self.save_btn.clicked.connect(self._save_problem_file)
      io_row.addWidget(self.save_btn)
      io_row.addStretch(1)
      self.io_status_label = QtWidgets.QLabel("")
      io_row.addWidget(self.io_status_label)
      layout.addLayout(io_row)

      self.field_table = FieldTable()
      layout.addWidget(self.field_table)

      field_btns = QtWidgets.QHBoxLayout()
      add_btn = QtWidgets.QPushButton("Добавить поле")
      add_btn.clicked.connect(self._add_field)
      field_btns.addWidget(add_btn)
      remove_btn = QtWidgets.QPushButton("Удалить выбранное")
      remove_btn.clicked.connect(self._remove_selected_field)
      field_btns.addWidget(remove_btn)
      layout.addLayout(field_btns)

      form_row = QtWidgets.QHBoxLayout()
      form_row.addWidget(QtWidgets.QLabel("Макс. число экранов мастера (N):"))
      self.max_forms_spin = QtWidgets.QSpinBox()
      self.max_forms_spin.setRange(1, 12)
      self.max_forms_spin.setValue(3)
      self.max_forms_spin.valueChanged.connect(self.maxFormsChanged.emit)
      form_row.addWidget(self.max_forms_spin)
      layout.addLayout(form_row)

      form_hint = QtWidgets.QLabel(
        "N задаёт, на сколько экранов алгоритм может распределить контролы."
      )
      form_hint.setWordWrap(True)
      layout.addWidget(form_hint)

      weights_box = QtWidgets.QGroupBox("Приоритеты оптимизации")
      weights_layout = QtWidgets.QVBoxLayout(weights_box)
      self.coef = WeightSliders()
      self.coef.changed.connect(self.weightsChanged.emit)
      weights_layout.addWidget(self.coef)
      hint = QtWidgets.QLabel(
        "P, O и R — расчётные свойства готового интерфейса, не вход. "
        "Здесь задаются только веса свёртки F = w₁P + w₂O + w₃R, сумма всегда равна 1. "
        "В сценарии «Баланс» веса равны точно 1/3; два знака после запятой — в «Свой вариант»."
      )
      hint.setWordWrap(True)
      weights_layout.addWidget(hint)
      layout.addWidget(weights_box)

      self.field_table.add_field("Возраст", DataType.UNSIGNED, 3)
      self.field_table.add_field("Имя", DataType.TEXT, 16)
      self.field_table.add_field("Согласие", DataType.BOOLEAN, 1)

    @property
    def max_forms(self) -> int:
      return int(self.max_forms_spin.value())

    def _add_field(self) -> None:
      self.field_table.add_field(self.field_table.next_default_field_name(), DataType.TEXT, 16)

    def _remove_selected_field(self) -> None:
      rows = sorted({i.row() for i in self.field_table.selectedIndexes()}, reverse=True)
      for row in rows:
        self.field_table.removeRow(row)

    def export_problem(self) -> Dict[str, Any]:
      self.coef.commit()
      weights = self.coef.weights()
      w1, w2, w3 = weights.as_tuple()
      return {
        "format": PROBLEM_DATA_FORMAT,
        "version": PROBLEM_DATA_VERSION,
        "optiflow_version": __version__,
        "max_forms": self.max_forms,
        "weight_preset": self.coef.current_preset(),
        "weights": {
          "potency": w1,
          "operativeness": w2,
          "resource_saving": w3,
        },
        "fields": self.field_table.export_fields(),
      }

    def import_problem(self, data: Dict[str, Any]) -> None:
      if data.get("format") != PROBLEM_DATA_FORMAT:
        raise ValueError(
          f"Неизвестный формат файла: {data.get('format')!r}. "
          f"Ожидается {PROBLEM_DATA_FORMAT!r}."
        )
      version = int(data.get("version", 0))
      if version != PROBLEM_DATA_VERSION:
        raise ValueError(
          f"Неподдерживаемая версия данных: {version}. "
          f"Ожидается {PROBLEM_DATA_VERSION}."
        )
      fields = data.get("fields")
      if not isinstance(fields, list):
        raise ValueError("В файле отсутствует массив fields.")
      if not fields:
        raise ValueError("Список полей не может быть пустым.")
      weights_raw = data.get("weights")
      if not isinstance(weights_raw, dict):
        raise ValueError("В файле отсутствует объект weights.")
      weights = CriterionWeights.from_raw(
        float(weights_raw.get("potency", 0.0)),
        float(weights_raw.get("operativeness", 0.0)),
        float(weights_raw.get("resource_saving", 0.0)),
      )
      max_forms = max(1, min(12, int(data.get("max_forms", 1))))
      preset = str(data.get("weight_preset", CUSTOM_WEIGHT_PRESET))
      self.max_forms_spin.setValue(max_forms)
      self.coef.apply_preset_or_weights(preset, weights)
      self.field_table.load_fields(fields)
      self.maxFormsChanged.emit(max_forms)
      self.weightsChanged.emit(self.coef.weights())

    def _save_problem_file(self) -> None:
      path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Сохранить постановку задачи",
        str(_APP_ROOT / "problem_data.json"),
        "JSON (*.json)",
      )
      if not path:
        return
      if not path.lower().endswith(".json"):
        path += ".json"
      try:
        payload = self.export_problem()
        with open(path, "w", encoding="utf-8") as fh:
          json.dump(payload, fh, ensure_ascii=False, indent=2)
          fh.write("\n")
        self.io_status_label.setText(f"Сохранено: {Path(path).name}")
      except Exception as exc:
        QtWidgets.QMessageBox.critical(
          self,
          "Ошибка сохранения",
          f"Не удалось сохранить постановку задачи:\n{exc}",
        )

    def _load_problem_file(self) -> None:
      path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Загрузить постановку задачи",
        str(_APP_ROOT),
        "JSON (*.json)",
      )
      if not path:
        return
      try:
        with open(path, encoding="utf-8") as fh:
          data = json.load(fh)
        if not isinstance(data, dict):
          raise ValueError("Корень JSON-файла должен быть объектом.")
        self.import_problem(data)
        self.io_status_label.setText(f"Загружено: {Path(path).name}")
      except Exception as exc:
        QtWidgets.QMessageBox.critical(
          self,
          "Ошибка загрузки",
          f"Не удалось загрузить постановку задачи:\n{exc}",
        )


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

    def commit_current(self) -> None:
      """Persist the open editor buffer into the registry (best-effort)."""
      ct: ControlType = self.control_combo.currentData()
      code = self.code_edit.toPlainText()
      try:
        self.registry.set_code_for_control(ct, code)
      except Exception:
        pass

    def reload_current(self) -> None:
      self._load_code()


  class AlgorithmLimitsTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None) -> None:
      super().__init__(0, 3, parent)
      self.setHorizontalHeaderLabels(["Алгоритм", "Макс. итераций", "Время, сек"])
      self.horizontalHeader().setStretchLastSection(True)
      self.verticalHeader().setVisible(False)
      self.setAlternatingRowColors(True)
      self._iteration_spins: List[Optional[QtWidgets.QSpinBox]] = []
      self._time_spins: List[QtWidgets.QSpinBox] = []
      for name in ALGORITHM_LIMIT_DEFAULTS:
        self._add_row(name)

    def _add_row(self, algorithm_name: str) -> None:
      row = self.rowCount()
      self.insertRow(row)
      name_item = QtWidgets.QTableWidgetItem(algorithm_name)
      name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
      self.setItem(row, 0, name_item)

      iter_key, default_iter = ALGORITHM_LIMIT_DEFAULTS[algorithm_name]
      iter_spin: Optional[QtWidgets.QSpinBox]
      if iter_key is None:
        iter_spin = None
        placeholder = QtWidgets.QTableWidgetItem("—")
        placeholder.setFlags(placeholder.flags() & ~QtCore.Qt.ItemIsEditable)
        placeholder.setTextAlignment(QtCore.Qt.AlignCenter)
        self.setItem(row, 1, placeholder)
      else:
        iter_spin = QtWidgets.QSpinBox()
        iter_spin.setRange(1, 1_000_000)
        iter_spin.setValue(default_iter)
        self.setCellWidget(row, 1, iter_spin)
      self._iteration_spins.append(iter_spin)

      time_spin = QtWidgets.QSpinBox()
      time_spin.setRange(0, 86_400)
      time_spin.setSpecialValueText("без лимита")
      time_spin.setValue(0)
      self.setCellWidget(row, 2, time_spin)
      self._time_spins.append(time_spin)

    def params_for(self, algorithm_name: str) -> Dict[str, float]:
      for row, name in enumerate(ALGORITHM_LIMIT_DEFAULTS):
        if name != algorithm_name:
          continue
        result: Dict[str, float] = {"time_limit_s": float(self._time_spins[row].value())}
        iter_key, _ = ALGORITHM_LIMIT_DEFAULTS[name]
        iter_spin = self._iteration_spins[row]
        if iter_key is not None and iter_spin is not None:
          result[iter_key] = float(iter_spin.value())
        return result
      return {"time_limit_s": 0.0}

    def all_params(self) -> Dict[str, Dict[str, float]]:
      return {name: self.params_for(name) for name in ALGORITHM_LIMIT_DEFAULTS}

    def export_limits(self) -> Dict[str, Dict[str, int]]:
      result: Dict[str, Dict[str, int]] = {}
      for row, name in enumerate(ALGORITHM_LIMIT_DEFAULTS):
        entry: Dict[str, int] = {"time_limit_s": int(self._time_spins[row].value())}
        iter_spin = self._iteration_spins[row]
        if iter_spin is not None:
          entry["max_iterations"] = int(iter_spin.value())
        result[name] = entry
      return result

    def import_limits(self, data: Dict[str, Any]) -> None:
      for row, name in enumerate(ALGORITHM_LIMIT_DEFAULTS):
        entry = data.get(name)
        if not isinstance(entry, dict):
          continue
        if "time_limit_s" in entry:
          self._time_spins[row].setValue(max(0, int(entry["time_limit_s"])))
        iter_spin = self._iteration_spins[row]
        if iter_spin is not None and "max_iterations" in entry:
          iter_spin.setValue(max(1, int(entry["max_iterations"])))


  class TaskSettingsTab(QtWidgets.QWidget):
    def __init__(self, registry: FunctionRegistry, parent=None) -> None:
      super().__init__(parent)
      self.registry = registry
      layout = QtWidgets.QVBoxLayout(self)

      io_row = QtWidgets.QHBoxLayout()
      self.load_btn = QtWidgets.QPushButton("Загрузить…")
      self.load_btn.clicked.connect(self._load_settings_file)
      io_row.addWidget(self.load_btn)
      self.save_btn = QtWidgets.QPushButton("Сохранить…")
      self.save_btn.clicked.connect(self._save_settings_file)
      io_row.addWidget(self.save_btn)
      io_row.addStretch(1)
      self.io_status_label = QtWidgets.QLabel("")
      io_row.addWidget(self.io_status_label)
      layout.addLayout(io_row)

      functions_box = QtWidgets.QGroupBox("Функции оценки контролов")
      functions_layout = QtWidgets.QVBoxLayout(functions_box)
      self.function_editor = FunctionEditor(registry)
      functions_layout.addWidget(self.function_editor)
      layout.addWidget(functions_box, stretch=3)

      limits_box = QtWidgets.QGroupBox("Лимиты выполнения алгоритмов")
      limits_layout = QtWidgets.QVBoxLayout(limits_box)
      self.limits_table = AlgorithmLimitsTable()
      limits_layout.addWidget(self.limits_table)
      hint = QtWidgets.QLabel(
        "Макс. итераций задаёт верхнюю границу шагов алгоритма. "
        "Время — лимит в секундах на один алгоритм (0 = без ограничения)."
      )
      hint.setWordWrap(True)
      limits_layout.addWidget(hint)
      layout.addWidget(limits_box, stretch=2)

    def limits_for(self, algorithm_name: str) -> Dict[str, float]:
      return self.limits_table.params_for(algorithm_name)

    def export_settings(self) -> Dict[str, Any]:
      self.function_editor.commit_current()
      from optiflow.models.scoring import KURTA_2024_META_SOURCE

      return {
        "format": TASK_SETTINGS_FORMAT,
        "version": TASK_SETTINGS_VERSION,
        "optiflow_version": __version__,
        "meta_source": KURTA_2024_META_SOURCE,
        "control_functions": self.registry.export_functions(),
        "algorithm_limits": self.limits_table.export_limits(),
      }

    def import_settings(self, data: Dict[str, Any]) -> None:
      if data.get("format") != TASK_SETTINGS_FORMAT:
        raise ValueError(
          f"Неизвестный формат файла: {data.get('format')!r}. "
          f"Ожидается {TASK_SETTINGS_FORMAT!r}."
        )
      version = int(data.get("version", 0))
      if version != TASK_SETTINGS_VERSION:
        raise ValueError(
          f"Неподдерживаемая версия настроек: {version}. "
          f"Ожидается {TASK_SETTINGS_VERSION}."
        )
      functions = data.get("control_functions")
      if not isinstance(functions, dict):
        raise ValueError("В файле отсутствует объект control_functions.")
      limits = data.get("algorithm_limits")
      if not isinstance(limits, dict):
        raise ValueError("В файле отсутствует объект algorithm_limits.")
      self.registry.import_functions(functions)
      self.limits_table.import_limits(limits)
      self.function_editor.reload_current()

    def _save_settings_file(self) -> None:
      path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Сохранить настройки задачи",
        str(_APP_ROOT / "task_settings.json"),
        "JSON (*.json)",
      )
      if not path:
        return
      if not path.lower().endswith(".json"):
        path += ".json"
      try:
        payload = self.export_settings()
        with open(path, "w", encoding="utf-8") as fh:
          json.dump(payload, fh, ensure_ascii=False, indent=2)
          fh.write("\n")
        self.io_status_label.setText(f"Сохранено: {Path(path).name}")
      except Exception as exc:
        QtWidgets.QMessageBox.critical(
          self,
          "Ошибка сохранения",
          f"Не удалось сохранить настройки:\n{exc}",
        )

    def _load_settings_file(self) -> None:
      path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Загрузить настройки задачи",
        str(_APP_ROOT),
        "JSON (*.json)",
      )
      if not path:
        return
      try:
        with open(path, encoding="utf-8") as fh:
          data = json.load(fh)
        if not isinstance(data, dict):
          raise ValueError("Корень JSON-файла должен быть объектом.")
        self.import_settings(data)
        self.io_status_label.setText(f"Загружено: {Path(path).name}")
      except Exception as exc:
        QtWidgets.QMessageBox.critical(
          self,
          "Ошибка загрузки",
          f"Не удалось загрузить настройки:\n{exc}",
        )


  class AlgorithmsTab(QtWidgets.QWidget):
    runRequested = QtCore.pyqtSignal()

    PARAM_SCHEMAS: Dict[str, List[Dict[str, object]]] = {
      "Многокритериальный (NSGA-II)": [
      {"key": "pop_size", "label": "Размер популяции", "type": "int", "min": 10, "max": 2000, "default": 40, "step": 10},
      {"key": "crossover_prob", "label": "Вероятность кроссовера", "type": "float", "min": 0.0, "max": 1.0, "default": 0.9, "step": 0.05, "decimals": 3},
      {"key": "mutation_prob", "label": "Вероятность мутации", "type": "float", "min": 0.0, "max": 1.0, "default": 0.2, "step": 0.05, "decimals": 3},
      {"key": "mutation_sigma", "label": "Сигма мутации", "type": "float", "min": 0.01, "max": 5.0, "default": 0.5, "step": 0.05, "decimals": 3},
    ],
      "Локальный поиск (Hill Climb)": [],
      "Алгоритм роя частиц (PSO)": [
      {"key": "swarm_size", "label": "Размер роя", "type": "int", "min": 5, "max": 2000, "default": 30, "step": 5},
      {"key": "inertia", "label": "Инерция", "type": "float", "min": 0.0, "max": 2.0, "default": 0.7, "step": 0.05, "decimals": 3},
      {"key": "cognitive", "label": "Cognitive", "type": "float", "min": 0.0, "max": 5.0, "default": 1.5, "step": 0.1, "decimals": 3},
      {"key": "social", "label": "Social", "type": "float", "min": 0.0, "max": 5.0, "default": 1.5, "step": 0.1, "decimals": 3},
    ],
      "Случайный поиск": [],
      "Жадный (Greedy)": [],
      "Полный перебор (Brute Force)": [],
      "Классический генетический алгоритм (GA)": [
      {"key": "pop_size", "label": "Размер популяции", "type": "int", "min": 10, "max": 1000, "default": 30, "step": 10},
    ],
      "Имитация отжига (SA)": [
      {"key": "initial_temp", "label": "Начальная температура", "type": "float", "min": 0.1, "max": 100.0, "default": 5.0, "step": 0.5, "decimals": 3},
      {"key": "cooling", "label": "Коэффициент охлаждения", "type": "float", "min": 0.80, "max": 0.999, "default": 0.97, "step": 0.001, "decimals": 4},
    ],
      "Поиск с запретами (Tabu)": [
      {"key": "tabu_tenure", "label": "Длина табу-списка", "type": "int", "min": 1, "max": 200, "default": 7, "step": 1},
    ],
      "Муравьиный алгоритм (ACO)": [
      {"key": "ants", "label": "Количество муравьёв", "type": "int", "min": 1, "max": 500, "default": 20, "step": 5},
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
        "Классический генетический алгоритм (GA)",
        "Полный перебор (Brute Force)",
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
      max_params = max((len(v) for v in self.PARAM_SCHEMAS.values()), default=0)
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

    def set_run_enabled(self, enabled: bool) -> None:
      self.run_btn.setEnabled(enabled)

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


  class ProgressOverlay(QtWidgets.QWidget):
    cancelRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      self.setObjectName("progressOverlay")
      self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
      self.setStyleSheet(
        """
        QWidget#progressOverlay { background-color: rgba(7, 10, 18, 175); }
        QFrame#progressCard {
          background-color: #1b2330;
          border: 1px solid #4d5d74;
          border-radius: 18px;
        }
        QLabel#progressTitle { color: #f4f7fb; font-size: 16px; font-weight: 600; }
        QLabel#progressStep { color: #8fa0b8; font-size: 12px; }
        QLabel#progressMetric { color: #d5deea; font-size: 13px; }
        QLabel#progressTime { color: #9eb0c7; font-size: 12px; }
        QPushButton#progressCancel {
          background-color: #c44536;
          color: white;
          border: none;
          border-radius: 8px;
          padding: 8px 22px;
          font-weight: 600;
        }
        QPushButton#progressCancel:hover { background-color: #a8382c; }
        QPushButton#progressCancel:disabled { background-color: #667084; }
        QProgressBar {
          border: none;
          border-radius: 8px;
          background: #2a3344;
          min-height: 16px;
          text-align: center;
          color: #eef3fa;
        }
        QProgressBar::chunk {
          border-radius: 8px;
          background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #3b82f6, stop:1 #22d3ee);
        }
        """
      )
      root = QtWidgets.QVBoxLayout(self)
      root.setContentsMargins(24, 24, 24, 24)
      root.addStretch(1)
      card = QtWidgets.QFrame()
      card.setObjectName("progressCard")
      card.setMinimumWidth(520)
      card.setMaximumWidth(640)
      card_layout = QtWidgets.QVBoxLayout(card)
      card_layout.setContentsMargins(28, 24, 28, 24)
      card_layout.setSpacing(12)
      self.title_label = QtWidgets.QLabel("Подготовка…")
      self.title_label.setObjectName("progressTitle")
      self.title_label.setWordWrap(True)
      self.step_label = QtWidgets.QLabel("Алгоритм 0 из 10")
      self.step_label.setObjectName("progressStep")
      self.bar = QtWidgets.QProgressBar()
      self.bar.setRange(0, 1000)
      self.bar.setValue(0)
      self.bar.setTextVisible(True)
      self.bar.setFormat("%p%")
      self.metrics_label = QtWidgets.QLabel("F = —    P = —    O = —    R = —")
      self.metrics_label.setObjectName("progressMetric")
      self.time_label = QtWidgets.QLabel("Время: 00:00")
      self.time_label.setObjectName("progressTime")
      self.cancel_btn = QtWidgets.QPushButton("Прервать")
      self.cancel_btn.setObjectName("progressCancel")
      self.cancel_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
      self.cancel_btn.clicked.connect(self._on_cancel)
      card_layout.addWidget(self.title_label)
      card_layout.addWidget(self.step_label)
      card_layout.addWidget(self.bar)
      card_layout.addWidget(self.metrics_label)
      card_layout.addWidget(self.time_label)
      card_layout.addWidget(self.cancel_btn, 0, QtCore.Qt.AlignLeft)
      row = QtWidgets.QHBoxLayout()
      row.addStretch(1)
      row.addWidget(card)
      row.addStretch(1)
      root.addLayout(row)
      root.addStretch(1)
      self._elapsed = QtCore.QElapsedTimer()
      self._ticker = QtCore.QTimer(self)
      self._ticker.timeout.connect(self._refresh_elapsed)
      self.hide()

    def start(self) -> None:
      self.cancel_btn.setEnabled(True)
      self.cancel_btn.setText("Прервать")
      self.bar.setValue(0)
      self.title_label.setText("Запуск оптимизации…")
      self.step_label.setText("Алгоритм 0 из 10")
      self.metrics_label.setText("F = —    P = —    O = —    R = —")
      self.time_label.setText("Время: 00:00")
      self._elapsed.start()
      self._ticker.start(100)
      self.show()
      self.raise_()

    def stop(self) -> None:
      self._ticker.stop()
      self.hide()

    def apply_report(self, report: ProgressReport) -> None:
      self.title_label.setText(
        f"{report.algorithm} — шаг {report.iteration} / {report.max_iterations}"
      )
      self.step_label.setText(
        f"Алгоритм {report.algorithm_index + 1} из {report.algorithm_count}"
      )
      self.bar.setValue(int(round(report.overall_fraction * 1000)))
      self.metrics_label.setText(
        f"F = {report.best_fitness:.4f}    "
        f"P = {report.potency:.3f}    "
        f"O = {report.operativeness:.3f}    "
        f"R = {report.resource_saving:.3f}"
      )

    def _refresh_elapsed(self) -> None:
      ms = self._elapsed.elapsed() if self._elapsed.isValid() else 0
      seconds = ms // 1000
      self.time_label.setText(f"Время: {seconds // 60:02d}:{seconds % 60:02d}")

    def _on_cancel(self) -> None:
      self.cancel_btn.setEnabled(False)
      self.cancel_btn.setText("Остановка…")
      self.cancelRequested.emit()


  class OptimizationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(object)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
      self,
      space: DecisionSpace,
      evaluator: ObjectiveEvaluator,
      params_by_label: Dict[str, Dict[str, float]],
      parent=None,
    ) -> None:
      super().__init__(parent)
      self._space = space
      self._evaluator = evaluator
      self._params = params_by_label
      self.control: Optional[OptimizationControl] = None

    def run(self) -> None:
      try:
        payload = run_optimization_suite(
          self._space, self._evaluator, self._params, self.control
        )
        self.completed.emit(payload)
      except Exception as exc:
        self.failed.emit(f"{type(exc).__name__}: {exc}")


  # Стили серий: цвет + linestyle + marker — тройная кодировка против наложения плато.
  _CHART_SERIES_STYLES: Tuple[Tuple[str, str, str], ...] = (
    ("#1f77b4", "-", "o"),
    ("#ff7f0e", "--", "s"),
    ("#2ca02c", "-.", "^"),
    ("#d62728", ":", "D"),
    ("#9467bd", "-", "v"),
    ("#8c564b", "--", "P"),
    ("#e377c2", "-.", "X"),
    ("#7f7f7f", ":", "*"),
    ("#bcbd22", "-", "h"),
    ("#17becf", "--", "d"),
  )

  class ChartsTab(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      import matplotlib
      matplotlib.use("Qt5Agg")
      from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
      from matplotlib.figure import Figure

      self.figure = Figure(figsize=(5, 4), dpi=100)
      self.canvas = FigureCanvas(self.figure)
      self._histories: List[Tuple[str, List[float]]] = []
      self._line_by_label: Dict[str, object] = {}
      self._legend = None
      self._hover_cid: Optional[int] = None
      self._pick_cid: Optional[int] = None
      self._hovered_legend_idx: Optional[int] = None

      layout = QtWidgets.QVBoxLayout(self)
      controls = QtWidgets.QHBoxLayout()
      controls.addWidget(QtWidgets.QLabel("Отображение:"))
      self.filter_combo = QtWidgets.QComboBox()
      self.filter_combo.addItem("Топ-3 по итоговому F", "top3")
      self.filter_combo.addItem("Все алгоритмы", "all")
      self.filter_combo.setCurrentIndex(0)
      self.filter_combo.currentIndexChanged.connect(self._redraw)
      controls.addWidget(self.filter_combo)
      hint = QtWidgets.QLabel(
        "Клик по легенде — вкл/выкл серию · наведение — выделить кривую"
      )
      hint.setStyleSheet("color: #666;")
      controls.addWidget(hint, stretch=1)
      layout.addLayout(controls)
      layout.addWidget(self.canvas)
      self.results_label = QtWidgets.QLabel(
        "Запустите алгоритмы, чтобы увидеть расчётные P, O, R выбранных решений."
      )
      self.results_label.setWordWrap(True)
      self.results_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
      layout.addWidget(self.results_label)

    def plot_histories(self, histories: List[Tuple[str, List[float]]]) -> None:
      self._histories = list(histories)
      self._redraw()

    def _final_fitness(self, history: List[float]) -> float:
      return float(history[-1]) if history else float("-inf")

    def _visible_histories(self) -> List[Tuple[str, List[float]]]:
      mode = str(self.filter_combo.currentData() or "top3")
      if mode != "top3" or len(self._histories) <= 3:
        return list(self._histories)
      ranked = sorted(
        self._histories,
        key=lambda item: self._final_fitness(item[1]),
        reverse=True,
      )
      return ranked[:3]

    def _redraw(self) -> None:
      self.figure.clear()
      self._line_by_label = {}
      self._hovered_legend_idx = None
      if self._hover_cid is not None:
        self.canvas.mpl_disconnect(self._hover_cid)
        self._hover_cid = None
      if self._pick_cid is not None:
        self.canvas.mpl_disconnect(self._pick_cid)
        self._pick_cid = None

      ax = self.figure.add_subplot(111)
      visible = self._visible_histories()
      if not visible:
        ax.set_title("Скорость выхода на плато F (нет данных)")
        ax.set_xlabel("Итерации")
        ax.set_ylabel("Скалярная пригодность F")
        self.canvas.draw_idle()
        return

      for idx, (label, history) in enumerate(visible):
        color, linestyle, marker = _CHART_SERIES_STYLES[idx % len(_CHART_SERIES_STYLES)]
        markevery = max(1, len(history) // 12) if history else 1
        (line,) = ax.plot(
          history,
          label=label,
          color=color,
          linestyle=linestyle,
          linewidth=2.0,
          marker=marker,
          markersize=5.5,
          markevery=markevery,
          alpha=0.95,
          picker=True,
          pickradius=5,
        )
        self._line_by_label[label] = line

      ax.set_title("Скорость выхода на плато (F = w₁P + w₂O + w₃R)")
      ax.set_xlabel("Итерации")
      ax.set_ylabel("Скалярная пригодность F")
      self._legend = ax.legend(loc="best", framealpha=0.92)
      if self._legend is not None:
        for legend_line in self._legend.get_lines():
          legend_line.set_picker(True)
          legend_line.set_pickradius(8)
      self._pick_cid = self.canvas.mpl_connect("pick_event", self._on_legend_pick)
      self._hover_cid = self.canvas.mpl_connect(
        "motion_notify_event", self._on_legend_hover
      )
      self.figure.tight_layout()
      self.canvas.draw_idle()

    def _on_legend_pick(self, event) -> None:
      artist = getattr(event, "artist", None)
      if artist is None or self._legend is None:
        return
      legend_lines = list(self._legend.get_lines())
      if artist not in legend_lines:
        return
      idx = legend_lines.index(artist)
      labels = list(self._line_by_label.keys())
      if idx >= len(labels):
        return
      line = self._line_by_label[labels[idx]]
      visible = not bool(line.get_visible())
      line.set_visible(visible)
      artist.set_alpha(1.0 if visible else 0.25)
      self.canvas.draw_idle()

    def _on_legend_hover(self, event) -> None:
      if self._legend is None or not self._line_by_label:
        return
      legend_lines = list(self._legend.get_lines())
      hovered_idx: Optional[int] = None
      for idx, legend_line in enumerate(legend_lines):
        contains, _ = legend_line.contains(event)
        if contains:
          hovered_idx = idx
          break
      if hovered_idx == self._hovered_legend_idx:
        return
      self._hovered_legend_idx = hovered_idx
      labels = list(self._line_by_label.keys())
      for idx, label in enumerate(labels):
        line = self._line_by_label[label]
        if not line.get_visible():
          continue
        if hovered_idx is None:
          line.set_alpha(0.95)
          line.set_linewidth(2.0)
        elif idx == hovered_idx:
          line.set_alpha(1.0)
          line.set_linewidth(3.0)
        else:
          line.set_alpha(0.18)
          line.set_linewidth(1.5)
      self.canvas.draw_idle()

    def show_results(
      self,
      weights: CriterionWeights,
      rows: List[Tuple[str, Optional[EfficiencyTriple], float, int]],
    ) -> None:
      w1, w2, w3 = weights.display_parts(digits=2)
      lines = [
        (
          f"Веса (вход): w₁ результативность={w1}, "
          f"w₂ оперативность={w2}, w₃ ресурсоэкономность={w3} "
          f"(Σ=1)"
        ),
        "Эффективность сгенерированного интерфейса (выход), сортировка по F:",
      ]
      for name, triple, fitness, form_count in rows:
        if triple is None:
          lines.append(f"• {name}: нет решения")
          continue
        lines.append(
          f"• {name}: P={triple.potency:.3f}, O={triple.operativeness:.3f}, "
          f"R={triple.resource_saving:.3f}, F={fitness:.3f}, экранов={form_count}"
        )
      self.results_label.setText("\n".join(lines))


  class VisualizationTab(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      self._summaries: List[AlgorithmRunSummary] = []
      self._preview_dir = Path(tempfile.gettempdir()) / "optiflow_previews"
      self._preview_dir.mkdir(parents=True, exist_ok=True)

      layout = QtWidgets.QVBoxLayout(self)
      top = QtWidgets.QHBoxLayout()
      top.addWidget(QtWidgets.QLabel("Алгоритм:"))
      self.algorithm_combo = QtWidgets.QComboBox()
      self.algorithm_combo.setMinimumWidth(360)
      self.algorithm_combo.currentIndexChanged.connect(self._show_selected)
      top.addWidget(self.algorithm_combo, stretch=1)
      layout.addLayout(top)

      self.metrics_label = QtWidgets.QLabel(
        "Запустите оптимизацию, чтобы просмотреть сгенерированные интерфейсы."
      )
      self.metrics_label.setWordWrap(True)
      layout.addWidget(self.metrics_label)

      if _HAS_WEBENGINE:
        self.web_view: Optional[QtWidgets.QWidget] = QWebEngineView()
        layout.addWidget(self.web_view, stretch=1)
      else:
        self.web_view = None
        fallback = QtWidgets.QLabel(
          "Для интерактивного просмотра установите PyQtWebEngine "
          "(pip install PyQtWebEngine). Ниже — упрощённый предпросмотр первого шага."
        )
        fallback.setWordWrap(True)
        layout.addWidget(fallback)
        self.fallback_browser = QtWidgets.QTextBrowser()
        self.fallback_browser.setOpenExternalLinks(True)
        layout.addWidget(self.fallback_browser, stretch=1)

    def show_results(self, summaries: List[AlgorithmRunSummary]) -> None:
      self._summaries = sorted(
        summaries_with_layouts(summaries),
        key=lambda item: item.fitness,
        reverse=True,
      )
      self.algorithm_combo.blockSignals(True)
      self.algorithm_combo.clear()
      for item in self._summaries:
        self.algorithm_combo.addItem(
          f"{item.label} — F={item.fitness:.4f}",
          item.key,
        )
      self.algorithm_combo.blockSignals(False)
      if self._summaries:
        self.algorithm_combo.setCurrentIndex(0)
        self._show_selected()
      else:
        self.metrics_label.setText("Нет алгоритмов с готовым layout для визуализации.")
        if self.web_view is not None and _HAS_WEBENGINE:
          self.web_view.setHtml("<p>Нет данных для отображения.</p>")  # type: ignore[union-attr]
        elif hasattr(self, "fallback_browser"):
          self.fallback_browser.setHtml("<p>Нет данных для отображения.</p>")

    def _show_selected(self) -> None:
      if not self._summaries or self.algorithm_combo.count() == 0:
        return
      key = self.algorithm_combo.currentData()
      item = next((s for s in self._summaries if s.key == key), None)
      if item is None or item.layout is None or item.triple is None:
        return
      self.metrics_label.setText(
        f"P={item.triple.potency:.3f}  O={item.triple.operativeness:.3f}  "
        f"R={item.triple.resource_saving:.3f}  F={item.fitness:.4f}  "
        f"экранов={item.form_count}  шагов истории={item.history_steps}\n"
        f"Контролы: {', '.join(c.name for c in item.layout.controls_flat())}"
      )
      html = generate_html_from_layout(item.layout)
      preview_path = self._preview_dir / f"{item.key}.html"
      preview_path.write_text(html, encoding="utf-8")
      if self.web_view is not None and _HAS_WEBENGINE:
        self.web_view.load(QtCore.QUrl.fromLocalFile(str(preview_path.resolve())))  # type: ignore[union-attr]
      elif hasattr(self, "fallback_browser"):
        self.fallback_browser.setHtml(html)


  class ReportTab(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      layout = QtWidgets.QVBoxLayout(self)
      self.report_edit = QtWidgets.QPlainTextEdit()
      self.report_edit.setReadOnly(True)
      self.report_edit.setPlaceholderText(
        "Запустите оптимизацию — здесь появится текстовый отчёт по всем алгоритмам."
      )
      self.report_edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
      layout.addWidget(self.report_edit)

    def show_report(self, text: str) -> None:
      self.report_edit.setPlainText(text)


  class GeminiInterpretWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
      self,
      api_key: str,
      prompt: str,
      model: str,
      parent=None,
    ) -> None:
      super().__init__(parent)
      self._api_key = api_key
      self._prompt = prompt
      self._model = model

    def run(self) -> None:
      try:
        text = request_gemini_flash(self._api_key, self._prompt, model=self._model)
        self.completed.emit(text)
      except Exception as exc:
        self.failed.emit(f"{type(exc).__name__}: {exc}")


  class InterpretationTab(QtWidgets.QWidget):
    _SETTINGS_KEY = "gemini_api_key"
    _MODEL_SETTINGS_KEY = "gemini_model_id"

    def __init__(self, parent=None) -> None:
      super().__init__(parent)
      self._settings = QtCore.QSettings("OptiFlow", "OptiFlow")
      self._summaries: List[AlgorithmRunSummary] = []
      self._report_text = ""
      self._fields: List[FieldSpec] = []
      self._max_forms = 1
      self._weights = CriterionWeights.balanced()
      self._cancelled = False
      self._warning: Optional[str] = None
      self._worker: Optional[GeminiInterpretWorker] = None

      layout = QtWidgets.QVBoxLayout(self)

      key_row = QtWidgets.QHBoxLayout()
      key_row.addWidget(QtWidgets.QLabel("API-ключ Gemini Flash:"))
      self.api_key_edit = QtWidgets.QLineEdit()
      self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
      self.api_key_edit.setPlaceholderText("AIza…")
      saved_key = self._settings.value(self._SETTINGS_KEY, "", type=str)
      if saved_key:
        self.api_key_edit.setText(saved_key)
      key_row.addWidget(self.api_key_edit, stretch=1)
      layout.addLayout(key_row)

      model_row = QtWidgets.QHBoxLayout()
      model_row.addWidget(QtWidgets.QLabel("Модель Gemini Flash:"))
      self.model_combo = QtWidgets.QComboBox()
      saved_model = self._settings.value(
        self._MODEL_SETTINGS_KEY, DEFAULT_GEMINI_MODEL, type=str
      )
      saved_index = 0
      for index, (label, model_id) in enumerate(GEMINI_MODEL_CHOICES):
        self.model_combo.addItem(label, model_id)
        if model_id == saved_model:
          saved_index = index
      self.model_combo.setCurrentIndex(saved_index)
      self.model_combo.setMinimumWidth(280)
      model_row.addWidget(self.model_combo, stretch=1)
      layout.addLayout(model_row)

      actions = QtWidgets.QHBoxLayout()
      self.send_btn = QtWidgets.QPushButton("Отправить")
      self.send_btn.clicked.connect(self._send_interpretation)
      actions.addWidget(self.send_btn)
      self.status_label = QtWidgets.QLabel("")
      actions.addWidget(self.status_label, stretch=1)
      layout.addLayout(actions)

      hint = QtWidgets.QLabel(
        "Сначала запустите оптимизацию. По кнопке «Отправить» формируется промпт "
        "из постановки задачи, отчёта и layout всех алгоритмов, затем запрос уходит в Gemini Flash. "
        "Ключ хранится локально в настройках приложения."
      )
      hint.setWordWrap(True)
      layout.addWidget(hint)

      self.result_edit = QtWidgets.QPlainTextEdit()
      self.result_edit.setReadOnly(True)
      self.result_edit.setPlaceholderText(
        "Здесь появится интерпретация результатов от Gemini Flash."
      )
      layout.addWidget(self.result_edit, stretch=1)

    def set_run_context(
      self,
      *,
      summaries: List[AlgorithmRunSummary],
      report_text: str,
      fields: List[FieldSpec],
      max_forms: int,
      weights: CriterionWeights,
      cancelled: bool,
      warning: Optional[str],
    ) -> None:
      self._summaries = summaries
      self._report_text = report_text
      self._fields = fields
      self._max_forms = max_forms
      self._weights = weights
      self._cancelled = cancelled
      self._warning = warning
      self.status_label.setText("Данные прогона обновлены.")

    def _send_interpretation(self) -> None:
      if self._worker is not None and self._worker.isRunning():
        return
      if not self._summaries:
        QtWidgets.QMessageBox.warning(
          self,
          "Интерпретация",
          "Нет данных прогона. Сначала нажмите «Запустить» на вкладке «Алгоритмы».",
        )
        return
      api_key = self.api_key_edit.text().strip()
      if not api_key:
        QtWidgets.QMessageBox.warning(self, "Интерпретация", "Введите API-ключ Gemini Flash.")
        return

      prompt = build_interpretation_prompt(
        summaries=self._summaries,
        report_text=self._report_text,
        fields=self._fields,
        max_forms=self._max_forms,
        weights=self._weights,
        cancelled=self._cancelled,
        warning=self._warning,
        optiflow_version=__version__,
      )

      self.send_btn.setEnabled(False)
      model_id = str(self.model_combo.currentData() or DEFAULT_GEMINI_MODEL)
      self.status_label.setText(
        f"Запрос к {gemini_model_label(model_id)}…"
      )
      self.result_edit.setPlainText(f"Ожидание ответа ({gemini_model_label(model_id)})…\n")

      self._worker = GeminiInterpretWorker(api_key, prompt, model_id, parent=self)
      self._worker.completed.connect(self._on_completed)
      self._worker.failed.connect(self._on_failed)
      self._worker.finished.connect(self._clear_worker)
      self._worker.start()

    def _on_completed(self, text: str) -> None:
      self._settings.setValue(self._SETTINGS_KEY, self.api_key_edit.text().strip())
      model_id = str(self.model_combo.currentData() or DEFAULT_GEMINI_MODEL)
      self._settings.setValue(self._MODEL_SETTINGS_KEY, model_id)
      self.result_edit.setPlainText(text)
      self.status_label.setText("Интерпретация получена.")
      self.send_btn.setEnabled(True)

    def _on_failed(self, message: str) -> None:
      self.result_edit.setPlainText(f"Ошибка:\n{message}")
      self.status_label.setText("Ошибка запроса.")
      self.send_btn.setEnabled(True)
      QtWidgets.QMessageBox.critical(self, "Gemini Flash", message)

    def _clear_worker(self) -> None:
      self._worker = None


  class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
      super().__init__()
      self.setWindowTitle(f"OptiFlow {__version__}")
      self.resize(1100, 800)
      self.registry = FunctionRegistry()
      self.weights = CriterionWeights.balanced()
      self.max_forms = 3

      tabs = QtWidgets.QTabWidget()
      self.setCentralWidget(tabs)

      self.data_tab = DataTaskTab()
      self.data_tab.maxFormsChanged.connect(self._set_max_forms)
      self.data_tab.weightsChanged.connect(self._set_weights)
      self.max_forms = self.data_tab.max_forms
      self.weights = self.data_tab.coef.weights()

      self.task_tab = TaskSettingsTab(self.registry)
      self.alg_tab = AlgorithmsTab()
      self.alg_tab.runRequested.connect(self.run_algorithms)
      self.charts_tab = ChartsTab()
      self.visualization_tab = VisualizationTab()
      self.report_tab = ReportTab()
      self.interpretation_tab = InterpretationTab()

      tabs.addTab(self.data_tab, "Данные, тип, длина")
      tabs.addTab(self.alg_tab, "Алгоритмы")
      tabs.addTab(self.task_tab, "Настройка задачи")
      tabs.addTab(self.charts_tab, "Графики")
      tabs.addTab(self.visualization_tab, "Визуализация")
      tabs.addTab(self.report_tab, "Отчёт")
      tabs.addTab(self.interpretation_tab, "Интерпретация")

      file_menu = self.menuBar().addMenu("Файл")
      export_action = QtWidgets.QAction("Экспорт веб-страницы", self)
      export_action.triggered.connect(self.export_web)
      file_menu.addAction(export_action)

      self.last_best_layouts: Dict[str, Optional[InterfaceLayout]] = {}
      self._last_run_summaries: List[AlgorithmRunSummary] = []
      self._last_report_text = ""
      self._last_run_cancelled = False
      self._last_run_warning: Optional[str] = None
      self._worker: Optional[OptimizationWorker] = None
      self._run_control: Optional[OptimizationControl] = None
      self.overlay = ProgressOverlay(self)
      self.overlay.cancelRequested.connect(self._cancel_optimization)
      self.overlay.hide()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
      super().resizeEvent(event)
      self.overlay.setGeometry(self.rect())

    def _cancel_optimization(self) -> None:
      if self._run_control is not None:
        self._run_control.cancel()

    def _set_weights(self, weights: CriterionWeights) -> None:
      self.weights = weights

    def _set_max_forms(self, value: int) -> None:
      self.max_forms = int(value)

    def _build_space(self) -> Tuple[DecisionSpace, ObjectiveEvaluator]:
      self.data_tab.coef.commit()
      self.weights = self.data_tab.coef.weights()
      fields = self.data_tab.field_table.fields()
      _ensure_allowed_controls(fields)
      space = DecisionSpace(fields, max_forms=self.max_forms)
      evaluator = ObjectiveEvaluator(self.registry, self.weights)
      return space, evaluator

    def run_algorithms(self) -> None:
      if self._worker is not None and self._worker.isRunning():
        return
      space, evaluator = self._build_space()
      params_by_label: Dict[str, Dict[str, float]] = {}
      for _, label in SUITE_STEPS:
        params = self.alg_tab.params_for(label)
        params.update(self.task_tab.limits_for(label))
        params_by_label[label] = params
      self._worker = OptimizationWorker(space, evaluator, params_by_label, parent=self)
      self._run_control = OptimizationControl(on_progress=self._worker.progress.emit)
      self._worker.control = self._run_control
      self._worker.progress.connect(self.overlay.apply_report)
      self._worker.completed.connect(self._on_suite_finished)
      self._worker.failed.connect(self._on_suite_failed)
      self._worker.finished.connect(self._clear_worker)
      self.alg_tab.set_run_enabled(False)
      self.overlay.setGeometry(self.rect())
      self.overlay.start()
      self._worker.start()

    def _on_suite_finished(self, payload: object) -> None:
      data = payload if isinstance(payload, dict) else {}
      results: Dict[str, Dict[str, object]] = data.get("results") or {}
      warning = data.get("warning")
      cancelled = bool(data.get("cancelled"))
      self.last_best_layouts = {}
      for key, _label in SUITE_STEPS:
        payload_one = results.get(key) or {}
        self.last_best_layouts[key] = payload_one.get("best_layout")
      summaries = build_algorithm_summaries(results, self.registry, self.weights)
      self._last_run_summaries = summaries
      self.charts_tab.plot_histories(histories_from_results(results))
      result_rows: List[Tuple[str, Optional[EfficiencyTriple], float, int]] = []
      for item in sorted(
        summaries,
        key=lambda s: (s.fitness if s.layout is not None else -1.0),
        reverse=True,
      ):
        if item.layout is None:
          result_rows.append((item.key, None, 0.0, 0))
          continue
        result_rows.append((item.key, item.triple, item.fitness, item.form_count))
      self.charts_tab.show_results(self.weights, result_rows)
      self.visualization_tab.show_results(summaries)
      fields = self.data_tab.field_table.fields()
      total_elapsed = data.get("total_elapsed_s")
      total_elapsed_s = float(total_elapsed) if total_elapsed is not None else None
      report_text = format_optimization_report(
        summaries,
        weights=self.weights,
        field_count=len(fields),
        max_forms=self.max_forms,
        cancelled=cancelled,
        warning=str(warning) if warning else None,
        optiflow_version=__version__,
        total_elapsed_s=total_elapsed_s,
        fields=fields,
      )
      self.report_tab.show_report(report_text)
      self._last_report_text = report_text
      self._last_run_cancelled = cancelled
      self._last_run_warning = str(warning) if warning else None
      self.interpretation_tab.set_run_context(
        summaries=summaries,
        report_text=report_text,
        fields=self.data_tab.field_table.fields(),
        max_forms=self.max_forms,
        weights=self.weights,
        cancelled=cancelled,
        warning=self._last_run_warning,
      )
      self.overlay.stop()
      self.alg_tab.set_run_enabled(True)
      if warning:
        QtWidgets.QMessageBox.warning(self, "Полный перебор (Brute Force)", str(warning))
      if cancelled:
        QtWidgets.QMessageBox.information(
          self,
          "Оптимизация прервана",
          "Расчёт остановлен. Сохранён лучший найденный на этот момент набор решений.",
        )
      tabs = self.centralWidget()
      if isinstance(tabs, QtWidgets.QTabWidget):
        tabs.setCurrentWidget(self.visualization_tab)

    def _on_suite_failed(self, message: str) -> None:
      self.overlay.stop()
      self.alg_tab.set_run_enabled(True)
      QtWidgets.QMessageBox.critical(self, "Ошибка оптимизации", message)

    def _clear_worker(self) -> None:
      self._worker = None
      self._run_control = None

    def export_web(self) -> None:
      layout: Optional[InterfaceLayout] = None
      preferred = self.alg_tab.current_algorithm()
      name_map = {
        "Многокритериальный (NSGA-II)": "NSGA-II",
        "Классический генетический алгоритм (GA)": "GA",
        "Полный перебор (Brute Force)": "BruteForce",
        "Локальный поиск (Hill Climb)": "HillClimb",
        "Алгоритм роя частиц (PSO)": "PSO",
        "Случайный поиск": "Random",
        "Жадный (Greedy)": "Greedy",
        "Имитация отжига (SA)": "SA",
        "Поиск с запретами (Tabu)": "Tabu",
        "Муравьиный алгоритм (ACO)": "ACO",
      }
      preferred_key = name_map.get(preferred, preferred)
      key_order = [
        preferred_key,
        "BruteForce",
        "GA",
        "NSGA-II",
        "PSO",
        "HillClimb",
        "Random",
        "Greedy",
        "SA",
        "Tabu",
        "ACO",
      ]
      for key in key_order:
        if self.last_best_layouts.get(key):
          layout = self.last_best_layouts[key]
          break
      if layout is None:
        fields = self.data_tab.field_table.fields()
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
  weights: CriterionWeights,
) -> Tuple[int, float]:
  """Prefer a multi-step wizard, then weighted scalar fitness."""
  multistep = int(layout.form_count > 1)
  fitness = calculate_fitness(triple, weights)
  return (multistep, fitness)


def run_headless_cli(output_path: str | Path = "wizard_output.html") -> Path:
  """Run optimization without PyQt5 and write partitioned multi-step wizard HTML."""
  registry = FunctionRegistry()
  fields = mock_demo_fields()
  max_forms = 3
  weights = CriterionWeights.from_raw(*WEIGHT_PRESETS["Упор на оперативность"])
  space = DecisionSpace(fields, max_forms=max_forms)
  evaluator = ObjectiveEvaluator(registry, weights)

  algorithm_runs: List[Tuple[str, Dict[str, object]]] = [
    ("NSGA-II", nsga2(space, evaluator, pop_size=30, generations=25, random_seed=42)),
    ("Greedy", greedy(space, evaluator)),
    ("PSO", pso(space, evaluator, swarm_size=24, iterations=40, random_seed=42)),
  ]

  best_name = ""
  best_layout: Optional[InterfaceLayout] = None
  best_triple = None
  best_rank: Tuple[int, float] = (-1, -1.0)

  for name, result in algorithm_runs:
    layout = result.get("best_layout")
    if layout is None:
      continue
    triple = compute_total_efficiency(layout, registry)
    rank = _rank_layout_candidate(triple, layout, weights)
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
    best_rank = (1, calculate_fitness(best_triple, weights))
    best_name = f"{best_name or 'forced'}-multi"

  best_fitness = best_rank[1]

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
    best_fitness = calculate_fitness(best_triple, weights)
    best_name = "fallback"

  assert best_layout is not None and best_triple is not None
  w1, w2, w3 = weights.display_parts(digits=2)
  print(
    f"[{best_name}] F={best_fitness:.4f} "
    f"P={best_triple.potency:.3f} O={best_triple.operativeness:.3f} R={best_triple.resource_saving:.3f} "
    f"weights=({w1},{w2},{w3}) forms={best_layout.form_count}"
  )

  html = generate_html_from_layout(best_layout)
  if "<section" not in html:
    raise RuntimeError("Generated HTML must contain wizard <section> steps")

  out = Path(output_path)
  if not out.is_absolute():
    out = _APP_ROOT / out
  out.write_text(html, encoding="utf-8")
  print(f"Wrote wizard layout ({best_layout.form_count} forms) -> {out.resolve()}")

  logging.basicConfig(level=logging.INFO)
  run_optimization_benchmark(runs_count=20, random_seed=42, output_dir=out.parent, log_markdown=True)
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
