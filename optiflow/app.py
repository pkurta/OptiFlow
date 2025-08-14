from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from optiflow.models.scoring import (
    ControlType,
    DataType,
    FieldSpec,
    FunctionRegistry,
)
from optiflow.optimization.algorithms import (
    DecisionSpace,
    ObjectiveEvaluator,
    hill_climb,
    nsga2,
    pso,
    random_search,
)
from optiflow.ui.web_generator import generate_html


def normalize_weights(a: float, b: float, c: float) -> Tuple[float, float, float]:
    s = max(1e-9, a + b + c)
    return (a / s, b / s, c / s)


class CoefficientSliders(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)
        self.s1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.s2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.s3 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        for s in (self.s1, self.s2, self.s3):
            s.setRange(0, 100)
            s.setValue(33)
            s.valueChanged.connect(self._rebalance)
        self.l1 = QtWidgets.QLabel("Результативность: 0.33")
        self.l2 = QtWidgets.QLabel("Оперативность: 0.33")
        self.l3 = QtWidgets.QLabel("Ресурсоэкономность: 0.34")
        layout.addWidget(self.l1, 0, 0)
        layout.addWidget(self.s1, 0, 1)
        layout.addWidget(self.l2, 1, 0)
        layout.addWidget(self.s2, 1, 1)
        layout.addWidget(self.l3, 2, 0)
        layout.addWidget(self.s3, 2, 1)
        self._rebalance()

    def _rebalance(self) -> None:
        v1, v2, v3 = self.s1.value(), self.s2.value(), self.s3.value()
        s = v1 + v2 + v3
        if s <= 0:
            v1 = v2 = v3 = 1
            s = 3
        w1, w2, w3 = normalize_weights(v1, v2, v3)
        self.l1.setText(f"Результативность: {w1:.2f}")
        self.l2.setText(f"Оперативность: {w2:.2f}")
        self.l3.setText(f"Ресурсоэкономность: {w3:.2f}")
        self.changed.emit(w1, w2, w3)

    def weights(self) -> Tuple[float, float, float]:
        v1, v2, v3 = self.s1.value(), self.s2.value(), self.s3.value()
        return normalize_weights(v1, v2, v3)


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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QFormLayout(self)
        # Common params
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItems(["NSGA-II", "Градиентный спуск (Hill Climb)", "PSO", "Случайный поиск"])
        layout.addRow("Алгоритм:", self.algorithm_combo)

        self.param1 = QtWidgets.QSpinBox()
        self.param1.setRange(10, 1000)
        self.param1.setValue(40)
        layout.addRow("Размер популяции/рой/итерации:", self.param1)

        self.param2 = QtWidgets.QSpinBox()
        self.param2.setRange(1, 1000)
        self.param2.setValue(40)
        layout.addRow("Поколения/итерации:", self.param2)

        self.run_btn = QtWidgets.QPushButton("Запустить")
        self.run_btn.clicked.connect(self.runRequested.emit)
        layout.addRow(self.run_btn)

    def current_algorithm(self) -> str:
        return self.algorithm_combo.currentText()

    def params(self) -> Tuple[int, int]:
        return self.param1.value(), self.param2.value()


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
        ax.set_title("Сравнение эффективности алгоритмов (лучший найденный скор)")
        ax.set_xlabel("Итерации")
        ax.set_ylabel("Скор")
        ax.legend()
        self.canvas.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OptiFlow")
        self.resize(1100, 800)
        self.registry = FunctionRegistry()
        self.weights = (1/3, 1/3, 1/3)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Tab 1: Data
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
        self.coef = CoefficientSliders()
        self.coef.changed.connect(self._set_weights)
        data_layout.addWidget(self.coef)

        # Tab 2: Task settings (functions)
        self.task_tab = FunctionEditor(self.registry)

        # Tab 3: Algorithms
        self.alg_tab = AlgorithmsTab()
        self.alg_tab.runRequested.connect(self.run_algorithms)

        # Tab 4: Charts
        self.charts_tab = ChartsTab()

        tabs.addTab(self.data_tab, "Данные, тип, длина")
        tabs.addTab(self.task_tab, "Настройка задачи")
        tabs.addTab(self.alg_tab, "Алгоритмы")
        tabs.addTab(self.charts_tab, "Графики")

        # Menu
        file_menu = self.menuBar().addMenu("Файл")
        export_action = QtWidgets.QAction("Экспорт веб-страницы", self)
        export_action.triggered.connect(self.export_web)
        file_menu.addAction(export_action)

        # Some defaults
        self.field_table.add_field("Возраст", DataType.NUMBER, 3)
        self.field_table.add_field("Имя", DataType.SHORT_TEXT, 16)
        self.field_table.add_field("Согласие", DataType.BOOLEAN, 1)

    def _add_field(self) -> None:
        self.field_table.add_field("Поле", DataType.SHORT_TEXT, 16)

    def _remove_selected_field(self) -> None:
        rows = sorted({i.row() for i in self.field_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.field_table.removeRow(r)

    def _set_weights(self, a: float, b: float, c: float) -> None:
        self.weights = (a, b, c)

    def _build_space(self) -> Tuple[DecisionSpace, ObjectiveEvaluator]:
        fields = self.field_table.fields()
        space = DecisionSpace(fields)
        evaluator = ObjectiveEvaluator(self.registry, self.weights)
        return space, evaluator

    def run_algorithms(self) -> None:
        space, evaluator = self._build_space()
        alg = self.alg_tab.current_algorithm()
        p1, p2 = self.alg_tab.params()
        histories: List[Tuple[str, List[float]]] = []

        # Always run all for comparison and plot
        res_nsga = nsga2(space, evaluator, pop_size=p1, generations=p2)
        res_hc = hill_climb(space, evaluator, iterations=p1)
        res_pso = pso(space, evaluator, swarm_size=p1, iterations=p2)
        res_rs = random_search(space, evaluator, iterations=p1)
        histories.append(("NSGA-II", res_nsga["history"]))
        histories.append(("HillClimb", res_hc["history"]))
        histories.append(("PSO", res_pso["history"]))
        histories.append(("Random", res_rs["history"]))
        self.charts_tab.plot_histories(histories)

        # store last best controls
        self.last_best_controls = {
            "NSGA-II": res_nsga.get("best_controls"),
            "HillClimb": res_hc.get("best_controls"),
            "PSO": res_pso.get("best_controls"),
            "Random": res_rs.get("best_controls"),
        }

    def export_web(self) -> None:
        fields = self.field_table.fields()
        # choose NSGA result first else any
        controls = None
        # prefer selected algorithm if available
        preferred = self.alg_tab.current_algorithm()
        key_order = [preferred, "NSGA-II", "PSO", "HillClimb", "Random"]
        for key in key_order:
            if hasattr(self, "last_best_controls") and self.last_best_controls.get(key):
                controls = self.last_best_controls[key]
                break
        if controls is None:
            QtWidgets.QMessageBox.warning(self, "Экспорт", "Сначала запустите алгоритмы")
            return
        html = generate_html(fields, controls)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить HTML", os.path.expanduser("~/optiflow.html"), "HTML Files (*.html)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        QtWidgets.QMessageBox.information(self, "Экспорт", f"Сохранено: {path}")


def run_app() -> None:
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()


