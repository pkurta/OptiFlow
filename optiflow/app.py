from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

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
    aco,
    greedy,
    hill_climb,
    nsga2,
    pso,
    random_search,
    simulated_annealing,
    tabu_search,
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
        # Common params
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

        # Always run all for comparison and plot (неселектed берут дефолты)
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

        # store last best controls
        self.last_best_controls = {
            "NSGA-II": res_nsga.get("best_controls"),
            "HillClimb": res_hc.get("best_controls"),
            "PSO": res_pso.get("best_controls"),
            "Random": res_rs.get("best_controls"),
            "Greedy": res_greedy.get("best_controls"),
            "SA": res_sa.get("best_controls"),
            "Tabu": res_tabu.get("best_controls"),
            "ACO": res_aco.get("best_controls"),
        }

    def export_web(self) -> None:
        fields = self.field_table.fields()
        # choose NSGA result first else any
        controls = None
        # prefer selected algorithm if available
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


