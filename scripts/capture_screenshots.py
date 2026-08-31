#!/usr/bin/env python3
"""Capture OptiFlow GUI tabs into docs/screenshots/optiflow_XX_*.webp."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from PyQt5 import QtCore, QtWidgets

from optiflow.app import MainWindow
from optiflow.optimization.runner import SUITE_STEPS, run_optimization_suite

SHOT_DIR = ROOT / "docs" / "screenshots"

SHOTS = (
  (0, "optiflow_01_data.webp"),
  (1, "optiflow_02_algorithms.webp"),
  (2, "optiflow_03_task_settings.webp"),
  (3, "optiflow_04_charts.webp"),
  (4, "optiflow_05_visualization.webp"),
  (5, "optiflow_06_report.webp"),
  (6, "optiflow_07_interpretation.webp"),
)


def _wait(app: QtWidgets.QApplication, ms: int) -> None:
  loop = QtCore.QEventLoop()
  QtCore.QTimer.singleShot(ms, loop.quit)
  loop.exec_()
  app.processEvents()


def _grab_window(window: QtWidgets.QMainWindow, dest_webp: Path) -> None:
  window.raise_()
  window.activateWindow()
  QtWidgets.QApplication.processEvents()
  screen = QtWidgets.QApplication.primaryScreen()
  if screen is None:
    raise RuntimeError("No primary screen for screenshots.")
  pixmap = screen.grabWindow(int(window.winId()))
  if pixmap.isNull():
    pixmap = window.grab()
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    png_path = Path(tmp.name)
  if not pixmap.save(str(png_path), "PNG"):
    raise RuntimeError(f"Failed to write PNG for {dest_webp.name}")
  dest_webp.parent.mkdir(parents=True, exist_ok=True)
  subprocess.run(
    ["cwebp", "-q", "82", "-m", "6", str(png_path), "-o", str(dest_webp)],
    check=True,
    capture_output=True,
  )
  png_path.unlink(missing_ok=True)


def _short_suite_params(window: MainWindow) -> dict:
  params: dict = {}
  for _key, label in SUITE_STEPS:
    item = window.alg_tab.params_for(label)
    item.update(window.task_tab.limits_for(label))
    for key in ("generations", "iterations"):
      if key in item:
        item[key] = min(float(item[key]), 12.0)
    if "pop_size" in item:
      item["pop_size"] = min(float(item["pop_size"]), 20.0)
    if "swarm_size" in item:
      item["swarm_size"] = min(float(item["swarm_size"]), 16.0)
    params[label] = item
  return params


def _wait_visualization(app: QtWidgets.QApplication, window: MainWindow) -> None:
  view = window.visualization_tab.web_view
  if view is None:
    _wait(app, 400)
    return
  loop = QtCore.QEventLoop()
  view.loadFinished.connect(lambda _ok: loop.quit())
  QtCore.QTimer.singleShot(6000, loop.quit)
  loop.exec_()
  _wait(app, 500)


def main() -> int:
  app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
  window = MainWindow()
  window.interpretation_tab.api_key_edit.clear()
  window.resize(1100, 800)
  window.show()
  _wait(app, 400)

  window.tabs.setCurrentIndex(0)
  _wait(app, 200)
  _grab_window(window, SHOT_DIR / SHOTS[0][1])

  window.tabs.setCurrentIndex(1)
  _wait(app, 200)
  _grab_window(window, SHOT_DIR / SHOTS[1][1])

  window.tabs.setCurrentIndex(2)
  _wait(app, 200)
  _grab_window(window, SHOT_DIR / SHOTS[2][1])

  space, evaluator = window._build_space()
  payload = run_optimization_suite(space, evaluator, _short_suite_params(window))
  window._on_suite_finished(payload)
  _wait(app, 400)

  window.tabs.setCurrentIndex(3)
  window.charts_tab.canvas.draw()
  _wait(app, 400)
  _grab_window(window, SHOT_DIR / SHOTS[3][1])

  window.tabs.setCurrentIndex(4)
  _wait_visualization(app, window)
  _grab_window(window, SHOT_DIR / SHOTS[4][1])

  window.tabs.setCurrentIndex(5)
  _wait(app, 200)
  _grab_window(window, SHOT_DIR / SHOTS[5][1])

  window.tabs.setCurrentIndex(6)
  _wait(app, 200)
  _grab_window(window, SHOT_DIR / SHOTS[6][1])

  window.close()
  print(f"Wrote screenshots to {SHOT_DIR}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
