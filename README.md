# OptiFlow

How to run:

Activate venv: source /Users/pavelkurta/My/workspace/OptiFlow/.venv/bin/activate

Launch: python -m optiflow.app

**v1**
* Built a PyQt desktop app with tabs: “Данные, тип, длина”, “Настройка задачи”, “Алгоритмы”, “Графики”.
* Implemented coefficient sliders that always normalize to sum=1.
* Designed a function registry for per-control scoring with safe exec. Added sensible defaults for all listed controls.
* Added algorithms: NSGA-II-style MOEA, hill climbing (as gradient descent proxy on discrete space), PSO, and random search. Plots compare best-score histories.
* Implemented HTML export for the chosen/best layout.

You can edit per-control functions in the “Настройка задачи” tab. Define fields and sizes in “Данные, тип, длина”, tune weights via sliders, choose algorithm and params in “Алгоритмы”, run, view comparison in “Графики”, and export HTML via menu “Файл → Экспорт веб-страницы”.
