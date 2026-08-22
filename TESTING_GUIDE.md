# OptiFlow: Руководство по тестированию и бенчмаркингу оптимизационного движка

**Версия документа:** 1.0  
**Целевая аудитория:** инженеры QA, разработчики ядра оптимизации, научные сопровождающие проекта  
**Система:** OptiFlow — параметрический синтез пользовательских интерфейсов по модели Kurta-Izrailov (2025)

---

## 1. Введение и Научный Базис

### 1.1. Назначение тестового контура

Автоматизированный тестовый контур OptiFlow предназначен для верификации **математической корректности** оптимизационного движка относительно мультипликативной модели эффективности Kurta-Izrailov (2025). Модель оперирует тройкой показателей `EfficiencyTriple` — **Результативность (potency)**, **Оперативность (operativeness)** и **Ресурсоэкономность (resource_saving)** — и вычисляет суммарную эффективность компоновки интерфейса как **покомпонентное произведение** скорректированных атомарных вкладов форм и элементов управления.

Фундаментальное уравнение (1), реализованное в функции `compute_total_efficiency()` модуля `optiflow/optimization/algorithms.py`:

```
Total = ∏(corrected forms) × ∏(double-corrected elements)
```

где каждый множитель — это `EfficiencyTriple`, перемножаемый через перегрузку оператора `*` в классе `EfficiencyTriple` (`optiflow/models/scoring.py`). Коррекции позиции элемента и индекса шага мастера применяются через конвейер `apply_element_position_correction()` и `apply_form_step_correction()`.

Тестовый набор проверяет три уровня соответствия:

1. **Структурная корректность** — декодирование хромосомы `DecisionSpace` (D токенов управления + N весов разбиения) в валидный `InterfaceLayout`.
2. **Численная корректность** — согласованность скалярной пригодности `calculate_fitness()` с нормированными весами `CriterionWeights` (\(F = w_1 P + w_2 O + w_3 R\)).
3. **Оптимизационная корректность** — способность эвристик приближаться к абсолютному глобальному максимуму, найденному полным перебором.

### 1.2. Brute Force как эталон Ground Truth

Алгоритм **Brute Force** (`brute_force()` в `optiflow/optimization/algorithms.py`) выполняет **исчерпывающий перебор** всех допустимых комбинаций:

- для каждого поля — все допустимые типы UI-контролов (`allowed_controls()`);
- для каждого разбиения — все композиции `D` полей по `N` формам мастера (метод «stars and bars», функция `_enumerate_field_partitions()`).

Размер пространства поиска вычисляется функцией `brute_force_search_space_size()`:

```
|Ω| = (∏ᵢ |allowed_controls(fieldᵢ)|) × C(D + N − 1, N − 1)
```

где `D = len(fields)`, `N = max(1, max_forms)`.

**Роль эталона:** результат Brute Force (`best_score`, `best_layout`) является **абсолютным глобальным максимумом** скалярной пригодности \(F(w)\) в рамках заданного `DecisionSpace` и `CriterionWeights`. Все метаэвристики (NSGA-II, Classic GA, PSO, Greedy, SA, Tabu, ACO) сравниваются с этим значением при вычислении **Precision Rate**.

**Ограничение безопасности:** если `|Ω| > 50 000`, функция `brute_force()` генерирует исключение `ValueError` с пояснением о превышении лимита `BRUTE_FORCE_MAX_COMBINATIONS`. Это предотвращает зависание потока при неконтролируемом росте пространства поиска. В Monte Carlo-бенчмарке такие прогоны помечаются как «baseline не вычислим», и метрика Precision Rate для них не накапливается.

Unit-тест `test_brute_force_finds_global_maximum` независимо пересчитывает все комбинации и проверяет, что `best_score` совпадает с `max(exhaustive_scores)` с точностью `assertAlmostEqual`.

### 1.3. Метрики Monte Carlo-бенчмарка

Monte Carlo-движок (`run_optimization_benchmark()` в `optiflow/benchmarks.py`) выполняет серию независимых прогонов на **идентичных случайных снимках данных** и агрегирует три ключевые метрики:

#### Precision Rate (точность эвристик относительно перебора)

Для каждого прогона, где Brute Force успешно вычислен (`baseline_score > 0`):

```
Precision = min(1.0, max(0.0, score_algorithm / score_brute_force))
```

Реализовано в функции `precision_vs_baseline()`. Значение `1.0` означает достижение глобального максимума. Для Brute Force в отчёте всегда фиксируется `1.0` на вычислимых прогонах. Итоговая метрика в отчёте — **среднее арифметическое** по всем накопленным значениям (`AlgorithmBenchmarkStats.mean_precision()`).

#### Convergence Speed (скорость выхода на плато)

Прокси-метрика скорости сходимости, вычисляемая функцией `convergence_plateau_iteration()` по истории `history` алгоритма (список лучших значений скалярной пригодности по итерациям/поколениям):

- параметр `epsilon = 1e-6` — порог значимого улучшения;
- параметр `patience = 3` — число последовательных итераций без улучшения, после которых фиксируется плато.

Возвращается **индекс итерации**, на которой последний раз было зафиксировано улучшение перед плато. Меньшее значение означает более быструю сходимость. Итог — **среднее по прогонам** (`mean_convergence()`).

#### Нормировка весов

Каждый прогон использует `CriterionWeights` с инвариантом \(w_1+w_2+w_3=1\), \(w_i \ge 0\). Целевая функция:

```
F = w1 * P + w2 * O + w3 * R - Penalties
```

`Penalties` в текущей реализации равны 0: мультипликативные коррекции позиции и шага мастера уже входят в \(P, O, R\).

#### Constraint Pass Rate (снято)

Метрика удержания Certain-ограничений `TargetProfile` больше не используется: целевые пороги P/O/R не являются входом оптимизатора. В отчёте бенчмарка колонка Constraint Pass Rate удалена.

---

## 2. Структура Тестового Стенда

### 2.1. Архитектура модулей

| Модуль | Назначение |
| --- | --- |
| `optiflow/models/scoring.py` | Модель данных: `FieldSpec`, `EfficiencyTriple`, `InterfaceLayout`, `FunctionRegistry`, построение layout |
| `optiflow/optimization/algorithms.py` | `DecisionSpace`, все алгоритмы оптимизации, скалярная пригодность, Brute Force, Classic GA |
| `optiflow/benchmarks.py` | Monte Carlo-движок, генерация случайных снимков, агрегация метрик, формирование отчёта |
| `tests/test_benchmarks.py` | Unit-тесты: Brute Force, Classic GA, хелперы метрик, интеграционный бенчмарк |
| `optiflow/app.py` | Headless CLI: экспорт `wizard_output.html` + запуск сокращённого бенчмарка (20 прогонов) |

### 2.2. Динамическая рандомизация данных (стресс-тестирование)

Функция `random_benchmark_snapshot()` формирует уникальный снимок задачи оптимизации:

**Генерация полей (`FieldSpec`):**

- число полей `n_fields` — случайное целое от 2 до 4 (`min_fields=2`, `max_fields=4`);
- для каждого поля случайно выбирается `DataType` из `{BOOLEAN, UNSIGNED, TEXT}`;
- размер поля (`size`):
  - для `TEXT`: от 1 до 24;
  - для `BOOLEAN` и `UNSIGNED`: от 1 до 6.

**Привязка допустимых контролов** (`ensure_allowed_controls()`):

| Тип данных | Допустимые контролы |
| --- | --- |
| `BOOLEAN` | `CHECKBOX` |
| `UNSIGNED` | `SPINNER`, `SLIDER` |
| `TEXT` | `TEXTBOX`, `DROPDOWNLIST` |

**Параметры мастера:**

- `max_forms` — случайное целое от 1 до `min(3, n_fields)`, что гарантирует возможность неоднородного разбиения при достаточном числе полей.

**Веса критериев (`CriterionWeights`):**

- три случайных неотрицательных числа нормализуются через `CriterionWeights.from_raw()` так, что \(w_1+w_2+w_3=1\).

Таким образом, каждый прогон Monte Carlo тестирует движок на **различных комбинациях типов, размеров, глубины мастера и точек симплекса весов**, имитируя реальную вариативность входных данных.

### 2.3. Хромосома DecisionSpace: разбиение D + N

Класс `DecisionSpace` реализует геномную структуру Kurta-Izrailov:

**Часть 1 — длина D (токены управления):**

- индекс `x[i]` для поля `i` кодирует выбор контрола из `allowed_controls()`;
- при округлении (`round_to_valid()`) значение ограничивается диапазоном `[0, k − 1]`, где `k = len(allowed_controls())`.

**Часть 2 — длина N (веса разбиения мастера):**

- `N = partition_dim() = max(1, max_forms)` — гарантирует минимум одну форму даже при `max_forms = 0` в конфигурации;
- веса `x[D..D+N−1]` декодируются методом `_partition_counts_from_genome()`:
  - нормализация весов к сумме `D`;
  - распределение целочисленных count через floor + раздачу остатка по наибольшим дробным частям;
  - при нулевой сумме весов — fallback на равномерные единичные веса.

**Обработка граничных условий:**

| Условие | Механизм обработки |
| --- | --- |
| Пустой список полей | `control_dim() = 0`, пространство вырождается; тесты используют минимум 1–2 поля |
| Одна форма мастера (`N = 1`) | `_partition_counts_from_genome()` возвращает `[D]` — все поля на одной форме |
| Нулевая кардинальность контролов | `brute_force()` использует `range(max(1, c))` — предотвращает пустой `itertools.product` |
| Превышение 50 000 комбинаций | `brute_force()` → `ValueError`; бенчмарк продолжает прогон без baseline |
| Нормировка весов | `CriterionWeights.from_raw()` гарантирует \(w_1+w_2+w_3=1\) |

### 2.4. Состав алгоритмов в бенчмарке

Константа `BENCHMARK_ALGORITHMS` определяет восемь алгоритмов, запускаемых на **одном и том же** `(space, evaluator)` в каждом прогоне:

| Ключ | Алгоритм | Параметры по умолчанию в бенчмарке |
| --- | --- | --- |
| `BruteForce` | Полный перебор | лимит 50 000 комбинаций |
| `NSGA-II` | Многокритериальный GA | `pop_size=24`, `generations=20` |
| `GA` | Classic GA (однокритериальный) | `pop_size=24`, `generations=20`, SBX + Gaussian mutation |
| `PSO` | Рой частиц | `swarm_size=20`, `iterations=30` |
| `Greedy` | Жадный локальный поиск | без параметров |
| `SA` | Имитация отжига | `iterations=120` |
| `Tabu` | Поиск с запретами | `iterations=80` |
| `ACO` | Муравьиный алгоритм | `ants=12`, `iterations=25` |

Для стохастических алгоритмов в каждом прогоне генерируется индивидуальный `random_seed`, что обеспечивает воспроизводимость при фиксированном `random_seed` бенчмарка.

### 2.5. Покрытие unit-тестами

Файл `tests/test_benchmarks.py` содержит шесть тестовых классов:

| Класс | Тесты | Что проверяется |
| --- | --- | --- |
| `CriterionWeightsTests` | 6 | нормировка весов, инвариант суммы 1, фитнес \(F=w\cdot E-\mathrm{Pen}\) |
| `BruteForceTests` | 3 | размер пространства, глобальный максимум, `ValueError` при превышении лимита |
| `ClassicGATests` | 1 | корректный возврат layout, history, согласие `best_score` с `calculate_fitness` |
| `BenchmarkHelperTests` | 3 | детекция плато, формула precision, случайный симплекс весов |
| `MonteCarloBenchmarkTests` | 2 | интеграционный бенчмарк, GA ≤ BF на малом пространстве |
| `MarkdownFormatTests` | 1 | корректность форматирования таблицы отчёта |

---

## 3. Инструкция по Запуску Тестов и Бенчмарков

### 3.1. Предварительные требования

1. Python 3.10 или выше.
2. Активированное виртуальное окружение проекта с установленными зависимостями из `requirements.txt`:

```bash
cd /Users/pavelkurta/My/workspace/OptiFlow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Корневая директория репозитория должна быть текущей рабочей директорией, чтобы Python корректно разрешал пакет `optiflow`.

### 3.2. Запуск автоматизированных unit-тестов

**Основная команда** (из корня репозитория):

```bash
python3 -m unittest tests.test_benchmarks -v
```

**Ожидаемый результат:** 10 тестов, статус `OK`, время выполнения порядка 5–15 секунд (зависит от железа; интеграционный тест `test_run_optimization_benchmark_small` выполняет 8 Monte Carlo-прогонов).

**Запуск отдельного тестового класса:**

```bash
python3 -m unittest tests.test_benchmarks.BruteForceTests -v
python3 -m unittest tests.test_benchmarks.ClassicGATests -v
python3 -m unittest tests.test_benchmarks.MonteCarloBenchmarkTests -v
```

**Запуск одного конкретного теста:**

```bash
python3 -m unittest tests.test_benchmarks.BruteForceTests.test_brute_force_finds_global_maximum -v
```

**Альтернативный запуск через файл:**

```bash
python3 tests/test_benchmarks.py
```

### 3.3. Запуск полного Monte Carlo-бенчмарка (100 прогонов)

#### Способ A: через Python REPL или скрипт

```bash
cd /Users/pavelkurta/My/workspace/OptiFlow
source .venv/bin/activate
python3 -c "
import logging
from pathlib import Path
from optiflow.benchmarks import run_optimization_benchmark

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
stats = run_optimization_benchmark(
    runs_count=100,
    random_seed=42,
    output_dir=Path('.'),
    log_markdown=True,
)
print('Benchmark complete. Algorithms:', list(stats.keys()))
"
```

По завершении в текущей директории появится файл **`benchmark_report.md`**, а полная Markdown-таблица будет выведена в stdout через `logger.info`.

#### Способ B: через headless CLI приложения

При отсутствии PyQt5 запускается headless-режим, который после экспорта HTML выполняет **сокращённый** бенчмарк (20 прогонов):

```bash
cd /Users/pavelkurta/My/workspace/OptiFlow
source .venv/bin/activate
python3 -m optiflow.app
```

или явно:

```bash
python3 -c "from optiflow.app import run_headless_cli; run_headless_cli('wizard_output.html')"
```

Отчёт `benchmark_report.md` сохраняется в **родительскую директорию** файла `wizard_output.html` (по умолчанию — корень репозитория).

#### Способ C: настраиваемый скрипт с полными 100 прогонами

Создайте файл `run_benchmark.py` в корне репозитория:

```python
import logging
from pathlib import Path
from optiflow.benchmarks import run_optimization_benchmark

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_optimization_benchmark(
        runs_count=100,
        random_seed=42,
        output_dir=Path("."),
        log_markdown=True,
    )
```

Запуск:

```bash
python3 run_benchmark.py
```

### 3.4. Параметры конфигурации бенчмарка

| Параметр | Значение по умолчанию | Описание |
| --- | --- | --- |
| `runs_count` | `100` | Число Monte Carlo-прогонов |
| `random_seed` | `42` | Seed генератора случайных снимков для воспроизводимости |
| `output_dir` | `None` | Директория для записи `benchmark_report.md`; при `None` файл не создаётся |
| `log_markdown` | `True` | Вывод итоговой таблицы в лог |

Прогресс выполнения логируется каждые 10% прогонов: `Benchmark progress: X / 100 runs`.

---

## 4. Интерпретация Результатов и Отчётность

### 4.1. Структура автоматически генерируемого отчёта

Файл **`benchmark_report.md`** создаётся функцией `format_benchmark_markdown()` и имеет следующую структуру:

```markdown
# OptiFlow Optimization Benchmark

Monte Carlo runs: **100** | Brute-force baseline computable: **87**

| Algorithm | Precision Rate | Convergence (iter) | Baseline Samples |
| --- | ---: | ---: | ---: |
| BruteForce | 1.0000 | 5.2 | 87 |
| NSGA-II | 0.9612 | 14.8 | 87 |
| GA | 0.9534 | 11.3 | 87 |
| PSO | 0.9287 | 18.6 | 87 |
| Greedy | 0.8712 | 1.0 | 87 |
| SA | 0.9023 | 22.4 | 87 |
| Tabu | 0.8891 | 12.7 | 87 |
| ACO | 0.8456 | 16.2 | 87 |
```

> **Примечание:** числовые значения в примере выше иллюстративны и служат образцом формата. Реальные значения зависят от `random_seed` и аппаратного времени выполнения.

### 4.2. Образец таблицы для интерпретации (русскоязычная версия)

Для внутренней отчётности рекомендуется транслировать метрики в следующую форму:

| Алгоритм | Средняя точность (Precision) | Скорость сходимости (итерации до плато) | Выборок baseline |
| --- | ---: | ---: | ---: |
| BruteForce | 1.0000 | 5.2 | 87 |
| NSGA-II | 0.9612 | 14.8 | 87 |
| **GA (Classic)** | **0.9534** | **11.3** | **87** |
| PSO | 0.9287 | 18.6 | 87 |
| Greedy | 0.8712 | 1.0 | 87 |
| SA | 0.9023 | 22.4 | 87 |
| Tabu | 0.8891 | 12.7 | 87 |
| ACO | 0.8456 | 16.2 | 87 |

**Строка заголовка отчёта** содержит два контрольных числа:

- **Monte Carlo runs** — общее число прогонов (должно совпадать с переданным `runs_count`);
- **Brute-force baseline computable** — число прогонов, где `|Ω| ≤ 50 000` и Brute Force успешно завершился. Именно по этим прогонам вычисляется Precision Rate для эвристик (столбец **Baseline Samples**).

### 4.3. Пороги качества и критерии приёмки

#### Precision Rate (Средняя точность)

| Диапазон | Интерпретация | Рекомендуемое действие |
| --- | --- | --- |
| ≥ 0.95 | Высокое качество: эвристика стабильно находит решения, близкие к глобальному максимуму | Принять в production |
| 0.85 – 0.94 | Приемлемое качество с потерей до 15% от оптимума | Провести анализ регрессии; рассмотреть увеличение `generations`/`iterations` |
| < 0.85 | Низкое качество | Блокировать релиз; исследовать пространство поиска и параметры алгоритма |

**Контрольный порог для Classic GA:** средняя точность **> 0.95** (95%) относительно Brute Force на вычислимых baseline-прогонах. Это означает, что в среднем GA достигает не менее 95% глобального максимума скалярной пригодности.

Unit-тест `test_benchmark_brute_force_beats_or_matches_metaheuristics_on_tiny_space` проверяет **жёсткое** условие на малом пространстве: `score(BF) ≥ score(GA) − 1e-9`.

#### Convergence Speed (Скорость сходимости)

- Меньшее среднее значение — алгоритм быстрее выходит на плато.
- Greedy часто показывает `1.0`, поскольку его `history` содержит одну точку (нет итерационного процесса).
- SA и PSO, как правило, имеют более высокие значения из-за длинных циклов (`iterations=120` и `iterations=30` соответственно).
- Classic GA с SBX и Gaussian mutation обычно сходится за 10–15 итераций на типичных снимках.

#### Веса критериев

- Вход оптимизатора — `CriterionWeights`, а не целевые значения P, O, R.
- Unit-тесты `CriterionWeightsTests` проверяют \(\sum w_i = 1\) и формулу \(F = w_1 P + w_2 O + w_3 R - \mathrm{Penalties}\).

### 4.4. Пошаговая процедура верификации релиза

1. **Запустить unit-тесты:**

   ```bash
   python3 -m unittest tests.test_benchmarks -v
   ```

   Убедиться: `Ran 16 tests` → `OK`, ноль ошибок и провалов.

2. **Запустить полный бенчмарк:**

   ```bash
   python3 -c "
   import logging
   from pathlib import Path
   from optiflow.benchmarks import run_optimization_benchmark
   logging.basicConfig(level=logging.INFO)
   run_optimization_benchmark(runs_count=100, random_seed=42, output_dir=Path('.'), log_markdown=True)
   "
   ```

3. **Открыть `benchmark_report.md`** и проверить:

   - `Brute-force baseline computable` ≥ 50 (достаточная статистическая выборка; при значении ниже 30 рекомендуется уменьшить `max_fields` в `random_benchmark_snapshot` или увеличить `runs_count`);
   - `BruteForce | Precision Rate` = `1.0000` (инвариант эталона);
   - `GA | Precision Rate` ≥ `0.9500`;
   - отсутствие аномалий: Precision Rate не должен превышать `1.0000` (контроль формулы `min(1.0, ...)`).

4. **Сравнить с предыдущим отчётом** (при наличии): отклонение Precision Rate Classic GA более чем на 3 процентных пункта сигнализирует о регрессии.

5. **Зафиксировать артефакты:** сохранить `benchmark_report.md` и лог stdout в CI-архив или приложение к отчёту о тестировании.

### 4.5. Диагностика типовых проблем

| Симптом | Возможная причина | Решение |
| --- | --- | --- |
| `ValueError: Brute force search space exceeds limit` в логах | Случайный снимок сгенерировал слишком большое пространство | Ожидаемое поведение; baseline для этого прогона пропускается |
| `Baseline Samples = 0` для всех алгоритмов | Все 100 прогонов превысили лимит 50 000 | Уменьшить `max_fields` до 3, `max_forms_cap` до 2 в `random_benchmark_snapshot` |
| `Precision Rate = n/a` | Ни один прогон не имел вычислимого baseline | См. выше |
| `ImportError: No module named optiflow` | Запуск не из корня репозитория | Выполнить `cd` в корень проекта или установить пакет editable: `pip install -e .` |
| GA Precision < 0.95 стабильно | Недостаточно поколений или высокая стохастичность профилей | Увеличить `generations` в `_run_algorithm()` или `runs_count` для усреднения |
| Unit-тест `test_brute_force_finds_global_maximum` падает | Регрессия в `brute_force()` или `calculate_fitness()` | Сравнить `best_score` с независимым перебором в теле теста |

### 4.6. Связь отчёта с экспортом интерфейса

Headless-режим (`run_headless_cli()`) генерирует два артефакта в одной директории:

| Файл | Содержание |
| --- | --- |
| `wizard_output.html` | HTML-разметка оптимального многошагового мастера |
| `benchmark_report.md` | Статистическая матрица качества всех алгоритмов |

Оба файла должны храниться совместно для трассируемости: HTML отражает **конкретный** выбранный layout, а Markdown — **статистическую валидацию** движка оптимизации на множестве случайных задач.

---

## Приложение A. Формулы и константы (справочник)

| Константа / параметр | Значение | Модуль |
| --- | --- | --- |
| `BRUTE_FORCE_MAX_COMBINATIONS` | `50 000` | `algorithms.py` |
| `epsilon` (convergence) | `1e-6` | `benchmarks.py` |
| `patience` (convergence) | `3` | `benchmarks.py` |
| `calculate_fitness` | \(F=w_1P+w_2O+w_3R-\mathrm{Pen}\) | `algorithms.py` |
| `runs_count` (default) | `100` | `benchmarks.py` |
| `random_seed` (default) | `42` | `benchmarks.py` |

## Приложение B. Быстрая шпаргалка команд

```bash
# Unit-тесты
python3 -m unittest tests.test_benchmarks -v

# Полный бенчмарк 100 прогонов
python3 -c "import logging; from pathlib import Path; from optiflow.benchmarks import run_optimization_benchmark; logging.basicConfig(level=logging.INFO); run_optimization_benchmark(runs_count=100, random_seed=42, output_dir=Path('.'), log_markdown=True)"

# Headless: HTML + бенчмарк 20 прогонов
python3 -m optiflow.app
```

---

*Документ подготовлен для проекта OptiFlow. Файл: `TESTING_GUIDE.md`.*
