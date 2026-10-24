# Ядра затухания вместо ручных окон в рекомендательной системе

Исследование на Yambda-50M: можно ли выбирать масштаб памяти признаков автоматически и заранее (без обучения бустинга) оценивать, какой набор признаков лучше. Признаки считаются как суммы по ядру (окно или экспонента) от потока событий; на них обучается XGBoost; отдельная «теория» (линейный R² по гистограмме лагов) предсказывает качество набора до обучения.

Подробности, таблицы и ограничения — в `t_RecSys_.pdf`.

## Главные результаты

- Экспоненциальные конверсии лучше оконных при тех же полупериодах (7 и 30 дней): +0,0080 по AUC внутри пользователя над классикой, во всех 5 сидах. Эффект сохраняется после подбора окон. Масштаб в конверсиях выбран вручную, это эффект формы ядра, а не автоподбора.
- Заранее заданное утверждение (6 ядер, выбранных теорией, лучше ручных окон) **не подтвердилось**: при общем хвосте признаков (`last`, `age`, конверсии) выбор ядер почти ничего не решает.
- Без хвоста отбор по теории значимо лучше случайных ядер и простого top-k, предсказанный R² умеренно ранжирует наборы. Но полностью автоматический набор ядер без хвоста уступает ручному.
- Данные не различают окно в 7 и 10 дней.

Все выводы получены на одном датасете, пяти сидах и трёх тестовых днях
## Структура

| Файл | Назначение |
|---|---|
| `download_data.py` | скачивание и проверка Yambda с Hugging Face |
| `data.py` | загрузка, снапшоты и кандидаты, метрики (AUC внутри пользователя, NDCG, recall) |
| `features.py` | признаки как спецификации (семейство, канал, ядро, масштаб) и их расчёт на GPU |
| `theory.py` | достаточные статистики, R² наборов без обучения, жадный отбор, проверка двух шкал |
| `models.py` | XGBoost на CUDA |
| `run.py` | эксперименты по сидам и этапам |
| `analyze.py` | парные разности с доверительными интервалами, графики |
| `wiener_gap.py`, `scales_report.py`, `selected_report.py` | сводки: зазор до ядра Винера, две шкалы, какие ядра выбирает теория |
| `learn_kernel.py` | обучаемые ядра |

## Установка

```
pip install numpy pandas scikit-learn scipy pyarrow matplotlib "xgboost>=2.0" huggingface_hub
# torch — сборка под вашу CUDA: https://pytorch.org/get-started/locally/
```

Нужна видеокарта с CUDA. Для проверки на CPU: `--device cpu` и `--dataset synthetic`.

## Запуск

Проверка на синтетике (минуты):

```
python run.py --dataset synthetic --device cpu --users 250 --seeds 0,1 --n-test 3 \
    --theory-boot 4 --theory-bins 16 --n-rand 2 --stages heur,theory,sets
python analyze.py --dir results_multi
```

Основной эксперимент (5 сидов * 3 тестовых дня):

```
python download_data.py
python run.py --dataset yambda --size 50m --users 3000 --device cuda \
    --seeds 0,1,2,3,4 --n-test 3 --stages heur,theory,sets,tuned --out results_multi
python run.py ... --stages extra  --append --out results_multi
python run.py ... --stages extra2 --append --out results_multi
python run.py ... --stages scales,extra3 --append --out results_multi

python analyze.py         --dir results_multi
python wiener_gap.py      --dir results_multi
python scales_report.py   --dir results_multi
python selected_report.py --dir results_multi
```

Этапы `run.py`: `heur` (эвристики), `theory`, `sets` (все наборы признаков), `tuned` (подбор окон), `extra` и `extra2` (контроли), `scales` (проверка двух шкал), `extra3` (объединение окон и экспонент, гистограммы лагов), `learned` (обучаемые ядра). `--append` дописывает к уже посчитанным результатам.

## Выходные файлы

В папке `--out`: `seed*/results.json` (метрики и предсказания теории), `seed*/peruser.pkl` (метрики по пользователям), `analysis.md` и `analysis.json` (парные разности), `forest.png`, `pred_vs_actual_*.png`, `wiener_gap.md`, `scales_report.md`, `selected_report.md`.

## Ограничения

- Метрики считаются среди кандидатов (история и популярное), а не на полном каталоге.
- Теория линейна, учитывает только ядра и не видит `last`, `age`, конверсий и взаимодействий.
