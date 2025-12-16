# Зачетное состязание на Kaggle WiDS Datathon 2020 

(https://www.kaggle.com/competitions/widsdatathon2020/)

**Задача**: сравнить/победить LightAutoML кастомным решением .

## Спойлер:
**LAMA**
![Image One](https://skrinshoter.ru/s/161225/QBVlhpc2.jpg?download=1&name=%D0%A1%D0%BA%D1%80%D0%B8%D0%BD%D1%88%D0%BE%D1%82-16-12-2025%2013:10:57.jpg)

**LGBM + Optuna**
![Image Two](https://skrinshoter.ru/s/161225/I9anjGTH.jpg?download=1&name=%D0%A1%D0%BA%D1%80%D0%B8%D0%BD%D1%88%D0%BE%D1%82-16-12-2025%2013:14:18.jpg)

## 01_eda.ipynb — анализ данных

- **Данные**: train `91713 × 186`, test `39308 × 186` (WiDS Datathon 2020 https://www.kaggle.com/competitions/widsdatathon2020/) .
- **Таргет**: бинарный `hospital_death`, доля положительного класса ≈ **8.6%** (класс сильно несбалансирован).
- **ID-колонки**: `encounter_id`, `patient_id` исключаются из признаков и используются только для группировок/валидации.
- **Стабильность таргета во времени**: по скользящему среднему вдоль `encounter_id` утечки и трендов не обнаружено (таргет стабилен).
- **Признаки**: выделены числовые и категориальные столбцы, построены распределения и heatmap корреляций; многие клинические признаки имеют заметные пропуски, часть из них планируется либо дропать, либо аккуратно импатировать.
- **Идеи для feature engineering**: биннинги возраста/ИМТ, взаимодействия вероятностей смерти Apache, соотношение LOS до и в ICU, агрегаты по `hospital_id` и другим категориальным признакам.

---

## 02_lama_baseline.ipynb — LightAutoML бейслайн

- **Подготовка данных**: 
  - удалены признаки с >40% пропусков;
  - добавлены фичи: биннинги `age`/`bmi`, взаимодействие `apache_4a_hospital_death_prob × apache_4a_icu_death_prob`, отношение `pre_icu_los_days / icu_los_days` и target-like статистики по `hospital_id`, `icu_type`, `apache_3j_bodysystem`.
  - train `91713 × 119`, test `39308 × 118`; таргет по-прежнему ~**8.6%** единиц.
- **Валидация**: `StratifiedKFold`, 5 фолдов, фиксированный `random_state=42`.
- **Config A (GPU tiny)**: быстрый GPU-preset с LGBM/CB/XGB; OOF ROC-AUC ≈ **0.9038**, сформирован сабмит `submission_gpu_tiny.csv`.
- **Config B (GPU extended)**: расширенный preset (LGBM+CatBoost+XGBoost+LinearL2, feature selection, тюнинг); OOF ROC-AUC **0.90437**, сабмит `submission_gpu_extended.csv`.
- **Вывод по ноутбуку**: LAMA extended даёт лёгкое улучшение по сравнению с быстрым пресетом и принимается как сильный табличный бейслайн.

---

## 07_lightgbm_optuna.ipynb — LGBM + Optuna (GPU)

- **Подготовка данных**: 
  - те же CSV, что и в EDA (train/test WiDS);
  - таргет `hospital_death` отделён от фичей, ID-колонки исключены из признаков;
  - категориальные признаки приведены к типу `category` для корректной работы LGBM.
- **GPU-детект**: на подвыборке успешно запускается `LGBMClassifier(device="gpu")`, используется видеокарта `NVIDIA GeForce RTX 3060`.
- **Optuna**: 
  - `StratifiedKFold` с `N_SPLITS=3`, оптимизация ROC-AUC;
  - 20 трейлов по основным гиперпараметрам (`learning_rate`, `num_leaves`, `max_depth`, `feature_fraction`, `bagging_fraction`, регуляризации, `n_estimators`).
  - Лучший результат на CV: **Best AUC ≈ 0.9060**.
- **Финальная модель**: 
  - обучение с лучшими гиперпараметрами и hold-out-валидацией (20% от train); 
  - **Hold-out ROC-AUC ≈ 0.9109**;
  - итоговая модель дообучается на всех данных и считает прогнозы для теста.
- **Сабмит**: сохранён в `models/submission_lgbm_optuna_gpu.csv`.

---

## Итоговое сравнение

На сабмите в Kaggle LGBM+Optuna (Private score: **0.90696**) показал небольшое превосходство по сравнению с LAMA extended (Private score: **0.90504**). LGBM+Optuna на GPU можно считать текущей лучшей моделью, а LAMA extended — сильным и более универсальным бейслайном.

