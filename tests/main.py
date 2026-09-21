from math import comb
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Пути вычисляются от расположения этого файла, а не от текущей рабочей директории.
project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "data" / "CaliforniaHousing.csv"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if not file_path.is_file():
    raise FileNotFoundError(f"Датасет не найден: {file_path}")

# Загрузка датасета
df = pd.read_csv(file_path)

# Быстрый просмотр
df

print("Shape:", df.shape)
print("\nInfo:")
df.info()

print("\nОписание числовых признаков:")
df.describe()

missing_values = df.isnull().sum()

print("Пропуски по столбцам:")
print(missing_values)

print("\nПроцент пропусков:")
print((missing_values / len(df)) * 100)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Категориальные признаки:")
print(categorical_cols)

# Посмотрим уникальные значения для них
for col in categorical_cols:
    print(f"\nКолонка: {col}")
    print(df[col].value_counts())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

outliers_dict = {}

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    outliers_dict[col] = len(outliers)

print("Количество выбросов по признакам (метод 3 сигм):")
for col, count in outliers_dict.items():
    print(f"{col}: {count}")

outliers_data = {}

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    if not outliers.empty:
        outliers_data[col] = outliers

# Пример: посмотреть выбросы по конкретной колонке
list(outliers_data.keys())

"""# **Подготовка данных**

## **Обработка пропусков**
"""

# Проверим количество пропусков до
print("Пропуски ДО:", df['total_bedrooms'].isnull().sum())

# Вычисляем среднее (по умолчанию NaN игнорируются)
mean_value = df['total_bedrooms'].mean()

print("Среднее значение total_bedrooms:", mean_value)

# Заполняем пропуски
df['total_bedrooms'] = df['total_bedrooms'].fillna(mean_value)

# Проверим после
print("Пропуски ПОСЛЕ:", df['total_bedrooms'].isnull().sum())

df

"""## **Обработка категориальных переменных**"""

from sklearn.preprocessing import LabelEncoder

# Приводим к строкам и чистим
df['ocean_proximity'] = (
    df['ocean_proximity']
    .astype(str)
    .str.strip()
)

# Проверим уникальные значения после очистки
print("Уникальные значения:", df['ocean_proximity'].unique())

# Инициализируем encoder
le = LabelEncoder()

# Обучаем и трансформируем
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# Посмотрим соответствие "категория → число"
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Mapping:", mapping)

# Проверка
print(df['ocean_proximity'].value_counts())

"""## **Обработка выбросов**"""

df_capped = df.copy()

numeric_cols = df_capped.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    mean = df_capped[col].mean()
    std = df_capped[col].std()

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # Применяем ограничение
    df_capped[col] = df_capped[col].clip(lower_bound, upper_bound)

print("Capping завершён")
df_capped

"""## **Создание новых признаков**"""

import numpy as np

# Координаты Лос-Анджелеса
LA_LAT = 34.0522
LA_LON = -118.2437

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # радиус Земли в метрах

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# Вычисляем расстояние
df_capped['distance_to_LA'] = haversine(
    df['latitude'],
    df['longitude'],
    LA_LAT,
    LA_LON
)

df_capped

df_final = df_capped.copy()
df_final

# Удаляем географические координаты
df_final = df_final.drop(columns=['latitude', 'longitude'])

# Проверка
print(df_final.columns)
df_final.head()

"""## **Нормализация и разделение на обучающие и тестовые выборки**"""

# Целевая переменная
y = df_final['median_house_value']

# Признаки
X = df_final.drop(columns=['median_house_value'])

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

print(X_train.shape, X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# обучаем только на train
X_train_scaled = scaler.fit_transform(X_train)

# применяем к test
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape, X_test_scaled.shape)

from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()

# масштабируем
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# возвращаем DataFrame с теми же колонками и индексами
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)

X_test_scaled

X_train_scaled




from eikg import (
    CombinatorialPolynomialNetwork,
    DeepPolyNetwork,
    DeepPolyNetworkCV,
    EIKGPolynomialRegressor,
    EIKGPolynomialRegressorCV,
)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# --- Ожидается, что у тебя уже есть:
# X_train, X_test, y_train, y_test
# Если y в DataFrame с одним столбцом, можно так:
# y_train = y_train.iloc[:, 0]
# y_test = y_test.iloc[:, 0]

n_features = X_train.shape[1]

# Степень одиночного полинома выбирается только по CV MSE на обучающей выборке.
polynomial_selector = EIKGPolynomialRegressorCV(
    max_degree=n_features,
    cv=5,
    scoring="neg_mean_squared_error",
    regularization="ridge",
    alpha_ridge=1e-6,
    fit_intercept=True,
    scale=True,
    normalize_latent=True,
    scale_y=False,
    check_input=True,
).fit(X_train_scaled, y_train)

polynomial_cv_results = pd.DataFrame({
    "degree": np.arange(1, n_features + 1),
    # Класс хранит отрицательный MSE, поскольку при выборе максимизируется score.
    "CV_MSE": -np.asarray(polynomial_selector.cv_scores_),
}).sort_values("CV_MSE", ascending=True)

print("\nCV MSE одиночных полиномов:")
print(polynomial_cv_results.reset_index(drop=True))

results = []
predictions_by_degree = {}

for degree in range(1, n_features + 1):
    model = EIKGPolynomialRegressor(
        degree=degree,
        regularization="ridge",   # можно "none"
        alpha_ridge=1e-6,
        fit_intercept=True,
        scale=True,
        normalize_latent=True,
        scale_y=False,
        check_input=True,
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "degree": degree,
        "MAE": mae,
        "MAPE": mape,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    })

    predictions_by_degree[degree] = y_pred

results_df = pd.DataFrame(results).sort_values("degree").reset_index(drop=True)
# display(results_df)
print(results_df)
best_degree = polynomial_selector.selected_degree_
best_idx = int(results_df.index[results_df["degree"] == best_degree][0])
print(f"Лучшая степень по минимальному CV MSE: {best_degree}")
print(f"Минимальный CV MSE: {-polynomial_selector.best_score_}")


# Сначала CV-сеть выбирает отдельную степень каждого слоя по минимальному
# fold-local CV MSE. Затем тот же кортеж явно используется в DeepPolyNetwork.
network_cv_model = DeepPolyNetworkCV(
    n_layers=3,
    max_degree=3,
    cv=3,
    scoring="neg_mean_squared_error",
    shuffle=True,
    random_state=42,
    regularization="ridge",
    alpha_ridge=1e-6,
    fit_intercept=True,
    scale=True,
    scale_y=True,
    normalize_latent=True,
)

print("\nПодбор степеней DeepPolyNetworkCV по минимальному CV MSE...")
network_cv_model.fit(X_train_scaled, y_train)

network_model = DeepPolyNetwork(
    n_layers=network_cv_model.n_layers_,
    degree=network_cv_model.selected_degrees_,
    regularization="ridge",
    alpha_ridge=1e-6,
    fit_intercept=True,
    scale=True,
    scale_y=True,
    normalize_latent=True,
)

print("\nОбучение DeepPolyNetwork с выбранными CV степенями...")
network_model.fit(X_train_scaled, y_train)

# Комбинаторная сеть перебирает все сочетания размером от 2 до количества
# исходных признаков включительно. Для 8 признаков это 247 кандидатов:
# C(8, 2) + C(8, 3) + ... + C(8, 8).
min_combination_size = 2
max_combination_size = n_features
n_combinatorial_candidates = sum(
    comb(n_features, size)
    for size in range(min_combination_size, max_combination_size + 1)
)

print(
    "\nПространство поиска CombinatorialPolynomialNetwork: "
    f"{n_combinatorial_candidates} комбинаций признаков "
    f"(размеры {min_combination_size}..{max_combination_size})."
)

combinatorial_model = CombinatorialPolynomialNetwork(
    top_k=25,
    min_combination_size=min_combination_size,
    max_combination_size=max_combination_size,
    max_candidates=1000,
    max_degree=n_features,
    cv=3,
    regularization="ridge",
    alpha_ridge=1e-6,
    fit_intercept=True,
    scale=True,
    scale_y=True,
    normalize_latent=True,
)

print("\nОбучение CombinatorialPolynomialNetwork по минимальному CV MSE...")
combinatorial_model.fit(X_train_scaled, y_train)

network_models = {
    "DeepPolyNetwork": network_model,
    "DeepPolyNetworkCV": network_cv_model,
    "CombinatorialPolynomialNetwork": combinatorial_model,
}

network_results = []
network_predictions = {}

for model_name, network_model in network_models.items():
    network_prediction = network_model.predict(X_test_scaled)

    network_mse = mean_squared_error(y_test, network_prediction)

    if isinstance(network_model, CombinatorialPolynomialNetwork):
        degree_description = (
            f"first={tuple(network_model.selected_degrees_)}; "
            f"final={network_model.final_degree_}"
        )
    elif isinstance(network_model, DeepPolyNetwork):
        degree_description = network_model.degrees_
    else:
        degree_description = network_model.selected_degrees_

    network_results.append({
        "model": model_name,
        "degrees": degree_description,
        "MAE": mean_absolute_error(y_test, network_prediction),
        "MAPE": mean_absolute_percentage_error(y_test, network_prediction),
        "MSE": network_mse,
        "RMSE": np.sqrt(network_mse),
        "R2": r2_score(y_test, network_prediction),
    })
    network_predictions[model_name] = network_prediction

network_results_df = pd.DataFrame(network_results)

print("\nРезультаты полиномиальных сетей:")
print(network_results_df)

print("\nСтепени, выбранные DeepPolyNetworkCV:")
print(network_models["DeepPolyNetworkCV"].selected_degrees_)

network_layer_cv_results = pd.DataFrame({
    "layer": np.arange(1, network_cv_model.n_layers_ + 1),
    "selected_degree": network_cv_model.selected_degrees_,
    "CV_MSE": -np.asarray(network_cv_model.layer_best_scores_),
})

print("\nМинимальный CV MSE на каждом слое DeepPolyNetworkCV:")
print(network_layer_cv_results)

combinatorial_ranking = pd.DataFrame(combinatorial_model.ranking_)
ranking_columns = [
    "rank",
    "combination",
    "feature_names",
    "selected_degree",
    "cv_mse_mean",
    "cv_mse_std",
    "selected",
]

print("\nTop-K комбинаций CombinatorialPolynomialNetwork по минимальному CV MSE:")
print(
    combinatorial_ranking.loc[combinatorial_ranking["selected"], ranking_columns]
    .reset_index(drop=True)
)
print(
    "Итоговая степень CombinatorialPolynomialNetwork: "
    f"{combinatorial_model.final_degree_}"
)

best_polynomial_result = results_df.loc[[best_idx]].copy()
best_polynomial_result.insert(0, "model", "EIKGPolynomialRegressor")
best_polynomial_result["degrees"] = best_polynomial_result["degree"].map(
    lambda degree: (int(degree),)
)

comparison_columns = ["model", "degrees", "MAE", "MAPE", "MSE", "RMSE", "R2"]
comparison_df = pd.concat(
    [best_polynomial_result[comparison_columns], network_results_df[comparison_columns]],
    ignore_index=True,
).sort_values("MSE", ascending=True)

print("\nИтоговое сравнение моделей (лучшие по test MSE сверху):")
print(comparison_df.reset_index(drop=True))
