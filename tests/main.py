

import pandas as pd
import numpy as np

# Путь к файлу
file_path = r'C:\Users\RobotComp.ru\Desktop\projects\EIKGPRegressor\data\CaliforniaHousing.csv'

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




import numpy as np
import pandas as pd

# from eikg import EIKGPolynomialRegressor
from eikg.regressors import EIKGPolynomialRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# --- Ожидается, что у тебя уже есть:
# X_train, X_test, y_train, y_test
# Если y в DataFrame с одним столбцом, можно так:
# y_train = y_train.iloc[:, 0]
# y_test = y_test.iloc[:, 0]

n_features = X_train.shape[1]

results = []
predictions_by_degree = {}

for degree in range(2, n_features + 1):
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
best_idx = results_df["R2"].idxmax()
best_degree = int(results_df.loc[best_idx, "degree"])
print(f"Лучшая степень по R2: {best_degree}")