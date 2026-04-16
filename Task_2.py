import joblib
import pandas as pd
import numpy as np
import sys

# Загрузка моделей (новые, без квадратов)
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Описания кластеров – посмотри средние из обучения и подставь свои названия
LABELS = {
    0: "Постоянные",
    1: "Прохожие(мало заказов+давно)",
    2: "Крупные(давно)",
}

def predict(r, f, m):
    """Предсказание кластера по трём признакам (без квадратов)"""
    if f == 0:
        f = 1
    avg_order = m / f
    log_m = np.log1p(m)
    log_f = np.log1p(f)
    rec_score = 1 / (r + 1)
    # Вектор из 7 признаков (порядок строго как при обучении)
    X = np.array([[r, f, m, avg_order, log_m, log_f, rec_score]])
    X_scaled = scaler.transform(X)
    c = kmeans.predict(X_scaled)[0]
    return c

if "--all" in sys.argv:
    # Режим обработки всего датасета (оставь как есть, если нужно)
    df = pd.read_csv('Assignment-1_Data.csv', sep=';', decimal=',', encoding='latin1', on_bad_lines='skip')
    df.columns = df.columns.str.replace('ï»¿', '').str.strip()
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['BillNo'].astype(str).str.contains('C', na=False)]
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    df['Revenue'] = df['Quantity'] * df['Price']
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['Date'])
    last_date = df['Date'].max()
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (last_date - x.max()).days,
        'BillNo': 'nunique',
        'Revenue': 'sum'
    }).rename(columns={'Date': 'Recency', 'BillNo': 'Frequency', 'Revenue': 'Monetary'})
    rfm = rfm[rfm['Monetary'] > 0]
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    rfm['LogMonetary'] = np.log1p(rfm['Monetary'])
    rfm['LogFrequency'] = np.log1p(rfm['Frequency'])
    rfm['RecencyScore'] = 1 / (rfm['Recency'] + 1)
    features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
                'LogMonetary', 'LogFrequency', 'RecencyScore']
    X_all = scaler.transform(rfm[features])
    rfm['Cluster'] = kmeans.predict(X_all)
    print(rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(1))
    print("\nРазмер кластеров:\n", rfm['Cluster'].value_counts().sort_index())
else:
    # Интерактивный режим
    print("\n=== Классификатор покупателей (исправленная версия) ===")
    print("Введите 'exit' или 'q' для выхода.\n")
    while True:
        try:
            r_input = input("Recency (дней): ").strip()
            if r_input.lower() in ('exit','q','quit'):
                break
            f_input = input("Frequency (заказов): ").strip()
            if f_input.lower() in ('exit','q','quit'):
                break
            m_input = input("Monetary (сумма): ").strip()
            if m_input.lower() in ('exit','q','quit'):
                break
            r = float(r_input)
            f = float(f_input)
            m = float(m_input)
            if r < 0 or f < 0 or m < 0:
                print("Ошибка: отрицательные значения. Попробуйте снова.\n")
                continue
            if f == 0:
                print("Ошибка: частота не может быть 0.\n")
                continue
            cluster = predict(r, f, m)
            print(f"\nРезультат: Кластер {cluster} — {LABELS.get(cluster, 'Неизвестный')}\n")
        except ValueError:
            print("Ошибка: введите число. Попробуйте снова.\n")
        except Exception as e:
            print(f"Ошибка: {e}\n")