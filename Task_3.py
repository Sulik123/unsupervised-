import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from sklearn.decomposition import PCA
import base64
from io import BytesIO
import sys

app = Flask(__name__)

# Загрузка моделей (обновлённые, без квадратов)
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

# Загрузка исходных данных для статистики и графика
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

# Добавляем признаки (те же 7, что при обучении)
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
rfm['LogMonetary'] = np.log1p(rfm['Monetary'])
rfm['LogFrequency'] = np.log1p(rfm['Frequency'])
rfm['RecencyScore'] = 1 / (rfm['Recency'] + 1)

features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
            'LogMonetary', 'LogFrequency', 'RecencyScore']
X = rfm[features]
X_scaled = scaler.transform(X)

# Предсказание кластеров для всех клиентов (для отображения на графике)
clusters_all = kmeans.predict(X_scaled)
rfm['Cluster'] = clusters_all

# PCA для всех точек (уже есть pca, но надо спроецировать)
X_pca = pca.transform(X_scaled)
rfm['PC1'] = X_pca[:, 0]
rfm['PC2'] = X_pca[:, 1]

# Описания кластеров (на основе полученных средних)
LABELS = {
    0: "Активные постоянные (Monetary~5200, Frequency~10, Recency~23 дня)",
    1: "Редкие / потерянные (Monetary~580, Frequency~2, Recency~120 дней)",
    2: "Крупные аномалии (Monetary~122k, Frequency~1.5, Recency~163 дня)",
}

# Функция предсказания для нового клиента (без квадратов)
def predict_cluster(r, f, m):
    if f == 0:
        f = 1
    avg_order = m / f
    log_m = np.log1p(m)
    log_f = np.log1p(f)
    rec_score = 1 / (r + 1)
    X_new = np.array([[r, f, m, avg_order, log_m, log_f, rec_score]])
    X_scaled_new = scaler.transform(X_new)
    c = kmeans.predict(X_scaled_new)[0]
    return c

# Генерация графика в base64 для вставки в HTML
def get_plot_base64():
    plt.figure(figsize=(8, 6))
    for c in sorted(rfm['Cluster'].unique()):
        subset = rfm[rfm['Cluster'] == c]
        plt.scatter(subset['PC1'], subset['PC2'], label=f'Кластер {c}: {LABELS.get(c, "")[:30]}', alpha=0.6)
    plt.xlabel('Первая главная компонента (PC1)')
    plt.ylabel('Вторая главная компонента (PC2)')
    plt.title('Визуализация кластеров покупателей (PCA)')
    plt.legend(loc='best', fontsize=8)
    # Сохраняем в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plot_data

# Главная страница
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            r = float(request.form['recency'])
            f = float(request.form['frequency'])
            m = float(request.form['monetary'])
            if r < 0 or f <= 0 or m < 0:
                result = "Ошибка: введите корректные положительные числа (частота > 0)."
            else:
                cluster = predict_cluster(r, f, m)
                result = f"**Результат:** Кластер {cluster} — {LABELS.get(cluster, 'Неизвестный кластер')}"
        except ValueError:
            result = "Ошибка: введите числа (например, 10, 5, 1500)."
        except Exception as e:
            result = f"Ошибка: {e}"
    # Получаем график в base64
    plot_base64 = get_plot_base64()
    # Статистика
    n_clients = len(rfm)
    n_clusters = len(rfm['Cluster'].unique())
    return render_template('index.html',
                           plot_base64=plot_base64,
                           n_clients=n_clients,
                           n_clusters=n_clusters,
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)