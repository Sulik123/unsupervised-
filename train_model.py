import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

# Загрузка
df = pd.read_csv('Assignment-1_Data.csv', sep=';', decimal=',', encoding='latin1', on_bad_lines='skip')
df.columns = df.columns.str.replace('ï»¿', '').str.strip()
df = df.dropna(subset=['CustomerID'])
df = df[~df['BillNo'].astype(str).str.contains('C', na=False)]
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
df['Revenue'] = df['Quantity'] * df['Price']
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', errors='coerce')
df = df.dropna(subset=['Date'])

# RFM
last_date = df['Date'].max()
rfm = df.groupby('CustomerID').agg({
    'Date': lambda x: (last_date - x.max()).days,
    'BillNo': 'nunique',
    'Revenue': 'sum'
}).rename(columns={'Date': 'Recency', 'BillNo': 'Frequency', 'Revenue': 'Monetary'})
rfm = rfm[rfm['Monetary'] > 0]

# Только нужные признаки (БЕЗ КВАДРАТОВ)
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
rfm['LogMonetary'] = np.log1p(rfm['Monetary'])
rfm['LogFrequency'] = np.log1p(rfm['Frequency'])
rfm['RecencyScore'] = 1 / (rfm['Recency'] + 1)

features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
            'LogMonetary', 'LogFrequency', 'RecencyScore']
X = rfm[features]

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Определение оптимального k (2-8)
inertia = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.0f}, silhouette={sil_scores[-1]:.3f}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(K_range, inertia, 'bo-')
plt.title('Метод локтя')
plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, 'ro-')
plt.title('Силуэт')
plt.tight_layout()
plt.show()

best_k = 3
print(f"\nОптимальное k по силуэту: {best_k}")

# Финальная модель
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
rfm['Cluster'] = clusters

# PCA и график (исправленная индексация)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
for c in range(best_k):
    mask = rfm['Cluster'] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Кластер {c}', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Кластеры покупателей (PCA)')
plt.legend()
plt.savefig('clusters_plot.png')
plt.show()

# Анализ кластеров
cluster_summary = rfm.groupby('Cluster')[['Monetary','Frequency','Recency']].mean()
print("\nСредние по кластерам:")
print(cluster_summary.round(1))

# Сохранение моделей
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca_model.pkl')
print("\nМодели сохранены (без квадратов).")