import joblib
import numpy as np

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

LABELS = {
    0: "Активные постоянные",
    1: "Редкие / потерянные",
    2: "Крупные и/или аномалии",
}

def predict(r, f, m):
    if f == 0:
        f = 1
    avg_order = m / f
    log_m = np.log1p(m)
    log_f = np.log1p(f)
    rec_score = 1 / (r + 1)
    X = np.array([[r, f, m, avg_order, log_m, log_f, rec_score]])
    X_scaled = scaler.transform(X)
    return kmeans.predict(X_scaled)[0]

print("\n=== Классификатор покупателей ===")
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

        if r < 0 or f <= 0 or m < 0:
            print("Ошибка: значения должны быть положительными (частота > 0).\n")
            continue

        cluster = predict(r, f, m)
        print(f"\nРезультат: Кластер {cluster} — {LABELS.get(cluster, 'Неизвестный')}\n")

    except ValueError:
        print("Ошибка: введите число.\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")
