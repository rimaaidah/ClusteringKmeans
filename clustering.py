import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def proses_clustering(filepath):
    try:
        # Baca file CSV
        df = pd.read_csv(filepath)

        # Kolom yang dibutuhkan
        if not {'Gender', 'Age', 'EstimatedSalary'}.issubset(df.columns):
            return None

        # Hapus duplikat & data kosong
        df = df.drop_duplicates().dropna()

        # Encode Gender (0: Female, 1: Male) atau sebaliknya
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])

        # Fitur yang digunakan (semua)
        fitur = ['Gender', 'Age', 'EstimatedSalary']
        X = df[fitur]

        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Inisialisasi untuk evaluasi
        inertia = []
        silhouette_scores = []
        k_range = range(2, 10)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))

        # Simpan Elbow Method
        os.makedirs('static', exist_ok=True)
        plt.figure()
        plt.plot(k_range, inertia, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Jumlah Cluster (k)')
        plt.ylabel('Inertia')
        plt.savefig('static/elbow.png')
        plt.close()

        # Simpan Silhouette Score
        plt.figure()
        plt.plot(k_range, silhouette_scores, marker='o', color='green')
        plt.title('Silhouette Score')
        plt.xlabel('Jumlah Cluster (k)')
        plt.ylabel('Nilai Silhouette')
        plt.savefig('static/silhouette.png')
        plt.close()

        # Ambil k terbaik dari silhouette tertinggi
        best_k = k_range[silhouette_scores.index(max(silhouette_scores))]

        # Clustering akhir
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df['Cluster'] = kmeans_final.fit_predict(X_scaled)

        # Visualisasi hasil clustering (pakai Age & Salary)
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x='Age', y='EstimatedSalary', hue='Cluster', palette='viridis')
        plt.title(f'Hasil Clustering (k = {best_k})')
        plt.xlabel('Umur')
        plt.ylabel('Perkiraan Gaji')
        plt.savefig('static/hasil.png')
        plt.close()
        df.to_csv('static/hasil_clustering.csv', index=False)


        return df

    except Exception as e:
        print("Gagal memproses clustering:", e)
        return None
