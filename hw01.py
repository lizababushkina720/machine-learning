import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def custom_kmeans(data, k, iters=100, seed=1):
    np.random.seed(seed)
    idx = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[idx].astype(float)
    
    for _ in range(iters):
        dists = np.array([np.linalg.norm(data - c, axis=1) for c in centroids]).T
        labels = np.argmin(dists, axis=1)
        
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if len(data[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    inertia = 0
    for i in range(k):
        points = data[labels == i]
        if len(points) > 0:
            inertia += np.sum((points - centroids[i])**2)
            
    return labels, centroids, inertia

def main():
    df = pd.read_csv("data.csv")
    
    X = df.values.astype(float)

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min )

    k_range = range(1, 20)
    inertia = []
    for k in k_range:
        res = custom_kmeans(X_norm, k)
        inertia.append(res[2]) 

    OPTIMAL_K = 4
    labels, centroids_10d, _ = custom_kmeans(X_norm, k=OPTIMAL_K)

    pca = PCA(n_components=2, random_state=1)
    X_pca = pca.fit_transform(X_norm)
    centroids_pca = pca.transform(centroids_10d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1.plot(k_range, inertia, 'ko--')
    ax1.axvline(x=OPTIMAL_K, color='red', linestyle=':')
    ax1.set_title("Метод локтя")
    ax1.set_xlabel("Количество кластеров")
    ax1.set_ylabel("inertia")

    colors = ["#E9E09F", '#4ECDC4', "#9B3AC19C", "#e308845f"]
    for i in range(OPTIMAL_K):
        points = X_pca[labels == i]
        ax2.scatter(points[:, 0], points[:, 1], s=15, c=colors[i], alpha=0.5)

    ax2.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                s=300, c='black', marker='*', edgecolors='white', linewidths=2)

    ax2.invert_yaxis()
    ax2.invert_xaxis()

    ax2.set_title("Кластеризация")
    ax2.grid(True, alpha=0.2)

    plt.show()

if __name__ == "__main__":
    main()