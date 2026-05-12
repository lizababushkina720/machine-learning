import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge               
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA

def get_score(data_encoded):
    data_encoded = data_encoded.fillna(data_encoded.median())

    X = data_encoded.drop(columns=['SalePrice'])
    y = data_encoded['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train_scaled, y_train)
    return r2_score(y_test, model.predict(X_test_scaled))

def main():

    
    file_path = r"C:\Machine learning\AmesHousing.csv"
    df = pd.read_csv(file_path)

    # Очистка данных 
    toNone = ['Alley', 'Pool QC', 'Fence']
    for column in toNone: 
        df[column] = df[column].fillna("None")

    toZero = ['Bsmt Full Bath', 'Garage Area']
    for column in toZero:
        df[column] = df[column].fillna(0)


    df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(
        lambda x: x.fillna(x.median()))
    
    #Обработка категориальных признаков (One-Hot Encoding)
    
    data_encoded = pd.get_dummies(df, drop_first=True)

    #2. Удаление аномалий
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='Gr Liv Area', y='SalePrice')
    plt.title('Зависимость цены от площади')
    plt.show()

    r2_before = get_score(data_encoded)
    
    isoForest = IsolationForest(random_state=41)
    outliers = isoForest.fit_predict(data_encoded[['Gr Liv Area', 'SalePrice']])
    forRidge = data_encoded[outliers == 1]
    
    forRidge = forRidge.fillna(forRidge.median())
    r2_after = get_score(forRidge)

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=forRidge, x='Gr Liv Area', y='SalePrice')
    plt.title('Зависимость цены от площади (Gr Liv Area)')
    plt.show()

    print(f"R2 до очистки: {r2_before:.4f}")
    print(f"R2 после очистки: {r2_after:.4f}")

    # 3. Топ-10 признаков модели Ridge
    X_final = forRidge.drop(columns=['SalePrice'])
    y_final = forRidge['SalePrice']
    scaler_f = StandardScaler()
    X_final_scaled = scaler_f.fit_transform(X_final)

    ridge_final = Ridge(alpha=10.0)
    ridge_final.fit(X_final_scaled, y_final)

    importance = pd.DataFrame({
        'Feature': X_final.columns,
        'Weight': ridge_final.coef_
    })
    importance['Abs_Weight'] = importance['Weight'].abs() 
    top_10 = importance.sort_values(by='Abs_Weight', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_10, x='Abs_Weight', y='Feature')
    plt.title('Топ-10 самых важных признаков')
    plt.show()

    # 4. Кластеризация
    num_cols = forRidge.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != 'SalePrice']

    X_clust = StandardScaler().fit_transform(forRidge[num_cols])

    kmeans = KMeans(n_clusters=5, random_state=42)
    forRidge['Segment'] = kmeans.fit_predict(X_clust)

    print("Размер сегментов:\n", forRidge['Segment'].value_counts())

    # 5. PCA (Метод главных компонент) и регрессия 
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_clust)

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca, forRidge['SalePrice'], test_size=0.2, random_state=42
    )
    model_pca = Ridge()
    model_pca.fit(X_train_pca, y_train_pca)
    print(f"Качество модели на PCA (90% инфо): {r2_score(y_test_pca, model_pca.predict(X_test_pca)):.4f}")

    # 6. Анализ динамики цен и сезонности

    forRidge['House_Age'] = forRidge['Yr Sold'] - forRidge['Year Built']
    
    forRidge['Years_Since_Remod'] = forRidge['Yr Sold'] - forRidge['Year Remod/Add']

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=forRidge, x='Yr Sold', y='SalePrice', estimator='median', marker='o')
    plt.xticks(forRidge['Yr Sold'].unique())
    plt.title('Динамика цен по годам (2006-2010)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=forRidge, x='Mo Sold', y='SalePrice', estimator='median', palette='coolwarm')
    plt.title('Сезонность: Цена по месяцам')
    plt.show()



if __name__ == "__main__":
    main()