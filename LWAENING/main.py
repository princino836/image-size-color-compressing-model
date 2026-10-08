# UNSUPERVISED MACHINE LEARNING

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score



df=pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())

x=df[["Age","Annual Income (k$)", "Spending Score (1-100)"]]
scale=StandardScaler()
x_sacled=scale.fit_transform(x)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
km.fit(x_sacled)
df["cluster"]=km.labels_
print(df)


plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["cluster"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()

inertial=[]
for k in range( 1,11):
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    random_state=42
    n_init=10
    model.fit(x_sacled)
    inertial.append(model.inertia_)
plt.plot(range(1,11),inertial,marker="o",label="Inertia")
plt.show()
new_customer = [[25, 70, 80]]

pred=model.predict(new_customer)
print(pred)
score=silhouette_score(x_sacled,km.labels_)
print(score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(x_sacled)

    score = silhouette_score(x_sacled, labels)

    print(f"K = {k}, Silhouette Score = {score:.3f}")