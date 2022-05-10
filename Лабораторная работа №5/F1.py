from sklearn import datasets
from sklearn.manifold import TSNE

import numpy as np



import pandas as pd
import matplotlib.pyplot as plt
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def ma(df1):
    m=0
    j=0
    for i in range(len(df1)):
       if m<df1.e[i]:
           m=df1.e[i]
           j=i
    df1.drop(j, inplace = True)
    return df1


def exter(t,n,m):
    a=[]
    b=[]
    for i in range(len(df1)):
        if t.cluster[i] == n:
            a.append([t.a[i],t.b[i]])
        if t.cluster[i] == m:
            b.append([t.a[i],t.b[i]])
    m=10000000
    for i in range(len(a)):
        for j in range(len(b)):
            c=((a[i][0] - b[j][0])**(2)+(a[i][1] - b[j][1])**(2))**(0.5)
            if (m>c):
                m=c
    return m

def inter(t,n):
    a=[]
    for i in range(len(df1)):
        if t.cluster[i] == n:
            a.append([t.a[i],t.b[i]])
    m=0
    for i in range(len(a)):
        for j in range(len(a)):
            if i>j:
                c=((a[i][0] - a[j][0])**(2)+(a[i][1] - a[j][1])**(2))**(0.5)
                if (m<c):
                    m=c
    return m



df=pd.read_csv('data\\Region.csv',index_col=0)
df1 = pd.DataFrame.from_records(df, columns=['elementary_school_count','kindergarten_count','elderly_population_ratio','elderly_alone_ratio'])
df1 = df1.rename(columns={'elementary_school_count': 'e', 'kindergarten_count': 'k',
                          'elderly_population_ratio': 'popul','elderly_alone_ratio': 'alone',
                          },inplace=False)

df1=ma(df1)
df1=ma(df1)

df1.fillna(0, inplace=True)
plt.scatter(df1.e,df1.k)
plt.title('Elem-kinder')
plt.xlabel('elem')
plt.ylabel('kind')
plt.show()

df1.index.names = ['region']


pca = PCA(copy=True,n_components=2,whiten=False)
pca.fit(df1)
ex = pca.transform(df1)
ext = pd.DataFrame(ex)
ext.index = df1.index
ext.coumns = ['PC1','PC2']
print(ext.head())


df1.pop('alone')
#df1.pop('popul')




X=df1.e
y=df1.k
pr = KMeans(n_clusters=3).fit(df1)
centroids = pr.cluster_centers_
print(centroids)

plt.scatter(df1.e,df1.k, c=pr.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

#df1.pop('alone')
print(df1)

ts = PCA(n_components=3).fit_transform(df1)
t=pd.DataFrame(ts)
t.columns = ['a','b','c']
plt.figure(figsize=(14,12))
plt.scatter(ts[:, 0], ts[:, 1], c=y, 
            alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.show()

pr = KMeans(n_clusters=3).fit(ts)

t['cluster'] = pd.Series(pr.labels_, index=df1.index)
centroids = pr.cluster_centers_
print(centroids)


print("Внутри нулевого класстера внутренняя мера:",inter(t,0))
print("Внутри первого класстера внутренняя мера:",inter(t,1))
print("Внутри второго класстера внутренняя мера:",inter(t,2))

print("Внешняя мера между нулевым и первым кластером:",exter(t,0,1))
print("Внешняя мера между первым и вторым кластером:",exter(t,1,2))
print("Внешняя мера между нулевым и вторым кластером:",exter(t,0,2))


plt.scatter(ts[:, 0],ts[:, 1], c=pr.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()






    

