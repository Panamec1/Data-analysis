from sklearn.preprocessing import StandardScaler
from collections import Counter

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd


def done(name:str,gen:bool)->None:
    fe_boru=pd.read_csv(name)
    scaler=StandardScaler()
    df=fe_boru.drop('outcome', axis=1)
    scaler.fit(df)
    scaled_data=scaler.transform(df)
    
    pca=PCA(n_components=5)
    pca.fit(scaled_data)
    x_pca=pca.transform(scaled_data)
    
    t=np.asarray(fe_boru.outcome)
    t=t.astype(object)
    
    for i in range(len(t)):
        if t[i]==1:
            t[i]='cancer'
        else:
            t[i]='non-cancer'        
    pc_df=pd.DataFrame(data=x_pca,columns=['PC1','PC2','PC3','PC4','PC5'])
    pc_df['cluster']=t
    
    print("scaled_data.shape")
    print(scaled_data.shape)
    print()
    print()
    print("x_pca.shape")
    print(x_pca.shape)
    print()
    print()
    
    var= pca.explained_variance_ratio_
    
    ex =pd.DataFrame({'Eigenvalue coverage rate':var,'Principle Component':['PC1','PC2','PC3','PC4','PC5']})
    sns.barplot(x='Principle Component',y='Eigenvalue coverage rate',data=ex)

    p=''
    if (gen):
        p='PuBu_r'
    else:
        p='RdPu_r'
    sns.lmplot(x='PC1',y='PC2', data=pc_df,hue='cluster',legend=True, palette=p,scatter_kws={"s": 80})
    
    df_com = pd.DataFrame(pca.components_[0:2,:], columns=df.columns)
    
    pd.set_option("display.max_rows", 109)
    print(df_com.T)
    print()
    
    plt.figure(figsize=(12,6))
    sns.heatmap(df_com,cmap="coolwarm")
    print()
    print()
    print(plt)
