import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns
import smote

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from collections import Counter


import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def usage(name:str)->None:
    fe_boru=pd.read_csv(name)
    print(fe_boru.outcome.value_counts())
    print()
    print()
    print(fe_boru.head())
    
    
def sk(name:str,gen:bool)->None:
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
    
    

    
def balance(name:str)->None:
    # ML methods part
    fe_boru=pd.read_csv(name)
    fe_boru=fe_boru.reindex(np.random.permutation(fe_boru.index))
    X = fe_boru.drop('outcome', axis=1)
    y=fe_boru['outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    
    print(sorted(Counter(y_test).items()))
    
    fe_boru=pd.read_csv(name)
    fe_boru=fe_boru.reindex(np.random.permutation(fe_boru.index))
    X = fe_boru.drop('outcome', axis=1)
    y=fe_boru['outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    print("Sorted Counter")
    print(sorted(Counter(y_test).items()))
    print()
    print()
    
    rf=RandomForestClassifier(n_estimators=20,random_state=50)
    rf.fit(X_train,y_train)
    
    
    pred_test=rf.predict(X_test)
    print("classification_report")
    print(classification_report(y_test,pred_test))
    print('\n')
    print("confusion_matrix")
    print(confusion_matrix(y_test,pred_test))
    print('\n\n\n\n\n')
    
    print(rf.feature_importances_)
    feature_imp=pd.Series(rf.feature_importances_,
                          index=list(fe_boru.drop('outcome',axis=1).columns.values)).sort_values(ascending=False)
    
    
    plt.figure(figsize=(10,10))
    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    sns.set(font_scale=0.9)
    
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.show()
    plt.savefig('importance ranking for female')
    plt.close()