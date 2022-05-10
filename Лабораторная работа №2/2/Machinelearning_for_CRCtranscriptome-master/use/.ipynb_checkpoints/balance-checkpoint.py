from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns






def done(name:str)->None:
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
