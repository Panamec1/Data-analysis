import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy import stats



def gradCof8(df1,p,log):
    
    if (log):
        y=df1[p].apply(np.log)
    else:
        y=df1[p]
        
    z=[]
    counter=0
    for i in range(len(y)):
        counter=counter+y[i]
        
    mc=counter/(len(y))
    m2=mc/(1.5)
    m1=mc/3
    m3=mc*1.5
    m4=mc*3
    for i in range(len(y)):
        if y[i] < m1:
            z.append(1)
            continue
        if y[i] < m2 :
            z.append(2)
        if y[i] < mc:
            z.append(3)
            continue
        if y[i] < m3:
            z.append(4)
            continue
        if y[i] < m4:
            z.append(5)
            continue
        if y[i] >= m4:
            z.append(6)
            continue
    #print(z)
    return z

def rec(df2, st):
    y=df2[st]
    t = [y[i+1]-y[i] for i in range(0, len(df2)-1)]
    t.append(0)
    return t



def spGrad(df1,p,log):
    
    if (log):
        y=df1[p].apply(np.log)
    else:
        y=df1[p]
        
    z=[]
    counter=0
    for i in range(len(y)):
        counter=counter+y[i]
        
    mc=counter/(len(y))
    m1=mc/(1.5)
    m2=mc*1.5
    #mc=mc/(cof)
    for i in range(len(y)):
        if y[i] < m1:
            z.append(1)
            continue
        if y[i] < mc:
            z.append(2)
            continue
        if y[i] < m2:
            z.append(3)
            continue
        if y[i] >= m2:
            z.append(4)
            continue
    
    return z



def AdsTester(df1,p,b):
    print()
    value = adfuller(df1[p])[1]
    if value >0.05:
        print(value, ">", 0.05 )
        print("Нулевая гипотеза принимается")
        print("Ряд не стационарен")
    else:
        print(value,"<",0.05)
        print("Принимается гипотеза альтернативная нулевой")
        print("Ряд стационарен")
    plt.hist(df1[p], density=True, bins=b)
    plt.show()
    




df=pd.read_csv('data\\Apple_Historical_StockPrice2.csv')
df1 = pd.DataFrame.from_records(df, columns=['Date',
                                             'High',
                                             'Low',
                                             'Volume','Open','Close'])
df1 = df1.rename(columns={'High':  'h',
                          'Low': 'l',
                          'Date':'d',
                          'Volume':'v',
                          'Close':'c',
                          'Open':'o'},inplace=False)

x=[]
u=[]
for i in range(len(df1['l'])):
    x.append((df1['l'][i]+df1['h'][i])/2)
    u.append(i)
    
df1 = df1.assign(mid=x)
df1 = df1.assign(i=u)
plt.plot(df1.mid)
plt.title('Apple stock price')
plt.xlabel('dates')
plt.ylabel('price')
plt.show()



df1.mid=np.log(df1.mid)
df1.mid=rec(df1,'mid')
# Выборка по специальным значениям
z=spGrad(df1,'mid',False)

df1 = df1.assign(gradM=z)
AdsTester(df1,'gradM',3)
plt.plot(df1.gradM)
plt.title('Apple special price gradation')
plt.xlabel('dates')
plt.ylabel('gradation')
plt.show()


print("v")
#AdsTester(df1,'gradM',4)
AdsTester(df1,'v',3)
z=spGrad(df1,'v',False)

df1 = df1.assign(gradV=z)
plt.plot(df1.gradV)
plt.title('Apple vol gradation')
plt.xlabel('dates')
plt.ylabel('gradation')
plt.show()


AdsTester(df1,'h',3)
z=spGrad(df1,'h',False)

df1 = df1.assign(gradH=z)
plt.plot(df1.gradH)
AdsTester(df1,'gradH',3)
plt.title('Apple high gradation')
plt.xlabel('dates')
plt.ylabel('gradation')
plt.show()

AdsTester(df1,'l',3)
z=spGrad(df1,'l',False)


df1 = df1.assign(gradL=z)
plt.plot(df1.gradL)
plt.title('Apple low-price gradation')
plt.xlabel('dates')
plt.ylabel('gradation')
plt.show()
df1.pop('mid')

#df1.pop('gradM')
df1.pop('gradH')
df1.pop('gradL')
df1.pop('gradV')

df1.pop('l')
df1.pop('h')

df1.pop('d')

df1.pop('v')
df1.pop('i')

print()
print(df1.columns)
print()

knn = RandomForestClassifier()
X=df1.drop('gradM', axis = 1)
Y=df1['gradM']
X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, stratify=Y, test_size=0.2)

knn.fit(X_train,y_train)
y=knn.predict(X_test)
plt.plot(y)
#plt.show()

print()
print(metrics.balanced_accuracy_score(y_test,y))
print(metrics.accuracy_score( y,y_test))



