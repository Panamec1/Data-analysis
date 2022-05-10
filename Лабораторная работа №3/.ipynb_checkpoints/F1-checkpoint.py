import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

def mape(x,y):
    mask = x != 0
    return (np.fabs(x-y)/x)[mask].mean()

def r(df1):
    x=[]
    y=[]
    z=[]
    cl,ch,c = 0,0,0
    dat = int(df1['d'][0][5:7])
    for i in range(len(df1['l'])):
        da = int(df1['d'][i][5:7])
        cl = cl + df1['l'][i]
        ch = ch + df1['h'][i]
        c=c+1
        if da != dat :
            z.append(df1['d'][i][:7])
            x.append(cl/c)
            y.append(ch/c)
            cl,ch,c,dat=0,0,0,da

    df2 = pd.DataFrame()
    df2 = df2.assign(l=x,h=y,d=z)
    
    return df2


def tester(df2,st):
    p = adfuller(df2[st][1:])[1]
    print(p)
    if p<0.05:
        print("Ряд стационарный")
    else:
        print("Ряд не стационарный")

def rec(df2, st):
    y=df2[st]
    t = [y[i+1]-y[i] for i in range(0, len(df2)-1)]
    t.append(0)
    return t
    

    

df=pd.read_csv('data\\Netflix_Historical_StockPrice2.csv')
df1 = pd.DataFrame.from_records(df, columns=['Date','High', 'Low','Volume'])
df1 = df1.rename(columns={'High':  'h','Low': 'l','Date':'d','Volume':'v'},inplace=False)

plt.plot(df1.l)
plt.title('Netflix stock price')
plt.xlabel('dates')
plt.ylabel('price')
plt.show()



df2 = r(df1)
plt.plot(df2.l)
plt.show()

tester(df2,'l')

X = [i for i in range(0, len(df2))]
X = np.reshape(X, (len(X), 1))
model = LinearRegression()
model.fit(X, df2.l)

plt.plot(df2.l)
plt.plot(model.predict(X))
plt.show()


re=rec(df2,'l')
df2 = df2.assign(t=re)
plt.plot(df2.t)
plt.show()



df2.t=np.log(df2.l)
df2.t=rec(df2,'t')

tester(df2,'t')
plt.plot(df2.t)
plt.show()



X=df2.drop('t', axis = 1)
Y=df2.t

x,y,z=[],[],[]
for i in range(40):
    y.append(df2.t[i])
    z.append(i)

for i in range(60):
    x.append(i)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, stratify=Y, test_size=0.2)
d=[]
d.append(z)
d.append(y)



d=1
D=1
qs = range(1, 5)
Qs = range(1, 3)
ps = range(1, 5)
Ps = range(3, 5)
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)

df3 = pd.DataFrame()
df3 = df3.assign (c=z,t=y)
model = ARIMA(
            df3.t, 
            order=(3, 1, 2), 
            seasonal_order=(2, 1,2, 12)
        )
model = model.fit()

h=model.predict(40,59)
for j in range(40,59):
    y.append(h[i])
plt.plot(y,c='r')
plt.plot(df2.t)
plt.show()

results = []
best_aic = float("inf")
model = sm.tsa.statespace.SARIMAX(
            df3.t, 
            order=(2,1, 1), 
            seasonal_order=(2,1,1, 40)
        ).fit(disp=-1)

print(model.summary())

#df4=pd.Dataframe()
y=list(np.exp(model.predict(1,60)))
for i in range(len(y)):
    y[i]=y[i]-1
plt.plot(df2.t)
plt.plot(y,c='r')
plt.show()

print(len(y))
print(len(df3.t))

print(mape(df2.t,y))




