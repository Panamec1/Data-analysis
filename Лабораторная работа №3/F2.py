import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


import warnings
warnings.filterwarnings("ignore")


# Оценка корректности ответа
# через оценуц MAPE
def mape(x,y):
    mask = x != 0
    return (np.fabs(x-y)/x)[mask].mean()

def counter(df1):
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
 
# Тест Эдда-Фуллера на стационарность
def tester(df2,st):
    p = adfuller(df2[st][1:])[1]
    print(p)
    if p<0.05:
        print("Ряд стационарный")
    else:
        print("Ряд не стационарный")

# Дифференцирование
def diff(df2, st):
    y=df2[st]
    t = [y[i+1]-y[i] for i in range(0, len(df2)-1)]
    t.append(0)
    return t
    

    

df=pd.read_csv('data\\Amazon_Historical_StockPrice2.csv')
df1 = pd.DataFrame.from_records(df, columns=['Date','High', 'Low','Volume'])
df1 = df1.rename(columns={'High':  'h','Low': 'l','Date':'d','Volume':'v'},inplace=False)

plt.plot(df1.l)
plt.title('Amazon stock price')
plt.xlabel('dates')
plt.ylabel('price')
plt.show()


# Переподсчет средней цены не за день, а за месяц
df2 = counter(df1)
plt.plot(df2.l)
plt.show()

tester(df2,'l')

# Определение тренда
X = [i for i in range(0, len(df2))]
X = np.reshape(X, (len(X), 1))
model = LinearRegression()
model.fit(X, df2.l)

plt.plot(df2.l)
plt.plot(model.predict(X))
plt.show()

# Если ряд стационарен 
# больше шанс того, что в нем нет тренда
re=diff(df2,'l')
df2 = df2.assign(t=re)


# Уменьшим дисперсию с помощью логарифмирования

# Логарифмирование 
df2=df2.assign(t=np.log(df2.l))
# Диффиренцирование
df2.t=diff(df2,'t')


plt.plot(df2.t)
plt.show()

tester(df2,'t')
plt.plot(df2.t)
plt.show()



X=df2.drop('t', axis = 1)
Y=df2.t

# Будем делать предсказания в моделе вплоть начиная с 40 месяца до 59-го
# Все, что ранее 40 будут использованны для обучения регрессивной модели
# y и z хранят в себе значения значения в i месяц и i-ый месяц.
y,z=[],[]
for i in range(40):
    y.append(df2.t[i])
    z.append(i)

df3 = pd.DataFrame()
df3 = df3.assign (c=z,t=y)

# Тренируем модель с помощью метода SARIMAX
results = []
best_aic = float("inf")
model = sm.tsa.statespace.SARIMAX(
            df3.t, 
            order=(2,1, 1), 
            seasonal_order=(2,1,1, 40)
        ).fit(disp=-1)
print(model.summary())

# Сделаем предсказание для модели до 59-го дня
y=list(np.exp(model.predict(1,60)))
for i in range(len(y)):
    y[i]=y[i]-1
plt.plot(df2.t)
plt.plot(y,c='r')
plt.show()



# Высокая степень ошибки ввиду погрешности
# и возможная неточность в выборе параметров для Sarimax
print(mape(df2.t,y))




