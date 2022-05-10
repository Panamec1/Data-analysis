import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('data\\Netflix_Historical_StockPrice2.csv')
df1 = pd.DataFrame.from_records(df, columns=['Date','High', 'Low','Volume'])
df1 = df1.rename(columns={'High':  'h','Low': 'l','Date':'d','Volume':'v'},inplace=False)
x=[]
for i in range(len(df1['l'])):
    x.append((df1['l'][i]+df1['h'][i])/2)
df1 = df1.assign(mid=x)
plt.plot(df1.mid)
plt.title('Netflix stock price')
plt.xlabel('dates')
plt.ylabel('price')
plt.show()

#ax = plt.axes(projection='3d')
#ax.scatter(df1.h, df1.l, df1.v)
#plt.show()

plt.boxplot(df1.mid, vert=False, whis=0.8)
plt.show()

df1.pop('d')
df1.pop('v')
plt.boxplot(df1,labels=df1.columns)
plt.show()


df=pd.read_csv('data\\case.csv')
df1 = pd.DataFrame.from_records(df, columns=['province','confirmed'])
df1 = df1.rename(columns={'province':  'p','confirmed': 'c'},inplace=False)
plt.hist(df1.p)
plt.title('Counting the reporing of province in data')
plt.xlabel('province')
plt.ylabel('number')
plt.show()




df=pd.read_csv('data\\Region.csv')
df1 = pd.DataFrame.from_records(df, columns=['elementary_school_count','kindergarten_count','elderly_population_ratio','elderly_alone_ratio'])
df1 = df1.rename(columns={'elementary_school_count':  'e', 'kindergarten_count': 'k',
                          'elderly_population_ratio':'popul','elderly_alone_ratio':  'alone'},inplace=False)
df1.fillna(0, inplace=True)
a=[]
for i in range(int(len(df1.k)/2)):
    a.append(i)
plt.hist2d(df1.e,df1.k,bins=a)
plt.title('Elementary-kindergarten dependency')
plt.xlabel('elementary')
plt.ylabel('kindergartens')
plt.show()


df1.fillna(0, inplace=True)
plt.scatter(df1.popul,df1.alone)
plt.title('Elderly')
plt.xlabel('poulation')
plt.ylabel('alone')
plt.show()

    

