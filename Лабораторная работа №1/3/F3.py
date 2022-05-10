import matplotlib.pyplot as plt
import numpy as np
import time
import random


def hi(arr):
    d={}
    for i in range(len(arr)):
        if arr[i] in d:
            d[arr[i]]=d[arr[i]]+1
        else:
            d[arr[i]]=1
    return d


def his(arr,a):
    mi=min(arr)
    ma=max(arr)
    h=(ma-mi)
    coef=h/a
    b=np.zeros(a)
    
    for i in range(len(arr)):
        pos =int((arr[i]-mi)/coef)
        b[min(pos,a-1)]+=1
        
    c=np.arange(mi,ma,a)
    print(c,b)
    f=[]
    f.append(c)
    f.append(b)
    return b,c
        
    
    
    


a = [0.2,0.2,0.1,0.5]
for i in range(50):
    a.append(random.randint(1,10))

start_time = time.time()

value_counts = plt.hist(a, width=0.9)
plt.hist(a)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
#--- 0.1678144931793213 seconds ---

#start_time = time.time()
#d=hi(a)
#sorted_tuple = sorted(d.items(), key=lambda x: x[0])
#d=dict(sorted_tuple)

#numbers = list(d.keys())
#amountOfNumbers= list(d.values())

#plt.bar(list(d),amountOfNumbers,width = 0.4)
#plt.show()
#print("--- %s seconds ---" % (time.time() - start_time))
#--- 0.12401819229125977 seconds ---

b,c = his(a,4)
plt.bar(b,c,width = 0.4)
plt.show()


