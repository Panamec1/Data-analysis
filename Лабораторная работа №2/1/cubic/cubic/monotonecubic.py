import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class monospline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = self.y.size
        self.h = self.x[1:] - self.x[:-1]
        self.m = (self.y[1:] - self.y[:-1]) / self.h
        self.a = self.y[:]
        self.b = self.compute_b(self.x, self.y)
        self.c = (3 * self.m - self.b[1:] - 2 * self.b[:-1]) / self.h
        self.d = (self.b[1:] + self.b[:-1] - 2 * self.m) / (self.h * self.h)

    def compute_b(self, t, r):
        b = np.empty(self.n)
        for i in range(1, self.n - 1):
            is_mono = self.m[i - 1] * self.m[i] > 0
            if is_mono:
                b[i] = 3 * self.m[i - 1] * self.m[i] / (max(self.m[i - 1], self.m[i]) + 2 * min(self.m[i - 1], self.m[i]))
            else:
                b[i] = 0
            if is_mono and self.m[i] > 0:
                b[i] = min(max(0, b[i]), 3 * min(self.m[i - 1], self.m[i]))
            elif is_mono and self.m[i] < 0:
                b[i] = max(min(0, b[i]), 3 * max(self.m[i - 1], self.m[i]))

        b[0] = ((2 * self.h[0] + self.h[1]) * self.m[0] - self.h[0] * self.m[1]) / (self.h[0] + self.h[1])
        b[self.n - 1] = ((2 * self.h[self.n - 2] + self.h[self.n - 3]) * self.m[self.n - 2]
                         - self.h[self.n - 2] * self.m[self.n - 3]) / (self.h[self.n - 2] + self.h[self.n - 3])
        return b

    def evaluate(self, t_intrp):
        ans = []
        #for tau in range(len(t_intrp)):
        #    i=tau
        for tau in t_intrp:
            
            i = np.where(tau >= self.x)[0]
            if i.size == 0:
                i = 0
            else:
                i = i[-1]
            i = min(i, self.n-2)
            res = self.a[i]+ self.b[i] * (tau - self.x[i]) + self.c[i] * ((tau - self.x[i])**2) + self.d[i] *((tau - self.x[i])**3)
            ans.append(res)
        return ans





x1=np.linspace(0.0,10.0,11)
y1=np.sin(x1)
plt.plot(x1, y1, 'o')

intrp = monospline(x1, y1)
x = np.linspace(min(x1), max(x1), 5000)
y = intrp.evaluate(x)
plt.plot(x, y)


intrp = CubicSpline(x1, y1)
plt.plot(x, intrp(x))
plt.show()
