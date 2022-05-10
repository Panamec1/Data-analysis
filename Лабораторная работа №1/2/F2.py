import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.interpolate import barycentric_interpolate

x1=np.linspace(0.0,10.0,11)
y1=np.sin(x1)

#Кусочный кубический интерполяционный полином Эрмита
#
#Использует кусочный кубический полином p(x)
# () на кажом подинтервале x_i <= x <= x_(i+1)  для p полинома
# () первая производная непрерывная p(x) интерполирует y
# () кубический интерполант сохраняет форму Наклоны в xj выбраны таким способом,
#    который сохраняет форму данных 


ze=np.linspace(min(x1), max(x1), num=100)
spl = pchip_interpolate(x1,y1,ze)
plt.plot(x1,y1,"o",label="observation")
plt.plot(ze,spl,label="pchip interpolation")

# Барицентрическая интерполяция
ze=np.linspace(min(x1),max(x1),num=100)
spl = barycentric_interpolate(x1,y1,ze)
#plt.plot(ze,spl,label="barycentric interpolation")
plt.show()
