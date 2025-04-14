import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return(2*x**2)



def appx_tangent_line(x, appx_der, b):
    return(appx_der*x+b)

for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1+p2_delta

    y1 = f(x1)
    y2 = f(x2)

    appx_der = (y2-y1)/(x2-x1)
    b = y2 - appx_der*x2

x = np.arrange(0,50, 0.001)
y = f(x)

plt.plot(x,y)

p2_delta = 0.001

x1 = 1
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)


to_plot = [x1-0.9, x1, x1+0.9]
plt.plot(to_plot,
         [appx_tangent_line(point, appx_der, b)
          for point in to_plot],
          c=colors[i])
        
plt.show()