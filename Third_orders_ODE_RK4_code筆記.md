# Third oders ODE Runge-Kutta Method four orders 筆記
## 概念
這個方法主要是用於進行計算高階的微分方程式(ODE)，在計算的共乘中會與泰勒展開式互相有關係。

### RK4 公式
$$
      \begin{align}
      k1 &= f(x_n,y_n)\\
      k2 &= f(x_n + \frac{h}{2}k_1 , y_n + \frac{h}{2} )\\
      k3 &= f(x_n + \frac{h}{2}k_2 , y_n + \frac{h}{2} )\\
      k4 &= f(x_n + hk_3,y_n + h)\\
      \end{align}
$$
將上述的(1)到(4)式的公式算完，可以獲得$k_1$$k_2$$k_3$$k_4$。
接下來進行計算$y_{n+1}$:，並請用下列公式進行求取。

$y_{n+1} = y_0 + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4) $

完成計算後提高$x_{n+1}$作為下一次循環 ${\rm Step}$。

$x_{n+1} = x_0 + {\rm step}$

**p.s:如果使用矩陣的方式計算，可以不用加入上述的 $x_{n+1}$ 而矩陣間距為 ${\rm step}$。**

## 指令呈現
### 手打Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

if __name__ == '__main__':
    # Define the third order ODE
    def f(x, y):
        dy = np.zeros(4)
        dy[0] = y[1] #y'
        dy[1] = y[2] #y''
        dy[2] = -2 * y[2] + y[1] +2 * y[0] + np.exp(x) #y'''

        return dy

    '''
    可以確定
    dy/dx = y1 = dy[0]
    dy1/dx = y2 = dy[1]
    dy2/dx = u3 = dy[2]
    '''
    # Define the initial conditions
    y_0 = 1
    y_1 = 2
    y_2 = 0
    x0 = 0
    ass = (np.exp(x0) + 2 * y_0  + y_1 - 2 * y_2)
    y0 = np.array([1, 2, 0, ass])
    x0 = np.array([0, 0, 0, 0])



    # Define the step size
    h = 0.2

    # Define the interval of x to solve over
    x = np.arange(0, 3.2, h)
    x = np.transpose([x])


    # Initialise the approximation array
    y = np.zeros([len(x),len(y0)])
    y[0] = y0

    # Loop through the time steps, approximating this step from the prev step
    for i, x_i in enumerate(x[:-1]):

        k_1 = (f(x_i, y[i]))
        k_2 = (f(x_i + h / 2., y[i] + h / 2. * k_1))
        k_3 = (f(x_i + h / 2., y[i] + h / 2. * k_2))
        k_4 = (f(x_i + h, y[i] + h * k_3))
        y[i + 1] = y[i] + h / 6. * (k_1 + 2. * k_2 + 2. * k_3 + k_4)  # RK4

    #因為y會因為矩陣的循環因素所以會少一個因此要獨立設立x1
    x1 = np.arange(0, 3.2, h)
    x1 = np.transpose([x1])

    #print(y[:,3])
    #plt.plot(x1,y[:,3])


    #完美答案
    x2 = np.linspace(0,3,50)
    y2 = (43/36) * np.exp(x2) + (1 / 4) * np.exp(-x2) - (4 / 9) * np.exp(-2 * x2) + (1 / 6) * x2 * np.exp(x2)
    plt.plot(x1,y[:,0],'--b',x2,y2,'-r')
    plt.show()
    #plt.savefig('np.exp(x) + 2 * y[0] + y[1] - 2 * y[2]', dpi=300)

```
上述指令是訂一次手打的版本，但是有非常多沒有必要得指令。
等等會依照指令的比較進行比較優化修正前後。

### 電腦版本
```python

 if __name__ == '__main__':
     # Define the third order ODE
     def f(x, y):
         dy = np.zeros(4)
         dy[0] = y[1] #y'
         dy[1] = y[2] #y''
         dy[2] = -2*y[2]+y[1]+2*y[0]+np.exp(x) #y'''
         return dy

     # Define the initial conditions
     y0 = [1, 2, 0, 0]

     # Define the step size
     h = 0.2

     # Define the interval of x to solve over
     x = np.arange(0, 3+h, h)
     x = np.transpose([x])

     # Initialise the approximation array
     y = np.zeros([len(x), len(y0)])
     y[0] = y0

     # Loop through the time steps, approximating this step from the prev step
     for i, x_i in enumerate(x[:-1]):
         k_1 = f(x_i, y[i])
         k_2 = f(x_i + h / 2., y[i] + h / 2. * k_1)
         k_3 = f(x_i + h / 2., y[i] + h / 2. * k_2)
         k_4 = f(x_i + h, y[i] + h * k_3)
         y[i + 1] = y[i] + h / 6. * (k_1 + 2. * k_2 + 2. * k_3 + k_4)   RK4

     plt.plot(x, y[:, 0], label='y')
     plt.plot(x, y[:, 1], label='y\'')
     plt.plot(x, y[:, 2], label='y\'\'')
     plt.legend()
     plt.show()
 ```

就由上面的指令可以發手打的版本比要多無用指令，但是電腦版本的無用指令很少。

-針對# Define the initial conditions 的部分：

(手寫版本)
```python
    # Define the initial conditions
    y_0 = 1
    y_1 = 2
    y_2 = 0
    x0 = 0
    ass = (np.exp(x0) + 2 * y_0  + y_1 - 2 * y_2)
    y0 = np.array([1, 2, 0, ass])
    x0 = np.array([0, 0, 0, 0])
```

(電腦版本)
```python
 # Define the initial conditions
 y0 = [1, 2, 0, 0]

```
 

<style type="text/css">
    h1 { counter-reset: h2counter; }
    h2 { counter-reset: h3counter; }
    h3 { counter-reset: h4counter; }
    h4 { counter-reset: h5counter; }
    h5 { counter-reset: h6counter; }
    h6 { }
    h2:before {
      counter-increment: h2counter;
      content: counter(h2counter) ".\0000a0\0000a0";
    }
    h3:before {
      counter-increment: h3counter;
      content: counter(h2counter) "."
                counter(h3counter) ".\0000a0\0000a0";
    }
    h4:before {
      counter-increment: h4counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) ".\0000a0\0000a0";
    }
    h5:before {
      counter-increment: h5counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) ".\0000a0\0000a0";
    }
    h6:before {
      counter-increment: h6counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) "."
                counter(h6counter) ".\0000a0\0000a0";
    }
    </style>