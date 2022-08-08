import matplotlib
import numpy as np

matplotlib.use("GTK3Cairo")

import matplotlib.pyplot as plt

x = np.arange(10)
y = x ** 2

plt.plot(x, y)
plt.show()
