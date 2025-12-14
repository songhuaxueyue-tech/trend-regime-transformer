import numpy as np
import matplotlib.pyplot as plt

x = np.arange(5)
y = np.array([1, 2, 4, 4, 7])

coef = np.polyfit(x, y, 1)
print(coef)

plt.scatter(x, y)
plt.plot(x, coef[0] * x + coef[1], color='red')
plt.show()



