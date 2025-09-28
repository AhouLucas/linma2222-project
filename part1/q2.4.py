# Small plot script for question 2.4



import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np



def c(g):
    return np.maximum(g - 1/2 * g * g, 1 - np.exp(-g))


def f(x):
    return 1 / (2 * np.sqrt(3) * x) * quad(c, -np.sqrt(3) * x, np.sqrt(3) * x)[0]




x = np.linspace(0.01, 1, 100)

y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = f(x[i])


plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$F(\sigma)$')
# plt.xlim(0, 5)
# plt.ylim(-5, 2)
plt.title(r'Plot of $F(\sigma)$')
plt.grid(alpha=0.5)
plt.savefig('part1/figures/plot_q2.4.svg', format='svg')
plt.show()