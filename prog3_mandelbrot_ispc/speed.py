#!/usr/bin/env python

import matplotlib.pyplot as plt

x = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40]
y = [ 3.51, 6.70, 8.74, 8.73, 14.49, 16.22, 21.25, 27.59, 26.58, 32.73, 36.19 ]

plt.plot(x, y, marker='o')
plt.xlabel('X-tasks')
plt.ylabel('Y-Speed up')
plt.title('ispc tasks speed up')
plt.grid(True)
plt.show()
