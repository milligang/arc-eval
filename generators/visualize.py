import numpy as np
from matplotlib import pyplot as plt

from arc import plot_grid

grid = np.array(
    [[1,0],
    [0,1]]
)

plot_grid(grid)
plt.show()