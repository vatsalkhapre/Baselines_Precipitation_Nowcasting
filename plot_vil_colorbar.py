import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

PIXEL_SCALE = 90.0
THRESHOLDS = [12, 18, 24, 32]
COLOR_MAP = ['lavender', 'indigo', 'mediumblue', 'dodgerblue', 'skyblue', 'cyan',
                                  'olivedrab', 'lime', 'greenyellow', 'orange', 'red', 'magenta', 'pink',]

BOUNDS = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, PIXEL_SCALE]

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255


# Use equally spaced bounds for visual display, but label with real values
equal_bounds = list(range(len(BOUNDS)))  # [0, 1, 2, ..., 10]

cmap = ListedColormap(COLOR_MAP)
norm = BoundaryNorm(equal_bounds, cmap.N)

fig, ax = plt.subplots(figsize=(12, 1.5))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, cax=ax, orientation='horizontal', spacing='uniform')

# Place ticks at each boundary and label with real BOUNDS values
cb.set_ticks(equal_bounds)
cb.set_ticklabels([str(int(b)) for b in BOUNDS])
cb.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig('colorbar.png', dpi=300, bbox_inches='tight')
plt.show()