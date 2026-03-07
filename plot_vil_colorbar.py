import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

COLOR_MAP = [
    [0, 0, 0],
    [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
    [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
    [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
    [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
    [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
    [0.9607843137254902, 0.9607843137254902, 0.0],
    [0.9294117647058824, 0.6745098039215687, 0.0],
    [0.9411764705882353, 0.43137254901960786, 0.0],
    [0.6274509803921569, 0.0, 0.0],
    [0.9058823529411765, 0.0, 1.0],
]

BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]

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
cb.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig('colorbar.png', dpi=300, bbox_inches='tight')
plt.show()