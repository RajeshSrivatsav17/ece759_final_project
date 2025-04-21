import numpy as np
import pandas as pd
import pyvista as pv
import glob
import os

frame_paths = sorted(glob.glob("density_frame_*.csv"))

#Grid resolution
nx, ny, nz = 64, 64, 64
dims = (nx + 1, ny + 1, nz + 1)
spacing = (1.0, 1.0, 1.0)
origin = (0, 0, 0)

grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)

plotter = pv.Plotter(off_screen=True)
plotter.open_gif("smoke_volume_animation.gif")

for path in frame_paths:
    print(f"Rendering: {path}")
    df = pd.read_csv(path)
    volume = np.zeros((nz, ny, nx))

    for _, row in df.iterrows():
        i, j, k = int(row['x']), int(row['y']), int(row['z'])
        volume[k, j, i] = row['density']

    volume -= volume.min()
    volume /= volume.max() + 1e-8

    grid.cell_data["density"] = volume.flatten(order="F")
    plotter.clear()
    plotter.add_volume(grid, scalars="density",
                       opacity=[0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0],
                       cmap="gray_r")
    plotter.add_axes()
    plotter.write_frame()

plotter.close()
print("Generated smoke_volume_animation.gif'")