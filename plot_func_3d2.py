import numpy as np
import pandas as pd
import pyvista as pv

def load_volume_csv(filename, scalar_field, shape_order='zyx'):
    df = pd.read_csv(filename)
    nx = int(df['x'].max()) + 1
    ny = int(df['y'].max()) + 1
    nz = int(df['z'].max()) + 1
    volume = np.zeros((nz, ny, nx))  # (z, y, x)

    for _, row in df.iterrows():
        i, j, k = int(row['x']), int(row['y']), int(row['z'])
        volume[k, j, i] = row[scalar_field]

    return volume, (nx, ny, nz)

# Load smoke density and temperature fields
density, (nx, ny, nz) = load_volume_csv("density.csv", "density")
temperature, _ = load_volume_csv("Temperature.csv", "temperature")

# Create shared grid geometry
dims = (nx + 1, ny + 1, nz + 1)
spacing = (1.0, 1.0, 1.0)

# Create two ImageData grids for rendering
grid_density = pv.ImageData(dimensions=dims, spacing=spacing)
grid_temperature = pv.ImageData(dimensions=dims, spacing=spacing)

# Assign scalar fields (flattened in z-fastest order)
grid_density.cell_data["density"] = density.flatten(order="F")
grid_temperature.cell_data["temperature"] = temperature.flatten(order="F")

plotter = pv.Plotter()

plotter.add_volume(grid_density, scalars="density", opacity="sigmoid",
                   cmap="inferno", name="Smoke", show_scalar_bar=True)
plotter.add_volume(grid_temperature, scalars="temperature", opacity=[0.0, 0.2, 0.5, 0.8, 1.0],
                   cmap="coolwarm", name="Temperature", show_scalar_bar=True)

# Show combined plot
plotter.add_axes()
plotter.show()
