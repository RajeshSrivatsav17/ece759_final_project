import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load full volume
df = pd.read_csv("density.csv")

nx = df['x'].max() + 1
ny = df['y'].max() + 1
nz = df['z'].max() + 1

volume = np.zeros((nz, ny, nx))

for _, row in df.iterrows():
    i, j, k = int(row['x']), int(row['y']), int(row['z'])
    volume[k, j, i] = row['density']

# Plot all z slices
for k in range(nz):
    plt.imshow(volume[k], cmap='inferno', origin='lower')
    plt.title(f"Slice z = {k}")
    plt.colorbar()
    plt.pause(0.5)
    plt.clf()  # Clear figure for next slice
