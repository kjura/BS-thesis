# Testing and playground
import numpy as np

nx, ny = (6, 6)
x = np.linspace(1, 5, nx)
print(x, "\n")
y = np.linspace(5, 10, ny)
print(y, "\n")
xv, yv = np.meshgrid(x, y)
tablica = np.meshgrid(x, y)


print(f"This is xv: {xv}", "\n \n", f"This is yv: {yv}")
print("\n")
print(f"This is tablica: {tablica}")

