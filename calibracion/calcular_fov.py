import numpy as np


W = 640
H = 480


fx = 840.32
fy = 832.56


fov_horizontal_rad = 2 * np.arctan(W / (2 * fx))
fov_vertical_rad = 2 * np.arctan(H / (2 * fy))

fov_horizontal_deg = np.degrees(fov_horizontal_rad)
fov_vertical_deg = np.degrees(fov_vertical_rad)

print(f"FOV Horizontal: {fov_horizontal_deg:.2f}º")
print(f"FOV Vertical: {fov_vertical_deg:.2f}º")