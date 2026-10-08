import numpy as np
T = np.load('hardware/transform_cam_to_world.npy')
T[:3, 3] /= 1000.0   # мм → метры
np.save('hardware/transform_cam_to_world.npy', T)
print(T)