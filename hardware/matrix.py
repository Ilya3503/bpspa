import numpy as np
import os

# ==================== ТВОЯ МАТРИЦА ====================
transform_cam_to_world = np.array([
    [ 0.999344422,   0.0324113077,  0.0161317192,  -0.00113266],
    [ 0.0200134063, -0.865870773,   0.499867251,  -0.387638],
    [ 0.0301693355, -0.499216698,  -0.865951788,   0.656133],
    [ 0.0,           0.0,           0.0,             1.0       ]
], dtype=np.float32)
# =====================================================

# Сохраняем в ту же папку, где лежит этот скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "transform_cam_to_world.npy")

np.save(save_path, transform_cam_to_world)

print(f"✅ Матрица успешно сохранена:")
print(save_path)