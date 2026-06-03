import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mjcf/x5_2025.urdf")
data  = mujoco.MjData(model)

# 末端body id
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")

# 设置6个关节角度（单位：弧度）
q = [0, 0, 0, 0, 0, 0]   # 零位
for i, angle in enumerate(q):
    data.qpos[i] = angle

# 更新所有位置
mujoco.mj_forward(model, data)

# 读取末端位置和姿态
pos = data.xpos[ee_id]
rot = data.xmat[ee_id].reshape(3, 3)

print("=== 正运动学结果 ===")
print(f"末端位置 [x, y, z]:\n  {pos}")
print(f"末端旋转矩阵:\n{rot}")

# 转换为齐次变换矩阵
T = np.eye(4)
T[:3, :3] = rot
T[:3,  3] = pos
print(f"\n齐次变换矩阵 T:\n{np.round(T, 4)}")

# 测试几组关节角
print("\n=== 不同关节角测试 ===")
test_configs = [
    [0, 0, 0, 0, 0, 0],
    [0.5, -0.5, 0.8, 0, 0.3, 0],
    [1.0,  0.0, 1.0, 0, 0.5, 0],
]
for q in test_configs:
    for i, angle in enumerate(q):
        data.qpos[i] = angle
    mujoco.mj_forward(model, data)
    pos = data.xpos[ee_id]
    print(f"q={np.round(q,2)} → 末端位置: {np.round(pos,4)}")