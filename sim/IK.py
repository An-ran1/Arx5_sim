import mujoco
import mujoco.viewer
import numpy as np
import mink

# 加载模型
model = mujoco.MjModel.from_xml_path("mjcf/x5_2025.urdf")
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# 初始化 mink 配置
configuration = mink.Configuration(model)

# 定义末端任务
tasks = [
    mink.FrameTask(
        frame_name="link6",
        frame_type="body",
        position_cost=1.0,       # 位置权重
        orientation_cost=0.1,    # 姿态权重
        lm_damping=1e-3,
    )
]

# 设置目标位姿（修改这里来测试不同目标点）
target_pos  = np.array([0.2, 0.1, 0.3])
target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z] 单位四元数

target_tf = mink.SE3.from_rotation_and_translation(
    mink.SO3(target_quat), target_pos
)
tasks[0].set_target(target_tf)

# 迭代求解 IK
dt = 0.002
for _ in range(1000):
    vel = mink.solve_ik(configuration, tasks, dt, solver="quadprog", damping=1e-3)
    configuration.integrate_inplace(vel, dt)

# 输出结果
q_sol = configuration.q
print("=== 逆运动学结果 ===")
print(f"目标位置: {target_pos}")
print(f"关节角解: {np.round(q_sol, 4)}")

# 验证：把解出的关节角代入正运动学，看末端位置是否接近目标
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
for i, angle in enumerate(q_sol):
    data.qpos[i] = angle
mujoco.mj_forward(model, data)
pos_check = data.xpos[ee_id]
print(f"验证末端位置: {np.round(pos_check, 4)}")
print(f"位置误差:     {np.round(np.linalg.norm(pos_check - target_pos), 6)}")

# 可视化结果
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()