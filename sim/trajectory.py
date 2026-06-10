import mujoco
import mujoco.viewer
import numpy as np
import mink
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
import imageio


matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # 微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False        # 负号正常显示
# ─────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("mjcf/x5_2025.urdf")
data  = mujoco.MjData(model)
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")

def solve_ik(target_pos, target_quat=np.array([1.0, 0.0, 0.0, 0.0])):
    """给定目标位置，返回关节角解"""
    configuration = mink.Configuration(model)
    tasks = [
        mink.FrameTask(
            frame_name="link6",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1e-3,
        )
    ]
    target_tf = mink.SE3.from_rotation_and_translation(
        mink.SO3(target_quat), target_pos
    )
    tasks[0].set_target(target_tf)
    dt = 0.002
    for _ in range(500):
        vel = mink.solve_ik(configuration, tasks, dt,
                            solver="quadprog", damping=1e-3)
        configuration.integrate_inplace(vel, dt)
    return configuration.q

def get_ee_pos(q):
    """给定关节角，返回末端位置"""
    for i, angle in enumerate(q):
        data.qpos[i] = angle
    mujoco.mj_forward(model, data)
    return data.xpos[ee_id].copy()

# ─────────────────────────────────────────
# 轨迹 1：关节空间点到点（五次多项式插值）
# ─────────────────────────────────────────
print("=== 轨迹1：关节空间点到点 ===")

q_start = np.zeros(6)
q_end   = np.array([0.5, -0.5, 0.8, 0.0, 0.3, 0.0])
n_steps = 100

t = np.linspace(0, 1, n_steps)
# 五次多项式，起止速度和加速度均为0，运动更平滑
s = 10*t**3 - 15*t**4 + 6*t**5

traj_joint = np.outer(s, q_end - q_start) + q_start  # (100, 6)

# 记录末端轨迹
ee_traj1 = np.array([get_ee_pos(q) for q in traj_joint])
print(f"起点末端位置: {np.round(get_ee_pos(q_start), 4)}")
print(f"终点末端位置: {np.round(get_ee_pos(q_end),   4)}")

# ─────────────────────────────────────────
# 轨迹 2：笛卡尔空间直线轨迹
# ─────────────────────────────────────────
print("\n=== 轨迹2：笛卡尔空间直线轨迹 ===")

pos_start = np.array([0.15, -0.1, 0.25])
pos_end   = np.array([0.25,  0.1, 0.35])
n_steps2  = 100

t2 = np.linspace(0, 1, n_steps2)
s2 = 10*t2**3 - 15*t2**4 + 6*t2**5

# 末端位置线性插值
cart_positions = np.outer(1-s2, pos_start) + np.outer(s2, pos_end)

# 每个路径点求IK，得到关节角序列
traj_cart = np.array([solve_ik(pos) for pos in tqdm(cart_positions, desc="求解IK")])
ee_traj2  = np.array([get_ee_pos(q) for q in traj_cart])
print(f"直线起点: {pos_start}")
print(f"直线终点: {pos_end}")
print(f"实际末端误差(终点): "
      f"{np.round(np.linalg.norm(ee_traj2[-1] - pos_end), 6)}")

# 轨迹 3：笛卡尔空间圆弧轨迹
# ─────────────────────────────────────────
print("\n=== 轨迹3：笛卡尔空间圆弧轨迹 ===")

# 圆弧参数：在XY平面内画圆弧
center    = np.array([0.15, 0.0, 0.3])   # 圆心
radius    = 0.08                           # 半径
angle_start = 0                            # 起始角度
angle_end   = np.pi                        # 终止角度（180度圆弧）
n_steps3    = 100

# 生成圆弧上的点
angles = np.linspace(angle_start, angle_end, n_steps3)
# 平滑速度曲线
s3 = 10*(angles/angle_end)**3 - 15*(angles/angle_end)**4 + 6*(angles/angle_end)**5
smooth_angles = angle_start + s3 * (angle_end - angle_start)

arc_positions = np.array([
    center + np.array([radius * np.cos(a), radius * np.sin(a), 0])
    for a in smooth_angles
])

# 每个路径点求IK
traj_arc = np.array([solve_ik(pos) for pos in tqdm(arc_positions, desc="圆弧IK求解")])
ee_traj3 = np.array([get_ee_pos(q) for q in traj_arc])

print(f"圆弧圆心: {center}")
print(f"圆弧半径: {radius}m")
print(f"扫过角度: 180°")
# ─────────────────────────────────────────
# 可视化仿真（先播轨迹1，再播轨迹2）
# ─────────────────────────────────────────
# renderer = mujoco.Renderer(model, height=480, width=640)
# frames = []

# def capture_frame():
#     """截取当前帧"""
#     renderer.update_scene(data, camera=-1)  # -1 表示自由视角
#     return renderer.render().copy()

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     # 播放轨迹1
#     print("\n播放轨迹1：关节空间点到点...")
#     for q in traj_joint:
#         for i, angle in enumerate(q):
#             data.qpos[i] = angle
#         mujoco.mj_forward(model, data)
#         viewer.sync()
#         frames.append(capture_frame())
#         time.sleep(0.01)

#     time.sleep(0.5)

#     # 播放轨迹2
#     print("播放轨迹2：笛卡尔直线轨迹...")
#     for q in traj_cart:
#         for i, angle in enumerate(q):
#             data.qpos[i] = angle
#         mujoco.mj_forward(model, data)
#         viewer.sync()
#         frames.append(capture_frame())
#         time.sleep(0.01)

#     time.sleep(0.5)

#     # 播放轨迹3
#     print("播放轨迹3：笛卡尔圆弧轨迹...")
#     for q in traj_arc:
#         for i, angle in enumerate(q):
#             data.qpos[i] = angle
#         mujoco.mj_forward(model, data)
#         viewer.sync()
#         frames.append(capture_frame())
#         time.sleep(0.01)

# # 保存视频
# print("保存视频...")
# imageio.mimsave('results/simulation.mp4', frames, fps=30)
# print("视频已保存到 results/simulation.mp4")

# ─────────────────────────────────────────
# 画图
# 在原来4个子图后面加第5、6个
fig = plt.figure(figsize=(18, 10))

# 图1：轨迹1关节角
# ax1 = fig.add_subplot(231)
# for i in range(6):
#     ax1.plot(traj_joint[:, i], label=f'关节{i+1}')
# ax1.set_title('轨迹1：关节角度变化')
# ax1.set_xlabel('步数'); ax1.set_ylabel('角度 (rad)')
# ax1.legend(fontsize=7); ax1.grid(True)

# # 图2：轨迹1末端路径
# ax2 = fig.add_subplot(232, projection='3d')
# ax2.plot(ee_traj1[:,0], ee_traj1[:,1], ee_traj1[:,2], 'b-', linewidth=2)
# ax2.scatter(*ee_traj1[0],  c='g', s=50, label='起点')
# ax2.scatter(*ee_traj1[-1], c='r', s=50, label='终点')
# ax2.set_title('轨迹1：末端3D路径')
# ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
# ax2.legend()

# 图3：轨迹2关节角
ax3 = fig.add_subplot(233)
for i in range(6):
    ax3.plot(traj_cart[:, i], label=f'关节{i+1}')
ax3.set_title('轨迹2：笛卡尔直线关节角')
ax3.set_xlabel('步数'); ax3.set_ylabel('角度 (rad)')
ax3.legend(fontsize=7); ax3.grid(True)

# 图4：轨迹2末端路径
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot(ee_traj2[:,0], ee_traj2[:,1], ee_traj2[:,2], 'r-', linewidth=2)
ax4.scatter(*ee_traj2[0],  c='g', s=50, label='起点')
ax4.scatter(*ee_traj2[-1], c='r', s=50, label='终点')
ax4.set_title('轨迹2：笛卡尔直线末端路径')
ax4.set_xlabel('X'); ax4.set_ylabel('Y'); ax4.set_zlabel('Z')
ax4.legend()

# # 图5：轨迹3关节角
# ax5 = fig.add_subplot(235)
# for i in range(6):
#     ax5.plot(traj_arc[:, i], label=f'关节{i+1}')
# ax5.set_title('轨迹3：圆弧轨迹关节角')
# ax5.set_xlabel('步数'); ax5.set_ylabel('角度 (rad)')
# ax5.legend(fontsize=7); ax5.grid(True)

# # 图6：轨迹3末端路径
# ax6 = fig.add_subplot(236, projection='3d')
# ax6.plot(ee_traj3[:,0], ee_traj3[:,1], ee_traj3[:,2], 'm-', linewidth=2)
# ax6.scatter(*ee_traj3[0],  c='g', s=50, label='起点')
# ax6.scatter(*ee_traj3[-1], c='r', s=50, label='终点')
# ax6.set_title('轨迹3：圆弧末端路径')
# ax6.set_xlabel('X'); ax6.set_ylabel('Y'); ax6.set_zlabel('Z')
# ax6.legend()

plt.tight_layout()
plt.savefig('results/trajectory_results.png', dpi=150)
plt.show()
print("图表已保存到 results/trajectory_results.png")
