import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import mujoco
import mujoco.viewer
import time

# ─────────────────────────────────────────
# FK / Jacobian tools
# ─────────────────────────────────────────
def Rx(a):
    return np.array([[1,0,0,0],[0,np.cos(a),-np.sin(a),0],[0,np.sin(a),np.cos(a),0],[0,0,0,1]],dtype=float)
def Ry(a):
    return np.array([[np.cos(a),0,np.sin(a),0],[0,1,0,0],[-np.sin(a),0,np.cos(a),0],[0,0,0,1]],dtype=float)
def Rz(a):
    return np.array([[np.cos(a),-np.sin(a),0,0],[np.sin(a),np.cos(a),0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
def Trans(x,y,z):
    return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]],dtype=float)

def fk_all(q):
    t1,t2,t3,t4,t5,t6 = q
    T0 = np.eye(4)
    T1 = T0 @ Trans(0,0,0.0603) @ Rz(t1)
    T2 = T1 @ Trans(0.02,0,0.0402) @ Ry(t2)
    T3 = T2 @ Trans(-0.264,0,0) @ Rx(-np.pi) @ Ry(t3)
    T4 = T3 @ Trans(0.245,0,-0.056) @ Ry(t4)
    T5 = T4 @ Trans(0.06575,-0.001,-0.0825) @ Rz(t5)
    T6 = T5 @ Trans(0.02845,0,0.0825) @ Rx(-np.pi) @ Rx(t6)
    return [T0,T1,T2,T3,T4,T5,T6]

def get_axis(T, joint_idx):
    axes = {1:[0,0,1],2:[0,1,0],3:[0,1,0],4:[0,1,0],5:[0,0,1],6:[1,0,0]}
    return T[:3,:3] @ np.array(axes[joint_idx],dtype=float)

def jacobian(q):
    Ts = fk_all(q)
    pe = Ts[-1][:3,3]
    J = np.zeros((6,6))
    for i in range(6):
        pi = Ts[i][:3,3]
        zi = get_axis(Ts[i], i+1)
        J[:3,i] = np.cross(zi, pe-pi)
        J[3:,i] = zi
    return J

def manipulability(q):
    J = jacobian(q)
    sv = np.linalg.svd(J, compute_uv=False)
    return sv[-1], sv[0]/sv[-1]   # min_sv, condition_number

# ═══════════════════════════════════════════════════════════════
# 理论奇异条件
# ═══════════════════════════════════════════════════════════════
#
# 1. 肩部奇异 (Shoulder Singularity)
#    条件：腕部中心(joint5原点)落在joint1的Z轴上，即 p_wx=p_wy=0
#    物理意义：joint1绕Z轴旋转时末端不动，theta1不定
#    几何条件：sqrt(p5x^2 + p5y^2) = 0，其中p5是joint5原点的xy坐标
#
# 2. 肘部奇异 (Elbow Singularity)  
#    条件：joint2、joint3、joint4三轴平行，且两连杆完全共线
#    由于joint3处Rx(-pi)翻转，三轴实际方向为[0,1,0],[0,-1,0],[0,-1,0]
#    始终平行，所以肘部奇异取决于位置：
#    连杆2和连杆3在joint2的旋转平面内完全对齐（伸展极限）
#    几何条件：|joint2到joint4| = l1+l2（完全伸展）或 |l1-l2|（完全折叠）
#    等价条件：行列式(J_pos子矩阵列2,3,4)=0
#    搜索发现在 theta2≈-1.349, theta3≈-0.428 附近奇异值≈0（条件数>4e6）
#
# 3. 腕部奇异 (Wrist Singularity)
#    ARX5腕部轴序：joint4(Y) - joint5(Z) - joint6(X)
#    这不是标准球腕，但仍有奇异：
#    当joint4的Y轴与joint6的X轴共线（平行或反平行）时，
#    joint5的Z轴旋转不能区分joint4和joint6的贡献
#    几何条件：|joint4轴 × joint6轴| = 0
#    即 [0,-1,0] × R_z(theta5)*[1,0,0] = 0
#    展开：cross([0,-1,0], [cos(t5), sin(t5), 0]) = [0, 0, cos(t5)]
#    因此腕部奇异条件：cos(theta5) = 0，即 theta5 = ±π/2

print("=" * 60)
print("ARX5 奇异条件理论推导")
print("=" * 60)
print()
print("1. 肩部奇异：腕部中心落在joint1的Z轴上")
print("   条件：sqrt(p5x²+p5y²) = 0")
print()
print("2. 肘部奇异：连杆完全伸展/折叠")
print("   搜索最优奇异构型：theta2≈-1.349, theta3≈-0.428")
q_elbow = np.array([-1.349, -0.428, 0, 0, 0, 0])
sv_min, cond = manipulability(q_elbow)
print(f"   验证：q={np.round(q_elbow,3)}")
print(f"   最小奇异值={sv_min:.6f}, 条件数={cond:.1f}")
print()
print("3. 腕部奇异：cos(theta5) = 0，即 theta5 = ±π/2")
q_wrist = np.array([0, 0, 0, 0, np.pi/2, 0])
sv_min, cond = manipulability(q_wrist)
print(f"   验证：theta5=π/2, 最小奇异值={sv_min:.6f}, 条件数={cond:.1f}")
q_wrist2 = np.array([0, 0, 0, 0, -np.pi/2, 0])
sv_min2, cond2 = manipulability(q_wrist2)
print(f"   验证：theta5=-π/2, 最小奇异值={sv_min2:.6f}, 条件数={cond2:.1f}")

# ═══════════════════════════════════════════════════════════════
# 图1：肘部奇异 - 扫描theta2，固定theta3，看最小奇异值曲线
# ═══════════════════════════════════════════════════════════════
print("\n生成图1：肘部奇异特征曲线...")
t2_range = np.linspace(-np.pi/2, np.pi/2, 300)
sv_elbow = []
cond_elbow = []
for t2 in t2_range:
    q = np.array([0, t2, -0.428, 0, 0, 0])
    sv, cn = manipulability(q)
    sv_elbow.append(sv)
    cond_elbow.append(min(cn, 1e5))

# ═══════════════════════════════════════════════════════════════
# 图2：腕部奇异 - 扫描theta5，看最小奇异值曲线
# ═══════════════════════════════════════════════════════════════
print("生成图2：腕部奇异特征曲线...")
t5_range = np.linspace(-np.pi, np.pi, 400)
sv_wrist = []
cond_wrist = []
for t5 in t5_range:
    q = np.array([0, 0.3, 0.3, 0, t5, 0])
    sv, cn = manipulability(q)
    sv_wrist.append(sv)
    cond_wrist.append(min(cn, 1e5))

# ═══════════════════════════════════════════════════════════════
# 图3：工作空间奇异度云图（theta2-theta3平面）
# ═══════════════════════════════════════════════════════════════
print("生成图3：奇异度云图...")
t2_grid = np.linspace(-np.pi/2, np.pi/2, 80)
t3_grid = np.linspace(-np.pi/2, np.pi/2, 80)
T2G, T3G = np.meshgrid(t2_grid, t3_grid)
SV_MAP = np.zeros_like(T2G)
for i in range(len(t2_grid)):
    for j in range(len(t3_grid)):
        q = np.array([0, T2G[j,i], T3G[j,i], 0, 0, 0])
        sv, _ = manipulability(q)
        SV_MAP[j,i] = sv

# ═══════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#1a1a2e')

# --- 子图1：肘部奇异曲线 ---
ax1 = fig.add_subplot(221)
ax1.set_facecolor('#16213e')
ax1.plot(np.degrees(t2_range), sv_elbow, color='#e94560', linewidth=2, label='Min Singular Value')
ax1.axvline(np.degrees(-1.349), color='#ffd700', linestyle='--', linewidth=1.5, label=f'Singular point θ₂≈{np.degrees(-1.349):.1f}°')
ax1.axhline(0, color='white', linewidth=0.5, alpha=0.3)
ax1.set_xlabel('θ₂ (degrees)', color='white')
ax1.set_ylabel('Min Singular Value σ_min', color='white')
ax1.set_title('Elbow Singularity\n(θ₃=-0.428 rad fixed)', color='white', fontweight='bold')
ax1.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#404060')
ax1.set_ylim(bottom=0)
# 奇异区高亮
idx_sing = np.argmin(sv_elbow)
ax1.fill_between(np.degrees(t2_range), sv_elbow,
                  where=[abs(t-t2_range[idx_sing])<0.3 for t in t2_range],
                  color='#e94560', alpha=0.3, label='Singular region')
ax1.grid(True, alpha=0.2, color='#404060')

# --- 子图2：腕部奇异曲线 ---
ax2 = fig.add_subplot(222)
ax2.set_facecolor('#16213e')
ax2.plot(np.degrees(t5_range), sv_wrist, color='#00d4ff', linewidth=2, label='Min Singular Value')
ax2.axvline(90, color='#ffd700', linestyle='--', linewidth=1.5, label='θ₅=+90° (singular)')
ax2.axvline(-90, color='#ffd700', linestyle='--', linewidth=1.5, label='θ₅=-90° (singular)')
ax2.axhline(0, color='white', linewidth=0.5, alpha=0.3)
ax2.set_xlabel('θ₅ (degrees)', color='white')
ax2.set_ylabel('Min Singular Value σ_min', color='white')
ax2.set_title('Wrist Singularity\nCondition: cos(θ₅) = 0', color='white', fontweight='bold')
ax2.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#404060')
ax2.grid(True, alpha=0.2, color='#404060')

# --- 子图3：奇异度云图 ---
ax3 = fig.add_subplot(223)
ax3.set_facecolor('#16213e')
im = ax3.contourf(np.degrees(T2G), np.degrees(T3G), SV_MAP,
                   levels=30, cmap='RdYlGn')
cb = plt.colorbar(im, ax=ax3)
cb.set_label('σ_min (green=safe, red=singular)', color='white')
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
# 标出最小奇异值点
min_idx = np.unravel_index(np.argmin(SV_MAP), SV_MAP.shape)
ax3.plot(np.degrees(T2G[min_idx]), np.degrees(T3G[min_idx]),
         'r*', markersize=15, label=f'Most singular\nθ₂={np.degrees(T2G[min_idx]):.1f}°,θ₃={np.degrees(T3G[min_idx]):.1f}°')
ax3.set_xlabel('θ₂ (degrees)', color='white')
ax3.set_ylabel('θ₃ (degrees)', color='white')
ax3.set_title('Singularity Map\n(θ₂-θ₃ plane, σ_min)', color='white', fontweight='bold')
ax3.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)
ax3.tick_params(colors='white')
ax3.spines[:].set_color('#404060')

# --- 子图4：三种奇异的条件数对比 ---
ax4 = fig.add_subplot(224)
ax4.set_facecolor('#16213e')

# 逐渐趋近腕部奇异
t5_approach = np.linspace(0, np.pi/2 - 0.01, 100)
cond_approach = []
for t5 in t5_approach:
    q = np.array([0, 0.3, 0.3, 0, t5, 0])
    sv, cn = manipulability(q)
    cond_approach.append(min(cn, 1e5))

ax4.semilogy(np.degrees(t5_approach), cond_approach,
             color='#a855f7', linewidth=2, label='Wrist: cond(J) vs θ₅')
ax4.axvline(90, color='#ffd700', linestyle='--', linewidth=1.5, label='θ₅=90° (singular)')
ax4.axhline(1e3, color='#ff6b6b', linestyle=':', linewidth=1, alpha=0.7, label='Warning threshold (1000)')
ax4.set_xlabel('θ₅ (degrees)', color='white')
ax4.set_ylabel('Condition Number κ(J) [log]', color='white')
ax4.set_title('Condition Number Near\nWrist Singularity', color='white', fontweight='bold')
ax4.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)
ax4.tick_params(colors='white')
ax4.spines[:].set_color('#404060')
ax4.grid(True, alpha=0.2, color='#404060')

plt.suptitle('ARX5 Six-Axis Robot - Singularity Analysis',
             color='white', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/singularity_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("图表已保存")

# ═══════════════════════════════════════════════════════════════
# MuJoCo 仿真：展示腕部奇异构型
# ═══════════════════════════════════════════════════════════════
print("\n启动MuJoCo仿真，展示腕部奇异构型 (theta5=pi/2)...")
model = mujoco.MjModel.from_xml_path("mjcf/x5_2025.urdf")
data  = mujoco.MjData(model)

# 腕部奇异构型：theta5=pi/2
q_singular_wrist = np.array([0.3, 0.2, 0.3, 0, np.pi/2, 0])
# 正常构型对比
q_normal = np.array([0.3, 0.2, 0.3, 0, 0.3, 0])

frames = []
renderer = mujoco.Renderer(model, height=480, width=640)

def set_pose(q):
    for i, angle in enumerate(q):
        data.qpos[i] = angle
    mujoco.mj_forward(model, data)

def capture(azimuth=45, elevation=-20):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[0] = 0.15
    cam.lookat[1] = 0.0
    cam.lookat[2] = 0.25
    cam.distance  = 0.9
    cam.azimuth   = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()

import imageio

frames_sim = []

# 从正常构型过渡到奇异构型
n = 80
for i in range(n):
    t = i / (n-1)
    s = 3*t**2 - 2*t**3
    q_interp = q_normal + s*(q_singular_wrist - q_normal)
    set_pose(q_interp)
    frames_sim.append(capture())

# 在奇异位形停留，旋转视角
for az in np.linspace(45, 135, 40):
    set_pose(q_singular_wrist)
    frames_sim.append(capture(azimuth=az))

# 奇异构型下旋转theta4，展示theta4与theta6不可区分
for t4 in np.linspace(0, 1.0, 40):
    q_test = q_singular_wrist.copy()
    q_test[3] = t4
    set_pose(q_test)
    frames_sim.append(capture(azimuth=100))

# 回到正常
for i in range(n):
    t = i / (n-1)
    s = 3*t**2 - 2*t**3
    q_start = q_singular_wrist.copy(); q_start[3] = 1.0
    q_interp = q_start + s*(q_normal - q_start)
    set_pose(q_interp)
    frames_sim.append(capture())

imageio.mimsave('results/singularity_demo.mp4', frames_sim, fps=30)
print("仿真视频已保存：singularity_demo.mp4")

# 保存奇异构型截图
set_pose(q_singular_wrist)
img_sing = capture(azimuth=60, elevation=-25)
set_pose(q_normal)
img_norm = capture(azimuth=60, elevation=-25)

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor('#1a1a2e')
for ax, img, title, sv_info in zip(axes,
    [img_norm, img_sing],
    ['Normal Configuration\nθ₅=0.3 rad', 'Wrist Singularity\nθ₅=π/2≈1.571 rad'],
    [manipulability(q_normal), manipulability(q_singular_wrist)]):
    ax.imshow(img)
    ax.set_title(f'{title}\nσ_min={sv_info[0]:.4f}  κ={sv_info[1]:.1f}',
                 color='white', fontweight='bold')
    ax.axis('off')
    color = '#ff4444' if sv_info[1] > 500 else '#44ff88'
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(3); spine.set_visible(True)
plt.suptitle('ARX5 Wrist Singularity: Before vs At Singular Configuration',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/singularity_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("对比截图已保存：singularity_comparison.png")
print("\n完成！输出文件：")
print("  1. singularity_analysis.png  - 四格特征曲线图")
print("  2. singularity_demo.mp4      - 奇异过渡仿真视频")
print("  3. singularity_comparison.png- 正常 vs 奇异构型对比")