# 四旋翼NMPC控制器 20250304 Wakkk
# ACADOS NMPC
from acados_template import AcadosOcp, AcadosOcpSolver
from export_model import *
import numpy as np
import scipy.linalg
from os.path import dirname, join, abspath
import time
import matplotlib.pyplot as plt
from casadi import integrator

# np.set_printoptions(precision=3)  # 设置精度
np.set_printoptions(suppress=True)  # 禁用科学计数法输出

# ACADOS NMPC控制器
class NMPC_Controller:
    def __init__(self):
        self.ocp = AcadosOcp()       # OCP 优化问题
        self.model = export_model()  # 导出四旋翼物理模型

        self.Tf = 0.75                     # 预测时间长度(s)
        self.N = 50                         # 预测步数(节点数量)
        self.nx = self.model.x.size()[0]    # 状态维度 10维度
        self.nu = self.model.u.size()[0]    # 控制输入维度 4维度
        self.ny = self.nx + self.nu         # 评估维度
        self.ny_e = self.nx                 # 终端评估维度

        # set ocp_nlp_dimensions
        self.nlp_dims     = self.ocp.dims
        self.nlp_dims.N   = self.N

        # parameters
        self.g0  = 9.8066    # [m.s^2] accerelation of gravity
        self.mq  = 2.064     # [kg] total mass (with one marker)
        self.Ct  = 0.093746   # [N/krpm^2] Thrust coef

        # --- 关键修改：计算悬停推力 (N) 而非转速 ---
        self.hov_thrust = self.mq * self.g0  
        print(f"Hover total thrust: {self.hov_thrust:.3f} N")

        # set weighting matrices 状态权重矩阵
        Q = np.eye(self.nx)
        Q[0,0] = 100.0      # x
        Q[1,1] = 100.0      # y
        Q[2,2] = 200.0      # z
        Q[3,3] = 0.0        # qw
        Q[4,4] = 0.0        # qx
        Q[5,5] = 0.0        # qy
        Q[6,6] = 0.0        # qz
        Q[7,7] = 1.0        # vbx
        Q[8,8] = 1.0        # vby
        Q[9,9] = 4.0        # vbz

        R = np.eye(self.nu)   # 控制输入权重矩阵
        R[0,0] = 0.01    # thrust
        R[1,1] = 0.1    # wx
        R[2,2] = 0.1    # wy
        R[3,3] = 0.1    # wz

        self.ocp.cost.W = scipy.linalg.block_diag(Q, R)

        Vx = np.zeros((self.ny, self.nx))
        Vx[0,0] = 1.0
        Vx[1,1] = 1.0
        Vx[2,2] = 1.0
        Vx[3,3] = 1.0
        Vx[4,4] = 1.0
        Vx[5,5] = 1.0
        Vx[6,6] = 1.0
        Vx[7,7] = 1.0
        Vx[8,8] = 1.0
        Vx[9,9] = 1.0
        self.ocp.cost.Vx = Vx

        Vu = np.zeros((self.ny, self.nu))
        Vu[10,0] = 1.0
        Vu[11,1] = 1.0
        Vu[12,2] = 1.0
        Vu[13,3] = 1.0
        self.ocp.cost.Vu = Vu

        self.ocp.cost.W_e = 50.0 * Q

        Vx_e = np.zeros((self.ny_e, self.nx))
        Vx_e[0,0] = 1.0
        Vx_e[1,1] = 1.0
        Vx_e[2,2] = 1.0
        Vx_e[3,3] = 1.0
        Vx_e[4,4] = 1.0
        Vx_e[5,5] = 1.0
        Vx_e[6,6] = 1.0
        Vx_e[7,7] = 1.0
        Vx_e[8,8] = 1.0
        Vx_e[9,9] = 1.0
        self.ocp.cost.Vx_e = Vx_e

        # 过程参考向量(状态+输入)
        self.ocp.cost.yref   = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.hov_thrust, 0.0, 0.0, 0.0])
        # 终端参考向量(状态)
        self.ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0])

        # 构建约束
        # 假设最大推力为电机最大合力 (~34N)，最大角速度为 0.8 rad/s
        max_rate = 0.8
        max_thrust = 34.19432
        min_thrust = 0.0
        self.ocp.constraints.lbu = np.array([min_thrust, -max_rate, -max_rate, -max_rate])  # 电机最低转速输入
        self.ocp.constraints.ubu = np.array([max_thrust, max_rate, max_rate, max_rate])  # 电机最高转速输入
        self.ocp.constraints.x0  = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 初始状态
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3])  # 控制向量中的第 0, 1, 2, 3 个元素都需要受到数值范围的限制

        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0

        # set prediction horizon
        self.ocp.solver_options.tf = self.Tf
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # 显然更快 ~100Hz
        # self.ocp.solver_options.nlp_solver_type = 'SQP'  # ~10Hz

        self.ocp.model = self.model  # 传入模型
        # 构建编译OCP求解器
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file = 'acados_ocp.json')
        print("NMPC Controller Init Done")

    # 状态空间位点控制
    # current_state当前状态: [x, y, z, qw, qx, qy, qz, vbx, vby, vbz] 
    # goal_state目标状态:    [x, y, z, qw, qx, qy, qz, vbx, vby, vbz] 
    def nmpc_state_control(self, current_state, goal_state):
        _start = time.perf_counter()
        # Set initial condition, equality constraint
        self.acados_solver.set(0, 'lbx', current_state)
        self.acados_solver.set(0, 'ubx', current_state)

        # 构建参考向量：目标状态 + 悬停推力 + 0角速度
        y_ref = np.concatenate((goal_state, np.array([self.hov_thrust, 0, 0, 0])))
        # Set Goal State
        for i in range(self.N):
            self.acados_solver.set(i, 'yref', y_ref)   # 过程参考
        y_refN = goal_state 
        self.acados_solver.set(self.N, 'yref', y_refN)   # 终端参考

        # Solve Problem
        self.acados_solver.solve()
        # Get Solution
        w_opt_acados = np.ndarray((self.N, 4))  # 控制输入
        x_opt_acados = np.ndarray((self.N + 1, len(current_state)))   # 状态估计
        x_opt_acados[0, :] = self.acados_solver.get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_solver.get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_solver.get(i + 1, "x")
        # return w_opt_acados, x_opt_acados  # 返回控制输入和状态
        _end = time.perf_counter()
        _dt = _end - _start
        return _dt, w_opt_acados[0]  # 返回最近控制输入 4 Vector
        # control_input = self.acados_solver.get(0, "u")
        # state_estimate = self.acados_solver.get(self.N, "x")
        # return control_input, state_estimate  # 返回所有控制输入和状态

    # NMPC位置控制
    # goal_pos: 目标三维位置[x y z]
    def nmpc_position_control(self, current_state, goal_pos):
        goal_state = np.array([goal_pos[0], goal_pos[1], goal_pos[2], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _dt, control = self.nmpc_state_control(current_state, goal_state)
        return _dt, control
    
    # 轨迹跟踪控制
    def nmpc_trajectory_tracking(self, current_state, t_now):
        # 1. 设置当前状态作为初始约束
        self.acados_solver.set(0, 'lbx', current_state)
        self.acados_solver.set(0, 'ubx', current_state)

        # 2. 预测窗口参数
        dt = self.Tf / self.N  # 每一小步的时间间隔
        
        # 轨迹参数 (可根据需要修改)
        R = 1.0        # 半径 1米
        omega = 1.0    # 角速度 1 rad/s
        h = 1.0        # 高度 1米

        # 3. 为预测窗口内的每个节点计算参考轨迹
        for i in range(self.N):
            t_predict = t_now + i * dt
            
            # 计算该时刻的状态参考
            px_ref = R * np.cos(omega * t_predict)
            py_ref = R * np.sin(omega * t_predict)
            pz_ref = h
            vx_ref = -R * omega * np.sin(omega * t_predict)
            vy_ref = R * omega * np.cos(omega * t_predict)
            vz_ref = 0.0
            
            # 拼接 yref: [pos(3), quat(4), vel(3), thrust(1), rates(3)]
            yref = np.array([
                px_ref, py_ref, pz_ref,      # 位置
                1.0, 0.0, 0.0, 0.0,          # 姿态
                vx_ref, vy_ref, vz_ref,      # 速度
                self.hov_thrust, 0, 0, 0     # 控制输入参考
            ])
            self.acados_solver.set(i, "yref", yref)

        # 4. 设置终端参考 (Terminal cost)
        t_terminal = t_now + self.Tf
        yref_e = np.array([
            R * np.cos(omega * t_terminal), 
            R * np.sin(omega * t_terminal), 
            h,
            1.0, 0.0, 0.0, 0.0,
            -R * omega * np.sin(omega * t_terminal), 
            R * omega * np.cos(omega * t_terminal), 
            0.0
        ])
        self.acados_solver.set(self.N, "yref", yref_e)

        # 5. 求解
        status = self.acados_solver.solve()
        control = self.acados_solver.get(0, "u")
        return control

# TEST
if __name__ == '__main__':
    print("20250304 ACADOS NMPC TEST")
    nmpc_controller = NMPC_Controller()
    model = nmpc_controller.model
    current_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sim_time = 10.0  # 仿真10秒
    dt_sim = 0.02    # 仿真步长 50Hz
    steps = int(sim_time / dt_sim)

    history_pos = []
    print("Starting Circle Simulation...")
    for i in range(steps):
        t_now = i * dt_sim
        
        # 获取控制指令
        u = nmpc_controller.nmpc_trajectory_tracking(current_state, t_now)
        # 创建积分器 (使用 Runge-Kutta 4阶方法)
        dae = {'x': model.x, 'p': model.u, 'ode': model.f_expl_expr}
        opts = {'tf': dt_sim}
        I = integrator('I', 'rk', dae, opts)
        # 计算结果
        res = I(x0=current_state, p=u)
        x_next = res['xf'].full().flatten()
        current_state = x_next
        print(f"下一时刻位置: {x_next[0:3]}")
        
        # 记录位置
        history_pos.append(current_state[0:3].copy())
        # 在仿真循环中打印
        print(f"--- Step {i+1} ---")
        print(f"Pos: {current_state[0]:.3f}, {current_state[1]:.3f}, {current_state[2]:.3f}")
        print(f"Quat: {current_state[3]:.3f}, {current_state[4]:.3f}, {current_state[5]:.3f}, {current_state[6]:.3f} (qw, qx, qy, qz)")
        print(f"Vel: {current_state[7]:.3f}, {current_state[8]:.3f}, {current_state[9]:.3f}")
        print(f"Control: T={u[0]:.3f} N, wx={u[1]:.3f}, wy={u[2]:.3f}, wz={u[3]:.3f}")

    # 可视化
    history_pos = np.array(history_pos)
    plt.figure(figsize=(8,8))
    plt.plot(history_pos[:,0], history_pos[:,1], label='NMPC Path')
    plt.title("NMPC Circle Tracking Test (XY Plane)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.legend()
    plt.show()


