# 导出四旋翼物理模型 20250304 Wakkk
from acados_template import AcadosModel
from casadi import SX, vertcat

# 将 NMPC 的控制量从四个电机转速切换为三轴角速度 + 总推力
def export_model():
    model_name = 'crazyflie'
    # parameters
    g0  = 9.8066     # [m.s^2] accerelation of gravity
    mass  = 2.064      # [kg] total mass (with one marker)

    # 世界坐标系位置
    px = SX.sym('px')
    py = SX.sym('py')
    pz = SX.sym('pz')
    # 四元数
    q0 = SX.sym('q0')
    q1 = SX.sym('q1')
    q2 = SX.sym('q2')
    q3 = SX.sym('q3')
    # 世界坐标系速度
    vx = SX.sym('vx')
    vy = SX.sym('vy')
    vz = SX.sym('vz')
    # 构建状态向量
    # --- 1. 定义状态量 (缩减为 10 维) ---
    x = vertcat(px, py, pz, q0, q1, q2, q3, vx, vy, vz)

    # --- 2. 定义控制量 (总推力 + 三轴角速度) ---
    thrust = SX.sym('thrust') # 总推力 (牛顿 N)
    wx = SX.sym('wx')         # 机体系角速度
    wy = SX.sym('wy')
    wz = SX.sym('wz')
    u = vertcat(thrust, wx, wy, wz)

    # for f_impl
    px_dot = SX.sym('px_dot')
    py_dot = SX.sym('py_dot')
    pz_dot = SX.sym('pz_dot')
    q0_dot = SX.sym('q0_dot')
    q1_dot = SX.sym('q1_dot')
    q2_dot = SX.sym('q2_dot')
    q3_dot = SX.sym('q3_dot')
    vx_dot = SX.sym('vx_dot')
    vy_dot = SX.sym('vy_dot')
    vz_dot = SX.sym('vz_dot')

    # 构建导数状态向量
    xdot = vertcat(px_dot, py_dot, pz_dot, q0_dot, q1_dot, q2_dot, q3_dot, vx_dot, vy_dot, vz_dot)
    # 位置求导
    px_d = vx
    py_d = vy
    pz_d = vz

    # 速度求导 (总推力除以质量得到加速度)
    _thrust_acc_b = thrust / mass  # 机体坐标系中推力引起的加速度
    # 将机体坐标系推力加速度转换为世界坐标系推力加速度
    # Rwb * [0, 0, _thrust_acc_b]
    _thrust_accx_w = 2*(q1*q3+q0*q2)*_thrust_acc_b
    _thrust_accy_w = 2*(-q0*q1+q2*q3)*_thrust_acc_b
    _thrust_accz_w = 2*(0.5-q1**2-q2**2)*_thrust_acc_b
    vx_d = _thrust_accx_w
    vy_d = _thrust_accy_w
    vz_d = _thrust_accz_w - g0  # 重力加速度

    # 四元数求导
    q0_d = -(q1*wx)/2 - (q2*wy)/2 - (q3*wz)/2
    q1_d =  (q0*wx)/2 - (q3*wy)/2 + (q2*wz)/2
    q2_d =  (q3*wx)/2 + (q0*wy)/2 - (q1*wz)/2
    q3_d =  (q1*wy)/2 - (q2*wx)/2 + (q0*wz)/2

    # Explicit and Implicit functions
    # 构建显式表达式和隐式表达式
    f_expl = vertcat(px_d, py_d, pz_d, q0_d, q1_d, q2_d, q3_d, vx_d, vy_d, vz_d)
    f_impl = xdot - f_expl

    # algebraic variables
    z = []
    # parameters
    p = []
    # dynamics
    model = AcadosModel()  # 新建ACADOS模型

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name

    return model
