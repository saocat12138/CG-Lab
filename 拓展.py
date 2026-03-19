import taichi as ti
import math

# 初始化 Taichi，指定使用 CPU 后端
ti.init(arch=ti.cpu)

# 声明 Taichi 的 Field：存储立方体顶点和转换后的屏幕坐标
# 立方体8个顶点，12条边
cube_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

# 定义立方体的12条边（顶点索引对）
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 前平面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 后平面
    (0, 4), (1, 5), (2, 6), (3, 7)   # 连接前后平面的边
]

@ti.func
def get_model_matrix(angle: ti.f32):
    """
    模型变换矩阵：绕 Z 轴旋转
    """
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    """
    视图变换矩阵：将相机移动到原点
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """
    透视投影矩阵
    """
    # 视线看向 -Z 轴，实际坐标为负
    n = -zNear
    f = -zFar
    
    # 视角转化为弧度并求出 t, b, r, l
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    # 1. 挤压矩阵: 透视平截头体 -> 长方体
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # 2. 正交投影矩阵: 缩放与平移至 [-1, 1]^3
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    
    # 返回组合矩阵
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle: ti.f32):
    """
    在并行架构上计算立方体顶点的坐标变换
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])  # 相机位置在Z轴正方向，看向原点
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # MVP 矩阵：右乘原则
    mvp = proj @ view @ model
    
    for i in range(8):
        v = cube_vertices[i]
        # 补全齐次坐标
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法，转化为 NDC 坐标 [-1, 1]
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换：映射到 GUI 的 [0, 1] x [0, 1] 空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化立方体顶点（中心在原点，边长为2，坐标范围[-1,1]）
    # 前平面（z=-1）：0(1,1,-1), 1(-1,1,-1), 2(-1,-1,-1), 3(1,-1,-1)
    # 后平面（z=1）：4(1,1,1), 5(-1,1,1), 6(-1,-1,1), 7(1,-1,1)
    cube_vertices[0] = [1.0, 1.0, -1.0]
    cube_vertices[1] = [-1.0, 1.0, -1.0]
    cube_vertices[2] = [-1.0, -1.0, -1.0]
    cube_vertices[3] = [1.0, -1.0, -1.0]
    cube_vertices[4] = [1.0, 1.0, 1.0]
    cube_vertices[5] = [-1.0, 1.0, 1.0]
    cube_vertices[6] = [-1.0, -1.0, 1.0]
    cube_vertices[7] = [1.0, -1.0, 1.0]
    
    # 创建 GUI 窗口
    gui = ti.GUI("3D Cube Transformation (Taichi)", res=(700, 700))
    angle = 0.0
    # 为12条边定义不同颜色（增强立体感）
    edge_colors = [
        0xFF0000, 0xFF8800, 0xFFFF00, 0x88FF00,  # 前平面
        0x00FF00, 0x00FF88, 0x00FFFF, 0x0088FF,  # 后平面
        0x0000FF, 0x8800FF, 0xFF00FF, 0xFF0088   # 连接边
    ]
    
    while gui.running:
        # 交互控制
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 5.0  # 逆时针旋转（步长减小，旋转更顺滑）
            elif gui.event.key == 'd':
                angle -= 5.0  # 顺时针旋转
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 计算立方体顶点变换
        compute_transform(angle)
        
        # 绘制立方体的12条边
        for i, (v_idx1, v_idx2) in enumerate(cube_edges):
            p1 = screen_coords[v_idx1]
            p2 = screen_coords[v_idx2]
            gui.line(p1, p2, radius=2, color=edge_colors[i])
        
        # 显示窗口
        gui.show()

if __name__ == '__main__':
    main()
