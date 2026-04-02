import taichi as ti
import numpy as np

# 使用gpu后端
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线采样密度

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI绘制数据
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 曲线采样点GPU缓存
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# 模式标记：True=贝塞尔，False=B样条
use_bezier = ti.field(dtype=ti.i32, shape=())
use_bezier[None] = 1

# ------------------- 贝塞尔：De Casteljau 算法 -------------------
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i+1]
        x = (1 - t) * p0[0] + t * p1[0]
        y = (1 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

# ------------------- B样条：均匀三次B样条 -------------------
def uniform_cubic_bspline(control_points):
    """生成均匀三次B样条曲线点（分段，4点一段，C2连续）"""
    n = len(control_points)
    if n < 4:
        return np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    
    # 三次B样条基矩阵（标准均匀）
    B = np.array([
        [1, 4, 1, 0],
        [-3, 0, 3, 0],
        [3, -6, 3, 0],
        [-1, 3, -3, 1]
    ], dtype=np.float32) / 6.0

    num_segments_total = n - 3  # n个控制点生成n-3段
    samples_per_segment = NUM_SEGMENTS // num_segments_total if num_segments_total > 0 else NUM_SEGMENTS
    curve_points = []

    for i in range(num_segments_total):
        p0 = control_points[i]
        p1 = control_points[i+1]
        p2 = control_points[i+2]
        p3 = control_points[i+3]
        P = np.array([p0, p1, p2, p3], dtype=np.float32)

        for k in range(samples_per_segment + 1):
            t = k / samples_per_segment
            T = np.array([1, t, t**2, t**3], dtype=np.float32)
            pt = T @ B @ P
            curve_points.append(pt)

    # 对齐采样数量
    curve_np = np.array(curve_points, dtype=np.float32)
    if len(curve_np) > NUM_SEGMENTS + 1:
        curve_np = curve_np[:NUM_SEGMENTS + 1]
    return curve_np

# ------------------- 反走样绘制核心（3×3距离加权） -------------------
@ti.kernel
def draw_antialiased_curve(n: ti.i32):
    for i in range(n):
        x, y = curve_points_field[i]
        fx = x * WIDTH
        fy = y * HEIGHT

        # 遍历3×3邻域（亚像素）
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                cx = ti.cast(fx + dx, ti.i32)
                cy = ti.cast(fy + dy, ti.i32)
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    # 像素中心到几何点的欧氏距离
                    dist_sq = (fx - (cx + 0.5))**2 + (fy - (cy + 0.5))**2
                    # 距离衰减权重：越近越亮
                    weight = ti.exp(-dist_sq * 2.0)
                    # 混合颜色（绿色曲线）
                    pixels[cx, cy] += ti.Vector([0.0, weight, 0.0])

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# ------------------- 主程序 -------------------
def main():
    window = ti.ui.Window("Bezier / B-Spline (Anti-Aliased)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    print("=== 使用说明 ===")
    print("鼠标左键：添加控制点")
    print("B键：切换 贝塞尔 / B样条")
    print("C键：清空画布")
    
    while window.running:
        # 事件处理
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"添加控制点：{pos}")
            elif e.key == 'b':
                use_bezier[None] = 1 - use_bezier[None]
                mode = "贝塞尔" if use_bezier[None] else "B样条"
                print(f"切换至：{mode}")
            elif e.key == 'c':
                control_points = []
                print("画布已清空")
        
        clear_pixels()
        current_count = len(control_points)
        
        # 计算曲线点
        if current_count >= 2:
            if use_bezier[None]:
                # 贝塞尔
                curve_np = np.zeros((NUM_SEGMENTS+1,2), np.float32)
                for t_int in range(NUM_SEGMENTS+1):
                    t = t_int / NUM_SEGMENTS
                    curve_np[t_int] = de_casteljau(control_points, t)
            else:
                # B样条（至少4点才绘制）
                curve_np = uniform_cubic_bspline(control_points)
            
            curve_points_field.from_numpy(curve_np)
            draw_antialiased_curve(len(curve_np))
        
        # 绘制控制点与连线
        canvas.set_image(pixels)
        if current_count > 0:
            np_pts = np.full((MAX_CONTROL_POINTS,2), -10, np.float32)
            np_pts[:current_count] = control_points
            gui_points.from_numpy(np_pts)
            canvas.circles(gui_points, radius=0.006, color=(1,0,0))
            
            if current_count >= 2:
                indices = []
                for i in range(current_count-1):
                    indices += [i, i+1]
                np_idx = np.zeros(MAX_CONTROL_POINTS*2, np.int32)
                np_idx[:len(indices)] = indices
                gui_indices.from_numpy(np_idx)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5,0.5,0.5))
        
        window.show()

if __name__ == '__main__':
    main()
