import numpy as np
import cupy as cp
from itertools import combinations, product
from numba import jit
from vispy import scene, app
import time
from collections import deque
import sys # Import sys to check the Python path

# --- A quick check to see which Python interpreter is running the script ---
print(f"Running with Python interpreter: {sys.executable}\n")

@jit(nopython=True)
def find_line_segments_fast(roots_matrix):
    num_roots = roots_matrix.shape[0]
    line_indices = []
    for i in range(num_roots):
        for j in range(i + 1, num_roots):
            ip = np.dot(roots_matrix[i], roots_matrix[j])
            if np.abs(ip + 1.0) < 1e-9:
                line_indices.append((i, j))
    return line_indices

print("Generating E8 root system...")
roots1 = [vec for i, j in combinations(range(8), 2) for signs in product([-1, 1], repeat=2) for vec in [np.zeros(8)] if (vec.__setitem__(i, signs[0]), vec.__setitem__(j, signs[1])) == (None, None)]
roots2 = [np.array(signs) * 0.5 for signs in product([-1, 1], repeat=8) if np.prod(signs) == 1]
roots_matrix_cpu = np.array(roots1 + roots2, dtype=np.float32)

print("Finding line segments...")
line_segments_indices = find_line_segments_fast(roots_matrix_cpu)
line_connect_array = np.array(line_segments_indices, dtype=np.uint32)

from matplotlib.cm import viridis
initial_coords = roots_matrix_cpu[:, 0]
norm_coords = (initial_coords - initial_coords.min()) / (initial_coords.max() - initial_coords.min())
colors = viridis(norm_coords)

M_cpu = np.array([
    [0., -0.55679344, 0.19694925, -0.19694925, 0.08054772, -0.38529087, 0., 0.38529087],
    [0.18091315, 0., 0.16021295, 0.16021295, 0., 0.09901705, 0.76636042, 0.09901705],
    [0.33826121, 0, 0, -0.33826121, 0.67281636, 0.17150256, 0, -0.17150256]
], dtype=np.float32)

roots_matrix = cp.asarray(roots_matrix_cpu)
M = cp.asarray(M_cpu)
xp = cp

def get_rotation_matrix(theta, i, j, dim=8):
    R = xp.eye(dim, dtype=xp.float32)
    cos_theta, sin_theta = xp.cos(theta), xp.sin(theta)
    R[i, i], R[i, j], R[j, i], R[j, j] = cos_theta, -sin_theta, sin_theta, cos_theta
    return R

# --- VisPy Scene Setup ---
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1024, 1024), show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 4

points_visual = scene.visuals.Markers(parent=view.scene)
lines_visual = scene.visuals.Line(method='gl', parent=view.scene)
fps_text = scene.visuals.Text('', color='lime', font_size=10, pos=(20, 40), parent=canvas.scene)

# --- Fullscreen Toggle Function ---
def on_key_press(event):
    if event.key == 'Enter' and 'Alt' in event.modifiers:
        # Toggle the canvas fullscreen property
        canvas.fullscreen = not canvas.fullscreen

# Connect the function to the canvas's key press event
canvas.events.key_press.connect(on_key_press)

frame_times = deque(maxlen=60)
last_time = time.time()
frame_counter = 0

def update(event):
    global frame_counter, last_time
    
    theta = 2 * np.pi * frame_counter / 720.0
    phi = (1 + np.sqrt(5)) / 2
    R1 = get_rotation_matrix(theta, 0, 1)
    R2 = get_rotation_matrix(theta * phi, 2, 3)
    R3 = get_rotation_matrix(theta * phi**2, 4, 5)
    R = R1 @ R2 @ R3
    rotated_roots = roots_matrix @ R.T
    projected_gpu = (M @ rotated_roots.T).T
    projected_cpu = projected_gpu.get()
    
    points_visual.set_data(pos=projected_cpu, face_color=colors, size=6)
    lines_visual.set_data(pos=projected_cpu, connect=line_connect_array, color=(1, 1, 1, 0.2))
    
    current_time = time.time()
    delta = current_time - last_time
    last_time = current_time
    frame_times.append(delta)
    
    if frame_times:
        avg_time = sum(frame_times) / len(frame_times)
        if avg_time > 0:
            fps = 1.0 / avg_time
            fps_text.text = f'FPS: {fps:.1f}'

    frame_counter += 1

target_fps = 36.0
interval = 1.0 / target_fps
timer = app.Timer(interval=interval, connect=update, start=True)

if __name__ == '__main__':
    try:
        app.use_app('pyqt6')
    except Exception:
        print("PyQt6 backend not found, using default.")
    app.run()
