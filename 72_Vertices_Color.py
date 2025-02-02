import numpy as np
import math
import pyvista as pv
from numba import jit

@jit(nopython=True)
def prime_congruence(p):
    return p > 3 and (p % 6 == 1 or p % 6 == 5)

@jit(nopython=True)
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Generate prime candidates - increased upper limit to ensure we get enough primes
prime_candidates = [p for p in range(5, 1000) if is_prime(p) and prime_congruence(p)]
prime_vertices = [p - 1 for p in prime_candidates[:72]]  # Changed to 72 vertices

# Create adjacency matrix
num_vertices = len(prime_vertices)
adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=np.int32)

# Define edges
for i in range(num_vertices):
    for j in range(i + 1, num_vertices):
        should_connect = False
        
        # Connect if the difference in their prime indices is prime
        if is_prime(abs(prime_candidates[i] - prime_candidates[j])):
            should_connect = True
        # Connect if one is a multiple of the other
        elif prime_candidates[j] % prime_candidates[i] == 0 or prime_candidates[i] % prime_candidates[j] == 0:
            should_connect = True
        # Check if the difference forms a perfect square
        elif math.sqrt((prime_candidates[i] - prime_candidates[j])**2) % 1 == 0:
            should_connect = True
        # Simulate a cycle or grid pattern
        elif (i - j) % 5 == 0 or (j - i) % 5 == 0:
            should_connect = True
        # Additional geometric structures
        elif abs(prime_candidates[i] - prime_candidates[j]) in [3, 4, 5]:
            should_connect = True
        # Angle relationships
        elif math.atan2(prime_candidates[j] - prime_candidates[i], 
                       prime_candidates[j] + prime_candidates[i]) % (math.pi / 3) == 0:
            should_connect = True
            
        if should_connect:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

# Generate CNF clauses
cn = 5  # Number of colors
clauses = []

def lit(vertex_idx, color):
    vertex = prime_vertices[vertex_idx]
    if is_prime(vertex + 1) and prime_congruence(vertex + 1):
        return (vertex + 1) * cn + color + 1
    else:
        return (vertex + 1) * cn + color + 1

# Generate vertex color clauses
for v in range(num_vertices):
    clauses.append([lit(v, c) for c in range(cn)])
    for c1 in range(cn):
        for c2 in range(c1):
            clauses.append([-lit(v, c1), -lit(v, c2)])

# Generate edge color clauses
for i in range(num_vertices):
    for j in range(i + 1, num_vertices):
        if adjacency_matrix[i, j]:
            for c in range(cn):
                clauses.append([-lit(i, c), -lit(j, c)])

# Output CNF formula
print('p cnf', num_vertices * cn, len(clauses))
for clause in clauses:
    print(*clause, 0)

# Improved 3D Visualization with labels and color coding
plotter = pv.Plotter()

# Create vertex positions using a modified sphere layout
# Using golden spiral distribution for better spacing with more points
golden_ratio = (1 + np.sqrt(5)) / 2
n = num_vertices
points = np.zeros((n, 3))

for i in range(n):
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = 2 * np.pi * i / golden_ratio
    
    # Increased radius for better spacing
    r = 8
    points[i, 0] = r * np.cos(theta) * np.sin(phi)
    points[i, 1] = r * np.sin(theta) * np.sin(phi)
    points[i, 2] = r * np.cos(phi)

# Color coding based on prime congruence
vertex_colors = ['red' if prime_candidates[i] % 6 == 1 else 'blue' for i in range(num_vertices)]

# Add vertices with labels and color coding
for i in range(num_vertices):
    # Smaller sphere radius for less overlap
    sphere = pv.Sphere(center=points[i], radius=0.15)
    plotter.add_mesh(sphere, color=vertex_colors[i], opacity=0.8)
    # Smaller font size for better readability with more vertices
    plotter.add_point_labels(
        points[i:i+1], 
        [str(prime_candidates[i])],
        font_size=10,
        point_size=1,
        shape_opacity=0.0
    )

# Add edges with reduced opacity for better visibility
for i in range(num_vertices):
    for j in range(i + 1, num_vertices):
        if adjacency_matrix[i, j]:
            line = pv.Line(points[i], points[j])
            plotter.add_mesh(line, color='black', opacity=0.3, line_width=1)

# Set camera position and display properties
plotter.camera_position = 'xy'
plotter.background_color = 'white'  # White background for better contrast
plotter.show()
