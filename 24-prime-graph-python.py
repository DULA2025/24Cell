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

# Generate prime candidates
prime_candidates = [p for p in range(5, 500) if is_prime(p) and prime_congruence(p)]
prime_vertices = [p - 1 for p in prime_candidates[:24]]  # Subtract 1 to convert to 0-indexing

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

# Improved 3D Visualization with labels
plotter = pv.Plotter()

# Create vertex positions using a sphere layout
phi = np.pi * (3 - np.sqrt(5))  # golden angle
theta = phi * np.arange(num_vertices)
y = 1 - (np.arange(num_vertices) / (num_vertices - 1)) * 2
radius = np.sqrt(1 - y * y)
points = np.zeros((num_vertices, 3))
points[:, 0] = radius * np.cos(theta) * 5
points[:, 1] = y * 5
points[:, 2] = radius * np.sin(theta) * 5

# Add vertices with labels
for i in range(num_vertices):
    sphere = pv.Sphere(center=points[i], radius=0.2)
    plotter.add_mesh(sphere, color='red', opacity=0.8)
    # Add vertex label (showing the original prime number)
    plotter.add_point_labels(
        points[i:i+1], 
        [str(prime_candidates[i])],  # Use the original prime number as label
        font_size=14,
        point_size=1,
        shape_opacity=0.0  # Make the label background transparent
    )

# Add edges
for i in range(num_vertices):
    for j in range(i + 1, num_vertices):
        if adjacency_matrix[i, j]:
            line = pv.Line(points[i], points[j])
            plotter.add_mesh(line, color='black', line_width=2)

# Set camera position and display
plotter.camera_position = 'xy'
plotter.show()
