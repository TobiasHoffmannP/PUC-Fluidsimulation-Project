import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# --- Mesh File Reading ---
def readNode(filepath):
    with open(filepath, "r") as nodeFile:
        header = nodeFile.readline().strip().split()
        number_of_nodes = int(header[0])
        nodes_coords = np.zeros((number_of_nodes, 2), dtype=np.float64)
        nodes_boundary_markers = np.zeros(number_of_nodes, dtype=np.int32)
        for i in range(number_of_nodes):
            line = nodeFile.readline().strip().split()
            index = int(line[0]) - 1 
            nodes_coords[index, 0] = float(line[1])
            nodes_coords[index, 1] = float(line[2])
            if int(line[3]) != 0:
                nodes_boundary_markers[index] = int(line[3])
        return nodes_coords, nodes_boundary_markers

def readEle(filepath):
    with open(filepath, "r") as eleFile:
        header = eleFile.readline().strip().split()
        number_of_triangles = int(header[0])
        elements = np.zeros((number_of_triangles, 3), dtype=np.int32)
        for i in range(number_of_triangles):
            line = eleFile.readline().strip().split()
            elements[i, 0] = int(line[1]) - 1
            elements[i, 1] = int(line[2]) - 1
            elements[i, 2] = int(line[3]) - 1
        return elements

# --- FEM Helper Functions ---
def buildFemSystem(nodes, triangles):
    N = nodes.shape[0]
    AMatrix = np.zeros((N, N), dtype=np.float64)
    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]; x2, y2 = nodes[p2]; x3, y3 = nodes[p3]
        ADet = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if ADet == 0: continue
        y_diffs = [y2 - y3, y3 - y1, y1 - y2]
        x_diffs = [x3 - x2, x1 - x3, x2 - x1]
        for i in range(3):
            for j in range(3):
                integral_val = (y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j]) / (2.0 * ADet)
                AMatrix[tri[i], tri[j]] += integral_val
    return AMatrix

def find_boundary_pairs(nodes_coords, L=1.0, tol=1e-6):
    left_indices = np.where(np.abs(nodes_coords[:, 0]) < tol)[0]
    right_indices = np.where(np.abs(nodes_coords[:, 0] - L) < tol)[0]
    if len(left_indices) == 0 or len(right_indices) == 0: return []
    left_coords = nodes_coords[left_indices]
    right_coords = nodes_coords[right_indices]
    kdtree = KDTree(right_coords[:, 1].reshape(-1, 1))
    pairs = []
    for i, left_idx in enumerate(left_indices):
        left_y = left_coords[i, 1]
        _, right_array_idx = kdtree.query([left_y])
        right_node_idx = right_indices[right_array_idx]
        pairs.append((left_idx, right_node_idx))
    return pairs

def apply_periodic_bc(A, pairs):
    b_dummy = np.zeros(A.shape[0])
    for master_idx, slave_idx in pairs:
        A[master_idx, :] += A[slave_idx, :]
        A[slave_idx, :] = 0.0
        A[slave_idx, slave_idx] = 1.0
        A[slave_idx, master_idx] = -1.0

def apply_dirchlect_u(u, nodes, nodes_boundary_markers, H, tol, INNER_BOUNDARY_MARKER):
    for i in range(u.shape[0]):
        marker = nodes_boundary_markers[i]
        _, y_coord = nodes[i]
        is_wall = np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol
        is_inner_boundary = (marker == INNER_BOUNDARY_MARKER)
        if is_wall or is_inner_boundary:
            u[i, 0] = 0.0
            u[i, 1] = 0.0
    return u

def apply_periodic_u(u, pairs):
    for master, slave in pairs:
        u[slave] = u[master]
    return u

def calculate_divergence(nodes, triangles, u_star):
    N = nodes.shape[0]
    nodal_divergence_sum = np.zeros(N)
    nodal_area_sum = np.zeros(N)
    for tri in triangles:
        p1, p2, p3 = tri
        (x1, y1), (x2, y2), (x3, y3) = nodes[p1], nodes[p2], nodes[p3]
        A_det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if A_det == 0: continue
        area = 0.5 * A_det
        (ux1, uy1), (ux2, uy2), (ux3, uy3) = u_star[p1], u_star[p2], u_star[p3]
        d_ux_dx = (ux1*(y2-y3) + ux2*(y3-y1) + ux3*(y1-y2)) / A_det
        d_uy_dy = (uy1*(x3-x2) + uy2*(x1-x3) + uy3*(x2-x1)) / A_det
        div_tri = d_ux_dx + d_uy_dy
        for i in range(3):
            nodal_divergence_sum[tri[i]] += div_tri * area
            nodal_area_sum[tri[i]] += area
    return nodal_divergence_sum / (nodal_area_sum + 1e-12)

def calculate_gradient(nodes, triangles, p_vector):
    N = nodes.shape[0]
    nodal_gradient_sum = np.zeros((N, 2))
    nodal_area_sum = np.zeros(N)
    for tri in triangles:
        p1, p2, p3 = tri
        (x1, y1), (x2, y2), (x3, y3) = nodes[p1], nodes[p2], nodes[p3]
        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if ADet == 0: continue
        area = 0.5 * ADet
        p1_val, p2_val, p3_val = p_vector[p1], p_vector[p2], p_vector[p3]
        grad_p_x = (p1_val*(y2-y3) + p2_val*(y3-y1) + p3_val*(y1-y2)) / ADet
        grad_p_y = (p1_val*(x3-x2) + p2_val*(x1-x3) + p3_val*(x2-x1)) / ADet
        grad_p_tri = np.array([grad_p_x, grad_p_y])
        for i in range(3):
            nodal_gradient_sum[tri[i]] += grad_p_tri * area
            nodal_area_sum[tri[i]] += area
    return nodal_gradient_sum / (nodal_area_sum[:, np.newaxis] + 1e-12)

def build_advection_matrix(nodes, triangles, u):
    N = nodes.shape[0]
    A_adv = np.zeros((N, N))
    for tri in triangles:
        p1, p2, p3 = tri
        global_indices = [p1, p2, p3]
        (x1, y1), (x2, y2), (x3, y3) = nodes[p1], nodes[p2], nodes[p3]
        u_avg = (u[p1] + u[p2] + u[p3]) / 3.0
        A_element = np.zeros((3, 3))
        A_det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if A_det == 0: continue
        area = 0.5 * A_det
        y_diffs = [y2-y3, y3-y1, y1-y2]
        x_diffs = [x3-x2, x1-x3, x2-x1]
        for j in range(3):
            grad_phi_j_x = y_diffs[j] / A_det
            grad_phi_j_y = x_diffs[j] / A_det
            u_dot_grad_phi = u_avg[0] * grad_phi_j_x + u_avg[1] * grad_phi_j_y
            column_value = -u_dot_grad_phi * (area / 3.0)
            A_element[:, j] = column_value
        for i in range(3):
            for j in range(3):
                A_adv[global_indices[i], global_indices[j]] += A_element[i, j]
    return A_adv

# --- MAIN SCRIPT ---

# --- 1. SETUP ---
# Parameters
NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
H = 1.0
tol = 1e-6
INNER_BOUNDARY_MARKER = 2
DT = 0.001
NUM_STEPS = 1000
v = 0.1 # Viscosity
rho = 1.0 # Density

# Load Mesh
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
N = nodes_coords.shape[0]
triangles = readEle(ELE_FILEPATH)

# Periodic BC Setup
pairs = find_boundary_pairs(nodes_coords, L=1.0)
filtered_pairs = []
for master_idx, slave_idx in pairs:
    master_y = nodes_coords[master_idx, 1]
    if not (np.abs(master_y - 0.0) < tol or np.abs(master_y - H) < tol):
        filtered_pairs.append((master_idx, slave_idx))

# Build Base Matrices
A_base = buildFemSystem(nodes_coords, triangles)
b = np.zeros((N, 2))
b[:, 0] = 0.01  # Driving body force

# Velocity System Matrix
A_vel = A_base.copy()
apply_periodic_bc(A_vel, filtered_pairs)
for i in range(N):
    marker = nodes_boundary_markers[i]
    y_coord = nodes_coords[i, 1]
    is_wall = np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol
    is_inner = (marker == INNER_BOUNDARY_MARKER)
    if is_wall or is_inner:
        A_vel[i, :] = 0.0
        A_vel[i, i] = 1.0

# Pressure System Matrix
A_p = A_base.copy()
apply_periodic_bc(A_p, filtered_pairs)
A_p[0, :] = 0.0
A_p[0, 0] = 1.0

# Initial Conditions
u = np.zeros((N, 2), dtype=np.float64)
u = apply_dirchlect_u(u, nodes_coords, nodes_boundary_markers, H, tol, INNER_BOUNDARY_MARKER)
u = apply_periodic_u(u, pairs)

# --- 2. PLOTTING SETUP ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

# --- 3. TIME-STEPPING LOOP ---
for step in range(NUM_STEPS):
    # Advection-Viscosity System Setup
    A_adv = build_advection_matrix(nodes_coords, triangles, u)
    A_new = np.eye(N) + DT * A_adv + v * DT * A_vel

    # Body Force Update
    u_after_force = u + DT * b

    # Advection-Viscosity Solve
    rhs_x = u_after_force[:, 0]
    rhs_y = u_after_force[:, 1]
    u_star_x = np.linalg.solve(A_new, rhs_x)
    u_star_y = np.linalg.solve(A_new, rhs_y)
    u_star = np.stack([u_star_x, u_star_y], axis=1)

    # Pressure Solve
    div = calculate_divergence(nodes_coords, triangles, u_star)
    b_p = (rho / DT) * div
    b_p[0] = 0.0
    p = np.linalg.solve(A_p, b_p)

    # Projection
    grad_p = calculate_gradient(nodes_coords, triangles, p)
    u = u_star - DT * grad_p

    # Apply Velocity BCs
    u = apply_periodic_u(u, pairs)
    u = apply_dirchlect_u(u, nodes_coords, nodes_boundary_markers, H, tol, INNER_BOUNDARY_MARKER)

    # Update Plot
    if step > 0 and step % 20 == 0:
        print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Max P: {np.max(np.abs(p)):.2e}")
        ax.clear()

        # Velocity Magnitude
        u_magnitude = np.linalg.norm(u, axis=1)
        current_max_u = np.max(u_magnitude)
        vmax = current_max_u if current_max_u > 1e-9 else 0.1
        tpc = ax.tripcolor(triang, u_magnitude, shading="gouraud", cmap="viridis", vmin=0, vmax=vmax)

        # Quiver
        skip = 25
        ax.quiver(nodes_coords[::skip, 0], nodes_coords[::skip, 1], u[::skip, 0], u[::skip, 1],
                  color='white', scale=vmax * 20, width=0.004)

        # Streamlines
        interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
        interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])
        u_x_grid = interpolator_x(grid_x, grid_y)
        u_y_grid = interpolator_y(grid_x, grid_y)
        ax.streamplot(grid_x, grid_y, u_x_grid, u_y_grid, color='black', linewidth=1, density=0.7)

        # Reset plot properties
        ax.set_aspect("equal")
        ax.set_title(f"Time Evolution: Step {step}/{NUM_STEPS}")

        fig.canvas.draw_idle()
        plt.pause(0.001)

# --- 4. FINALIZE PLOT ---
print("Animation finished.")
plt.ioff()

# Redraw final state for a clean plot with a colorbar
ax.clear()
u_magnitude = np.linalg.norm(u, axis=1)
vmax = np.max(u_magnitude) if np.max(u_magnitude) > 1e-9 else 0.1
tpc = ax.tripcolor(triang, u_magnitude, shading="gouraud", cmap="viridis", vmin=0, vmax=vmax)
fig.colorbar(tpc, ax=ax, label="Velocity Magnitude")

skip=25
ax.quiver(nodes_coords[::skip, 0], nodes_coords[::skip, 1], u[::skip, 0], u[::skip, 1],
          color='white', scale=vmax * 20, width=0.004)

interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])
u_x_grid = interpolator_x(grid_x, grid_y)
u_y_grid = interpolator_y(grid_x, grid_y)
ax.streamplot(grid_x, grid_y, u_x_grid, u_y_grid, color='black', linewidth=1, density=0.7)

ax.set_aspect("equal")
ax.set_title(f"Final State at Step {NUM_STEPS}")
plt.show()
