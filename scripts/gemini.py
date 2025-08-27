import numpy as np
import numpy.linalg as la
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# =============================================================================
# DATA LOADING FUNCTIONS (Unchanged)
# =============================================================================
def readNode(filepath):
    """
    Prepares the data for processing

    Returns:
        nodes_coords: Array over the coordinates (x, y) of
        nodes with the index being the node number()
    """

    with open(filepath, "r") as nodeFile:
        header = nodeFile.readline().strip().split()
        # print(f"{header}")
        number_of_nodes = header[0]
        # print(f"{number_of_nodes}")

        nodes_coords = np.zeros((int(number_of_nodes), 2), dtype=np.float64)
        nodes_boundary_markers = np.zeros(int(number_of_nodes), dtype=np.int32)

        for i in range(int(number_of_nodes)):
            line = nodeFile.readline().strip().split()
            index = int(line[0]) - 1
            # print(f"{line}")
            # X coord
            nodes_coords[index, 0] = float(line[1])

            # Y coord
            nodes_coords[index, 1] = float(line[2])

            # boundary marker
            if int(line[3]) != 0:
                nodes_boundary_markers[index] = int(line[3])
                # print(f"{node_boundary_markers}")

        # print(f"{nodes_coords}")
        # print(f"{nodes_boundary_markers}")
        return nodes_coords, nodes_boundary_markers


def readEle(filepath):
    with open(filepath, "r") as eleFile:
        header = eleFile.readline().strip().split()
        # print(f"{header}")
        number_of_triangles = int(header[0])
        nodes_per_triangle = int(header[1])

        elements = np.zeros((number_of_triangles, 3), dtype=np.int32)

        for i in range(int(number_of_triangles)):
            line = eleFile.readline().strip().split()
            # index = int(line[0]) - 1 # 1 indexed
            # print(f"{index}")

            elements[i, 0] = int(line[1]) - 1
            elements[i, 1] = int(line[2]) - 1
            elements[i, 2] = int(line[3]) - 1

        # print(f"{elements}")
        return elements


# =============================================================================
# CORE FEM & BC FUNCTIONS (Corrected and Simplified)
# =============================================================================
def build_stiffness_matrix(nodes, triangles):
    N = nodes.shape[0]
    A = np.zeros((N, N), dtype=np.float64)
    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1, x2, y2, x3, y3 = nodes[p1,0], nodes[p1,1], nodes[p2,0], nodes[p2,1], nodes[p3,0], nodes[p3,1]
        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(ADet) < 1e-14: continue
        y_diffs = [y2-y3, y3-y1, y1-y2]
        x_diffs = [x3-x2, x1-x3, x2-x1]
        elem_K = (np.outer(y_diffs, y_diffs) + np.outer(x_diffs, x_diffs)) / (2.0 * abs(ADet))
        for i in range(3):
            for j in range(3):
                A[tri[i], tri[j]] += elem_K[i, j]
    return A

def calculate_divergence(nodes, triangles, u_star):
    N = nodes.shape[0]
    div_sum = np.zeros(N); area_sum = np.zeros(N)
    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1, x2, y2, x3, y3 = nodes[p1,0], nodes[p1,1], nodes[p2,0], nodes[p2,1], nodes[p3,0], nodes[p3,1]
        det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(det) < 1e-14: continue
        area = 0.5 * abs(det)
        inv2A = 1.0 / det
        u1, u2, u3 = u_star[p1], u_star[p2], u_star[p3]
        div_tri = (u1[0]*(y2-y3)+u2[0]*(y3-y1)+u3[0]*(y1-y2))*inv2A + (u1[1]*(x3-x2)+u2[1]*(x1-x3)+u3[1]*(x2-x1))*inv2A
        for i in range(3):
            div_sum[tri[i]] += div_tri * area
            area_sum[tri[i]] += area
    return div_sum / (area_sum + 1e-12)

def find_boundary_pairs(nodes_coords, L=1.0, tol=1e-6):
    left_indices = np.where(np.abs(nodes_coords[:, 0]) < tol)[0]
    right_indices = np.where(np.abs(nodes_coords[:, 0] - L) < tol)[0]
    if len(left_indices) == 0 or len(right_indices) == 0: return []
    right_y_coords_for_tree = nodes_coords[right_indices, 1].reshape(-1, 1)
    kdtree = KDTree(right_y_coords_for_tree)
    pairs = []
    for left_idx in left_indices:
        _, right_array_idx = kdtree.query([nodes_coords[left_idx, 1]])
        right_node_idx = right_indices[right_array_idx]
        pairs.append((left_idx, right_node_idx))
    return pairs

def apply_periodic_bc_penalty(A, pairs):
    penalty = 1.0e10
    for master_idx, slave_idx in pairs:
        A[master_idx, master_idx] += penalty
        A[slave_idx, slave_idx] += penalty
        A[master_idx, slave_idx] -= penalty
        A[slave_idx, master_idx] -= penalty

def calculate_gradient(nodes_coords, triangles, p_vector):
    """Calculates the nodal gradient using a robust area-weighted average."""
    N = nodes_coords.shape[0]
    grad_sum = np.zeros((N, 2), dtype=np.float64)
    area_sum = np.zeros(N, dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes_coords[p1]
        x2, y2 = nodes_coords[p2]
        x3, y3 = nodes_coords[p3]
        
        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(ADet) < 1e-14: continue
        
        area = 0.5 * abs(ADet)
        
        p1_val, p2_val, p3_val = p_vector[p1], p_vector[p2], p_vector[p3]
        
        # Calculate the constant gradient over the triangle
        grad_p_x = (p1_val*(y2-y3) + p2_val*(y3-y1) + p3_val*(y1-y2)) / ADet
        grad_p_y = (p1_val*(x3-x2) + p2_val*(x1-x3) + p3_val*(x2-x1)) / ADet
        grad_p_tri = np.array([grad_p_x, grad_p_y])
        
        # Add the triangle's contribution to each of its three nodes
        for i in range(3):
            grad_sum[tri[i]] += grad_p_tri * area
            area_sum[tri[i]] += area

    # --- THIS IS THE MISSING STEP ---
    # Divide the sum of contributions by the sum of areas to get the average
    final_grad = grad_sum / (area_sum[:, np.newaxis] + 1e-12)
    
    return final_grad

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# --- Parameters & Setup ---
NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
INNER_BOUNDARY_MARKER = 2
H = 1.0; tol = 1e-6; DT = 0.001; v = 0.01; rho = 1.0

nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
N = nodes_coords.shape[0]
triangles = readEle(ELE_FILEPATH)

# --- Boundary Conditions Setup ---
# 1. Periodic BCs
pairs = find_boundary_pairs(nodes_coords)
filtered_pairs = []
for master_idx, slave_idx in pairs:
    if not (np.abs(nodes_coords[master_idx, 1] - 0.0) < tol or np.abs(nodes_coords[master_idx, 1] - H) < tol):
        filtered_pairs.append((master_idx, slave_idx))

# 2. Dirichlet (no-slip) BCs
dirichlet_nodes = []
for i in range(N):
    y_coord = nodes_coords[i, 1]
    marker = nodes_boundary_markers[i]
    if np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol or marker == INNER_BOUNDARY_MARKER:
        dirichlet_nodes.append(i)

# --- Matrix Assembly ---
A = build_stiffness_matrix(nodes_coords, triangles)
Ap = A.copy()
A1 = np.eye(N) + DT * v * A

apply_periodic_bc_penalty(A1, filtered_pairs)
apply_periodic_bc_penalty(Ap, filtered_pairs)

for i in dirichlet_nodes:
    A1[i, :] = 0.0; A1[:, i] = 0.0; A1[i, i] = 1.0

try:
    interior_node_idx = np.where(nodes_boundary_markers == 0)[0][0]
except IndexError:
    interior_node_idx = 0
Ap[interior_node_idx, :] = 0.0; Ap[:, interior_node_idx] = 0.0; Ap[interior_node_idx, interior_node_idx] = 1.0

# --- Initial Conditions ---
u = np.zeros((N, 2))
p = np.zeros(N)
b = np.zeros((N, 2))
b[:, 0] = 0.1 # Apply a body force

# --- Simulation & Plotting ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
colorbar = None

for step in range(1, 601):
    rhs_x = u[:, 0] + DT * b[:, 0]
    rhs_y = u[:, 1] + DT * b[:, 1]
    
    for i in dirichlet_nodes:
        rhs_x[i] = 0.0
        rhs_y[i] = 0.0

    u_star_x = la.solve(A1, rhs_x)
    u_star_y = la.solve(A1, rhs_y)
    u_star = np.stack([u_star_x, u_star_y], axis=1)

    div = calculate_divergence(nodes_coords, triangles, u_star)
    b_p = -(rho / DT) * div
    b_p -= b_p.mean()
    b_p[interior_node_idx] = 0.0
    p = la.solve(Ap, b_p)
    
    grad_p = calculate_gradient(nodes_coords, triangles, p) # You will need your simple gradient function
    u = u_star - DT * grad_p
    
    for i in dirichlet_nodes:
        u[i, :] = 0.0
    for master, slave in filtered_pairs:
        u[slave, :] = u[master, :]

    if step % 5 == 0:
        final_div = calculate_divergence(nodes_coords, triangles, u)
        print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Max P: {np.max(np.abs(p)):.2e}, Final Div(u): {np.max(np.abs(final_div)):.2e}")
        # (Your plotting code here)

plt.ioff()
plt.show()
