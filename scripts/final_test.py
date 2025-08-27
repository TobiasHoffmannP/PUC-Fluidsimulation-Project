import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# =============================================================================
# CORE FEM FUNCTIONS (CLEAN IMPLEMENTATION)
# =============================================================================

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

def build_stiffness_matrix(nodes, triangles):
    N = nodes.shape[0]
    K = np.zeros((N, N), dtype=np.float64)
    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]
        
        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(ADet) < 1e-14: continue

        y_diffs = [y2-y3, y3-y1, y1-y2]
        x_diffs = [x3-x2, x1-x3, x2-x1]

        # Element stiffness matrix calculation
        elem_K_val = (np.outer(y_diffs, y_diffs) + np.outer(x_diffs, x_diffs)) / (2.0 * abs(ADet))
        
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += elem_K_val[i, j]
    return K

def calculate_divergence(nodes, triangles, u_field):
    N = nodes.shape[0]
    nodal_div_sum = np.zeros(N, dtype=np.float64)
    nodal_area_sum = np.zeros(N, dtype=np.float64)
    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]

        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(ADet) < 1e-14: continue
        
        area = 0.5 * abs(ADet)
        u1, u2, u3 = u_field[p1], u_field[p2], u_field[p3]

        # Note: ADet must be signed here for correct derivative signs
        d_ux_dx = (u1[0]*(y2-y3) + u2[0]*(y3-y1) + u3[0]*(y1-y2)) / ADet
        d_uy_dy = (u1[1]*(x3-x2) + u2[1]*(x1-x3) + u3[1]*(x2-x1)) / ADet
        div_tri = d_ux_dx + d_uy_dy
        
        for i in range(3):
            nodal_div_sum[tri[i]] += div_tri * area
            nodal_area_sum[tri[i]] += area
            
    return nodal_div_sum / (nodal_area_sum + 1e-12)

def calculate_gradient(nodes, triangles, p_field):
    N = nodes.shape[0]
    nodal_grad_sum = np.zeros((N, 2), dtype=np.float64)
    nodal_area_sum = np.zeros(N, dtype=np.float64)
    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]

        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(ADet) < 1e-14: continue
        
        area = 0.5 * abs(ADet)
        p1_val, p2_val, p3_val = p_field[p1], p_field[p2], p_field[p3]

        # Note: ADet must be signed here for correct derivative signs
        grad_p_x = (p1_val*(y2-y3) + p2_val*(y3-y1) + p3_val*(y1-y2)) / ADet
        grad_p_y = (p1_val*(x3-x2) + p2_val*(x1-x3) + p3_val*(x2-x1)) / ADet
        grad_p_tri = np.array([grad_p_x, grad_p_y])
        
        for i in range(3):
            nodal_grad_sum[tri[i]] += grad_p_tri * area
            nodal_area_sum[tri[i]] += area
            
    return nodal_grad_sum / (nodal_area_sum[:, np.newaxis] + 1e-12)


# =============================================================================
# SCRIPT TO TEST A SINGLE PROJECTION STEP
# =============================================================================

# 1. Load the mesh
NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
nodes_coords, _ = readNode(NODE_FILEPATH)
triangles = readEle(ELE_FILEPATH)
N = nodes_coords.shape[0]

# --- Create a known velocity field u* with known divergence ---
# u_star = [x, y], so ∇·u* = 2
u_star = np.copy(nodes_coords)

# --- Build the pressure system ---
A_p = build_stiffness_matrix(nodes_coords, triangles)
# Pin pressure at node 0 to make the system solvable
A_p[0, :] = 0.0
A_p[0, 0] = 1.0

# --- Perform ONE projection step ---
DT = 0.01
rho = 1.0

# 1. Calculate divergence of the known field
div_u_star = calculate_divergence(nodes_coords, triangles, u_star)

# 2. Solve for pressure
b_p = -(rho / DT) * div_u_star
b_p[0] = 0.0  # Apply the pinned BC
p = np.linalg.solve(A_p, b_p)

# 3. Project to get the final velocity u
grad_p = calculate_gradient(nodes_coords, triangles, p)
u_final = u_star - DT * grad_p

# 4. Verify: Calculate the divergence of the final field
final_div = calculate_divergence(nodes_coords, triangles, u_final)

# --- Print the results ---
print("--- Single Projection Test ---")
print(f"Max divergence of initial field u*: {np.max(np.abs(div_u_star)):.2e} (Analytical value is 2.0)")
print(f"Max divergence of final field u:   {np.max(np.abs(final_div)):.2e} (Should be near zero)")

# --- Plot the fields ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)

vmax1 = np.max(np.abs(div_u_star))
axes[0].tripcolor(triang, div_u_star, shading='gouraud', cmap='coolwarm', vmin=-vmax1, vmax=vmax1)
axes[0].set_title("Divergence of Initial Field (u*)")

vmax2 = np.max(np.abs(p))
axes[1].tripcolor(triang, p, shading='gouraud', cmap='viridis', vmin=0, vmax=vmax2)
axes[1].set_title("Resulting Pressure Field (p)")

vmax3 = np.max(np.abs(final_div)) if np.max(np.abs(final_div)) > 1e-12 else 1e-12
axes[2].tripcolor(triang, final_div, shading='gouraud', cmap='coolwarm', vmin=-vmax3, vmax=vmax3)
axes[2].set_title("Divergence of Final Field (u)")

for ax in axes:
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
