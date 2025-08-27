import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# --- Copy these five functions from your main script into this file ---
# readNode, readEle, buildFemSystem, calculate_divergence_nikolaj, calculate_gradient
def readNode(filepath):
    with open(filepath, "r") as nodeFile:
        header = nodeFile.readline().strip().split()
        # print(f"{header}")
        number_of_nodes = header[0]
        # print(f"{number_of_nodes}")

        nodes_coords = np.zeros((int(number_of_nodes), 2), dtype=np.float32)
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

def buildFemSystem(nodes, triangles, g_source):
    N = nodes.shape[0]
    AMatrix = np.zeros((N,N), dtype=np.float64)
    BVector = np.zeros(N, dtype=np.float64)
    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1, x2, y2, x3, y3 = nodes[p1, 0], nodes[p1, 1], nodes[p2, 0], nodes[p2, 1], nodes[p3, 0], nodes[p3, 1]
        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(ADet) < 1e-14: continue

        y_diffs = [y2 - y3, y3 - y1, y1 - y2]
        x_diffs = [x3 - x2, x1 - x3, x2 - x1]
        for i in range(3):
            for j in range(3):
                integral_val = (y_diffs[i]*y_diffs[j] + x_diffs[i]*x_diffs[j]) / (2.0 * abs(ADet))
                AMatrix[tri[i], tri[j]] += integral_val
        
        area = 0.5 * abs(ADet)
        g = g_source((x1+x2+x3)/3, (y1+y2+y3)/3) if callable(g_source) else g_source
        sourceIntegral = g * area / 3.0
        BVector[p1] += sourceIntegral
        BVector[p2] += sourceIntegral
        BVector[p3] += sourceIntegral
    return AMatrix, BVector

def build_mass_matrix(nodes, triangles):
    """Builds the FEM mass matrix M, robust to winding order."""
    N = nodes.shape[0]
    M = np.zeros((N, N), dtype=np.float64)
    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1, x2, y2, x3, y3 = nodes[p1, 0], nodes[p1, 1], nodes[p2, 0], nodes[p2, 1], nodes[p3, 0], nodes[p3, 1]
        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        
        # --- FIX: Use absolute area ---
        area = 0.5 * abs(ADet)
        if area < 1e-14: continue

        M_element = (area / 12.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        for i in range(3):
            for j in range(3):
                M[tri[i], tri[j]] += M_element[i, j]
    return M

def calculate_divergence_simple(nodes, triangles, u_star):
    N = nodes.shape[0]
    nodal_divergence_sum = np.zeros(N, dtype=np.float64)
    nodal_area_sum = np.zeros(N, dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri
        
        # Get vertex coordinates
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]
        
        # this is the same determinant used to find the area
        A_det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if A_det == 0: continue

        area = 0.5 * A_det     

        ux1, uy1 = u_star[p1]
        ux2, uy2 = u_star[p2]
        ux3, uy3 = u_star[p3]

        # div = (d(ux)/dx) + (d(uy)/dy)
        d_ux_dx = (ux1 * (y2 - y3) + ux2 * (y3 - y1) + ux3 * (y1 - y2)) / A_det
        d_uy_dy = (uy1 * (x3 - x2) + uy2 * (x1 - x3) + uy3 * (x2 - x1)) / A_det
        div_tri = d_ux_dx + d_uy_dy

        # add the triangle's divergence to its vertices, weighted by area
        for i in range(3):
            nodal_divergence_sum[tri[i]] += div_tri * area
            nodal_area_sum[tri[i]] += area

    # finalize the average by dividing by the total area contribution at each node
    # add a small epsilon to avoid division by zero for unused nodes
    final_divergence = nodal_divergence_sum / (nodal_area_sum + 1e-12)
    
    return final_divergence

def calculate_gradient_simple(nodes, triangles, p_vector):
    N = nodes.shape[0]
    # Initialize sum arrays for averaging
    nodal_gradient_sum = np.zeros((N, 2), dtype=np.float64) # For N x 2 vectors
    nodal_area_sum = np.zeros(N, dtype=np.float64)

    for tri in triangles:
        p1_idx, p2_idx, p3_idx = tri
        
        # Get vertex coordinates
        x1, y1 = nodes[p1_idx]
        x2, y2 = nodes[p2_idx]
        x3, y3 = nodes[p3_idx]
        
        # Calculate area determinant
        ADet = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if ADet == 0: continue
        
        area = 0.5 * ADet

        # Get nodal pressure values for this triangle
        p_val1 = p_vector[p1_idx]
        p_val2 = p_vector[p2_idx]
        p_val3 = p_vector[p3_idx]

        # Calculate the constant gradient vector (grad_p_x, grad_p_y) within this triangle
        grad_p_x = (p_val1 * (y2 - y3) + p_val2 * (y3 - y1) + p_val3 * (y1 - y2)) / ADet
        grad_p_y = (p_val1 * (x3 - x2) + p_val2 * (x1 - x3) + p_val3 * (x2 - x1)) / ADet
        
        grad_p_tri = np.array([grad_p_x, grad_p_y])

        # Add the triangle's gradient to its vertices, weighted by area
        for i in range(3):
            nodal_gradient_sum[tri[i]] += grad_p_tri * area
            nodal_area_sum[tri[i]] += area

    # Finalize the average by dividing. Add a small epsilon to avoid division by zero.
    # Use np.newaxis to make the dimensions compatible for broadcasting (N, 2) / (N, 1)
    final_gradient = nodal_gradient_sum / (nodal_area_sum[:, np.newaxis] + 1e-12)
    
    return final_gradient

# ----------------------------------------------------------------------

# --- Script to test a single projection step ---
OUTER_BOUNDARY_MARKER = 1
INNER_BOUNDARY_MARKER = 2

OUTER_BOUNDARY_VALUE = 1.0
INNER_BOUNDARY_VALUE = 0.0

# 1. Load the mesh
NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
triangles = readEle(ELE_FILEPATH)
N = nodes_coords.shape[0]
# M = build_mass_matrix(nodes_coords, triangles)

# --- Create a known velocity field u* with known divergence ---
# Let's create a simple radial field: u_star = [x, y]
# The analytical divergence is ∇·u* = ∂x/∂x + ∂y/∂y = 1 + 1 = 2
u_star = np.zeros((N, 2))
for i in range(N):
    x, y = nodes_coords[i]
    u_star[i, 0] = x
    u_star[i, 1] = y

# --- Build the pressure system ---
A_stiffness, _ = buildFemSystem(nodes_coords, triangles, g_source=0.0)
A_p = A_stiffness.copy()
# Pin pressure at node 0
A_p[0, :] = 0.0
A_p[0, 0] = 1.0

# --- Perform ONE projection step ---
DT = 0.01  # Arbitrary DT for the equations
rho = 1.0

# 1. Calculate divergence of our known field
div_u_star = calculate_divergence_simple(nodes_coords, triangles, u_star)

# 2. Solve for pressure
b_p = -(rho / DT) * div_u_star
b_p[0] = 0.0  # Apply the pinned BC
p = np.linalg.solve(A_p, b_p)

# 3. Project to get the final velocity u
grad_p = calculate_gradient_simple(nodes_coords, triangles, p)
u_final = u_star - DT * grad_p

# 4. Verify: Calculate the divergence of the final field
final_div = calculate_divergence_simple(nodes_coords, triangles, u_final)

# --- Print the results ---
print("--- Single Projection Test ---")
print(f"Max divergence of initial field u*: {np.max(np.abs(div_u_star)):.2e} (Analytical value is 2.0)")
print(f"Max divergence of final field u:   {np.max(np.abs(final_div)):.2e} (Should be near zero)")

# --- Plot the fields ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)

# Plot Div(u*)
vmax1 = np.max(np.abs(div_u_star))
axes[0].tripcolor(triang, div_u_star, shading='gouraud', cmap='coolwarm', vmin=-vmax1, vmax=vmax1)
axes[0].set_title("Divergence of Initial Field (u*)")
axes[0].set_aspect('equal')

# Plot Pressure
axes[1].tripcolor(triang, p, shading='gouraud', cmap='viridis')
axes[1].set_title("Resulting Pressure Field (p)")
axes[1].set_aspect('equal')

# Plot Final Div(u)
vmax2 = np.max(np.abs(final_div)) if np.max(np.abs(final_div)) > 1e-12 else 1e-12
axes[2].tripcolor(triang, final_div, shading='gouraud', cmap='coolwarm', vmin=-vmax2, vmax=vmax2)
axes[2].set_title("Divergence of Final Field (u)")
axes[2].set_aspect('equal')

plt.tight_layout()
plt.show()
