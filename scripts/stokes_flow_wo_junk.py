from os import wait
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List, Set
from jax.experimental import sparse
import matplotlib.tri as mtri
import collections
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
jax.config.update("jax_enable_x64", True)

# read the .ele, .node, .poly files
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


def calculate_divergence_simple(nodes, triangles, u_star):
    N = nodes.shape[0]
    div_sum = np.zeros(N, dtype=np.float64)
    area_sum = np.zeros(N, dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1, x2, y2, x3, y3 = nodes[p1,0], nodes[p1,1], nodes[p2,0], nodes[p2,1], nodes[p3,0], nodes[p3,1]
        
        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(ADet) < 1e-14: continue
        
        area = 0.5 * abs(ADet)
        
        ux1, uy1, ux2, uy2, ux3, uy3 = u_star[p1,0], u_star[p1,1], u_star[p2,0], u_star[p2,1], u_star[p3,0], u_star[p3,1]
        
        div_tri = ((ux1*(y2-y3) + ux2*(y3-y1) + ux3*(y1-y2)) + (uy1*(x3-x2) + uy2*(x1-x3) + uy3*(x2-x1))) / ADet

        # add the triangle's divergence to its vertices, weighted by area
        for i in range(3):
            div_sum[tri[i]] += div_tri * area
            area_sum[tri[i]] += area

    # finalize the average, avoiding division by zero
    return div_sum / (area_sum + 1e-12)

    

def calculate_gradient_simple(nodes, triangles, p_vector):
    N = nodes.shape[0]
    grad_sum = np.zeros((N, 2), dtype=np.float64)
    area_sum = np.zeros(N, dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1, x2, y2, x3, y3 = nodes[p1,0], nodes[p1,1], nodes[p2,0], nodes[p2,1], nodes[p3,0], nodes[p3,1]
        
        ADet = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(ADet) < 1e-14: continue
        
        area = 0.5 * abs(ADet)
        
        p1_val, p2_val, p3_val = p_vector[p1], p_vector[p2], p_vector[p3]
        
        grad_p_x = (p1_val*(y2-y3) + p2_val*(y3-y1) + p3_val*(y1-y2)) / ADet
        grad_p_y = (p1_val*(x3-x2) + p2_val*(x1-x3) + p3_val*(x2-x1)) / ADet
        grad_p_tri = np.array([grad_p_x, grad_p_y])
        
        for i in range(3):
            grad_sum[tri[i]] += grad_p_tri * area
            area_sum[tri[i]] += area

    # finalize the average, avoiding division by zero
    return grad_sum / (area_sum[:, np.newaxis] + 1e-12)


def apply_dirichlet_to_all_boundaries(A, rhs_x, rhs_y, nodes_coords, nodes_boundary_markers, H=1.0, tol=1e-6):
    N = nodes_coords.shape[0]
    
    for i in range(N):
        # get the node's properties
        x_coord, y_coord = nodes_coords[i]
        marker = nodes_boundary_markers[i]

        # check if the node is on any boundary
        is_outer_wall = (
            np.abs(x_coord - 0.0) < tol or
            np.abs(x_coord - 1.0) < tol or
            np.abs(y_coord - 0.0) < tol or
            np.abs(y_coord - H) < tol
        )
        is_inner_wall = (marker == INNER_BOUNDARY_MARKER)

        # if the node is on any wall, apply the Dirichlet condition
        if is_outer_wall or is_inner_wall:
            # modify the matrix row to enforce the condition (e.g., 1 * u_i = 0)
            A[i, :] = 0.0
            A[:, i] = 0.0 
            A[i, i] = 1.0
            
            # set the corresponding RHS value to the boundary value (0 for no-slip)
            rhs_x[i] = 0.0
            rhs_y[i] = 0.0

def apply_dirichlet_u_to_all_walls(u, nodes_coords, nodes_boundary_markers, H=1.0, tol=1e-6):
    for i in range(len(nodes_coords)):
        x_coord, y_coord = nodes_coords[i]
        marker = nodes_boundary_markers[i]

        is_outer_wall = (
            np.abs(x_coord - 0.0) < tol or np.abs(x_coord - 1.0) < tol or
            np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol
        )
        is_inner_wall = (marker == INNER_BOUNDARY_MARKER)

        if is_outer_wall or is_inner_wall:
            u[i, 0] = 0.0
            u[i, 1] = 0.0

    return u

NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.poly"

OUTER_BOUNDARY_MARKER = 1
INNER_BOUNDARY_MARKER = 2

OUTER_BOUNDARY_VALUE = 1.0
INNER_BOUNDARY_VALUE = 0.0

# --- Parameters ---
H = 1.0
tol = 1e-6
DT = 0.001  # --- CHANGE: A smaller, more stable timestep ---
NUM_STEPS = 500 
v = 1.0      
rho = 1.0

# --- Load Mesh ---
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
N = nodes_coords.shape[0]
triangles = readEle(ELE_FILEPATH)

# --- Setup ---
b = np.zeros((N, 2))
b[:, 0] = 0.1 

# Build the base stiffness matrix
A_stiffness, _ = buildFemSystem(nodes_coords, triangles, g_source=0.0)

# Build the CONSTANT system matrix for the Stokes velocity solve
A_new = np.eye(N) + v * DT * A_stiffness
# Apply Dirichlet BCs to the velocity system matrix using the corrected function
dummy_rhs = np.zeros(N) 
apply_dirichlet_to_all_boundaries(A_new, dummy_rhs, dummy_rhs, nodes_coords, nodes_boundary_markers)


# --- CHANGE: Corrected pressure matrix setup ---
A_p = A_stiffness.copy()
# Find the first interior node to pin the pressure (more stable)
try:
    interior_node_idx = np.where(nodes_boundary_markers == 0)[0][0]
except IndexError:
    # If no interior nodes, fall back to node 0 (less ideal)
    interior_node_idx = 0 
print(f"Pinning pressure at interior node: {interior_node_idx}")

# Symmetrically pin the pressure at the interior node
A_p[interior_node_idx, :] = 0.0
A_p[:, interior_node_idx] = 0.0 # Keep it symmetric
A_p[interior_node_idx, interior_node_idx] = 1.0


# --- Initial Conditions & Plotting Setup ---
u = np.zeros((N, 2), dtype=np.float64)
p = np.zeros(N)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
grid_x, grid_y = np.meshgrid(np.linspace(0.01, 0.99, 100), np.linspace(0.01, 0.99, 100))
colorbar = None

# --- Main Simulation Loop ---
for step in range(1, NUM_STEPS + 1):
    # --- Velocity Solve ---
    rhs_x = u[:, 0] + DT * b[:, 0] 
    rhs_y = u[:, 1] + DT * b[:, 1]
    
    # Set the RHS for the boundary nodes to 0
    # This must be done since A_new was already modified
    for i in range(N):
        if nodes_boundary_markers[i] != 0: # A simpler way to find all boundary nodes
            rhs_x[i] = 0.0
            rhs_y[i] = 0.0
            
    # Solve for intermediate velocity
    u_star_x = np.linalg.solve(A_new, rhs_x)
    u_star_y = np.linalg.solve(A_new, rhs_y)
    u_star = np.stack([u_star_x, u_star_y], axis=1)

    # --- Pressure Solve ---
    div = calculate_divergence_simple(nodes_coords, triangles, u_star)
    b_p = -(rho / DT) * div 
    
    # --- CHANGE: Apply the pressure compatibility condition ---
    b_p -= b_p.mean()
    
    b_p[interior_node_idx] = 0.0 # Enforce the pinned pressure on the RHS
    p = np.linalg.solve(A_p, b_p)

    # --- Projection ---
    grad_p = calculate_gradient_simple(nodes_coords, triangles, p)
    u = u_star - DT * grad_p
    
    # Verification
    final_div = calculate_divergence_simple(nodes_coords, triangles, u)

    # Strongly enforce BCs on the final velocity
    u = apply_dirichlet_u_to_all_walls(u, nodes_coords, nodes_boundary_markers)

    # --- Plotting & Output ---
    if step % 5 == 0:  
        print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Max P: {np.max(np.abs(p)):.2e}, Div(u*): {np.max(np.abs(div)):.2e}, Final Div(u): {np.max(np.abs(final_div)):.2e}")        
        ax.clear()

        u_magnitude = np.linalg.norm(u, axis=1)
        
        FIXED_VMAX = 0.1 
        tpc = ax.tripcolor(triang, u_magnitude, shading='gouraud', cmap='viridis', vmin=0, vmax=FIXED_VMAX)

        # streamliens
        interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
        interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])
        u_x_grid = interpolator_x(grid_x, grid_y)
        u_y_grid = interpolator_y(grid_x, grid_y)
        ax.streamplot(grid_x, grid_y, u_x_grid, u_y_grid, color='black', linewidth=1, density=0.7)

        if colorbar is None:
            colorbar = fig.colorbar(tpc, ax=ax)
        else:
            colorbar.update_normal(tpc) 
        colorbar.set_label("Velocity Magnitude")

        ax.set_aspect('equal')
        ax.set_title(f"Time Evolution: Step {step}/{NUM_STEPS}")
        fig.canvas.draw_idle()
        plt.pause(0.01)

print("DONE")
