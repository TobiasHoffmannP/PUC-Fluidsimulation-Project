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

NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"

POLY_FILEPATH = "mesh2.1.poly"

OUTER_BOUNDARY_MARKER = 1

#RIGHT_BOUNDARY_MARKER = 1
#LEFT_BOUNDARY_MARKER = 2
#TOP_BOTTOM_BOUNDARY_MARKER = 3
#INNER_BOUNDARY_MARKER = 4
INNER_BOUNDARY_MARKER = 2


OUTER_BOUNDARY_VALUE = 1.0
#TOP_BOTTOM_BOUNDARY_VALUE = 1.0
INNER_BOUNDARY_VALUE = 1.0


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


def readPoly(path: str):
    with open(path) as f:
        f.readline()  # skip first line
        segmentsHeader = f.readline().strip().split()
        # print(f"{segmentsHeader}")
        Nsegs = int(segmentsHeader[0])
        segments = np.zeros((Nsegs, 2), dtype=int)
        boundaryMarkers = np.zeros(Nsegs, dtype=int)

        for _ in range(Nsegs):
            parts = f.readline().strip().split()
            id = int(parts[0]) - 1
            segments[id] = (int(parts[1]) - 1, int(parts[2]) - 1)

            if len(parts) > 3:  # check if boundary markers are present
                boundaryMarkers[id] = int(parts[3])

        # print(f"{segments}")
        # print(f"{boundaryMarkers}")
        return segments, boundaryMarkers


def buildFemSystem(nodes, triangles, g_source):
    N = nodes.shape[0]
    AMatrix = np.zeros((N,N), dtype=np.float64)
    BVector = np.zeros(N, dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]
        
        ADet = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2
         
        if ADet == 0: 
            continue

        # print(triangles.shape[1])

        y_diffs = [y2 - y3, y3 - y1, y1 - y2]
        x_diffs = [x3 - x2, x1 - x3, x2 - x1]

        for i in range(3):
            for j in range(3):
                # 1/2A [(2i − y3i)(y2j − y3j) + (x3i − x2i)(x3j − x2j)]
                integral_val = (y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j]) / (2.0 * ADet)
                
                # assemble to global matrix (AMatrix)
                # += because it can get values from other triangles as well
                AMatrix[tri[i], tri[j]] += integral_val

        area = 0.5 * ADet
        
        # pyramid over triangle is 1 / 3 of the area
        # bj = ∫Ω g(x, y) ϕj(x, y) dΩ = g * Area / 3

        if callable(g_source):
            g = g_source((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)
        else:
            g = g_source

        sourceIntergralValue = g * (area / 3)
        # print(sourceIntergralValue)

        BVector[p1] += sourceIntergralValue
        BVector[p2] += sourceIntergralValue
        BVector[p3] += sourceIntergralValue

    return AMatrix, BVector


def find_boundary_pairs(nodes_coords, L=1.0, tol=1e-6):

    # get the indices of all nodes on the left and right boundaries
    left_indices = np.where(np.abs(nodes_coords[:, 0]) < tol)[0]
    right_indices = np.where(np.abs(nodes_coords[:, 0] - L) < tol)[0]

    # Return early if one of the boundaries has no nodes
    if len(left_indices) == 0 or len(right_indices) == 0:
        print("Warning: One or both boundaries have no nodes.")
        return []
 
    left_coords = nodes_coords[left_indices]
    right_coords = nodes_coords[right_indices]


    right_y_coords_for_tree = right_coords[:, 1].reshape(-1, 1)
    kdtree = KDTree(right_y_coords_for_tree)

    pairs = []
    for i, left_idx in enumerate(left_indices):
        left_y = left_coords[i, 1]
        
        dist, right_array_idx = kdtree.query([left_y])
        
        right_node_idx = right_indices[right_array_idx]
        
        pairs.append((left_idx, right_node_idx))

    return pairs


def apply_periodic_bc(A, b, pairs):
    #   left set    right set
    for master_idx, slave_idx in pairs:
        # add the slave's row contributions to the master's row
        A[master_idx, :] += A[slave_idx, :]
        b[master_idx] += b[slave_idx]

        # clear the slave's row to create the constraint equation
        A[slave_idx, :] = 0.0

        # enforce the constraint: 1*x_slave - 1*x_master = 0
        A[slave_idx, slave_idx] = 1.0
        A[slave_idx, master_idx] = -1.0
        b[slave_idx] = 0.0    

# TUBE_WALL_MARKER = 3 # A NEW marker for top/bottom walls

# WALL_VALUE = 0.0

def apply_dirchlect_u(u):
    for i in range(N):
        marker = nodes_boundary_markers[i]
        x_coord, y_coord = nodes_coords[i]
        theta = np.arctan2(y_coord, x_coord)

        is_wall = np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol
        is_inner_boundary = marker == INNER_BOUNDARY_MARKER

        if is_wall or is_inner_boundary:
            if is_inner_boundary:
                # Set velocity to (0, 0) on the inner obstacle
                u[i, 0] = np.sin(theta)
                u[i, 1] = np.cos(-theta)
            elif is_wall:
                # Set velocity on the outer walls to zero (no-slip)
                u[i, 0] = 0.0
                u[i, 1] = 0.0
    return u

def apply_periodic_u(u):
    for master, slave in pairs:
        u[slave] = u[master]
    return u

def calculate_divergence(nodes, triangles, u_star):
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

def build_pressure_system(nodes, triangles, source_vec):
    N = nodes.shape[0]
    AMatrix = np.zeros((N, N), dtype=np.float64)
    BVector = np.zeros(N, dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]
        
        ADet = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if ADet == 0: continue

        # Assemble Stiffness Matrix 'A' (this part is unchanged)
        y_diffs = [y2 - y3, y3 - y1, y1 - y2]
        x_diffs = [x3 - x2, x1 - x3, x2 - x1]

        for i in range(3):
            for j in range(3):
                integral_val = (y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j]) / (2.0 * ADet)
                AMatrix[tri[i], tri[j]] += integral_val

        # Assemble Source Vector 'b' from the pre-computed vector
        area = 0.5 * ADet
        
        # Get the source values at the triangle's vertices
        g1 = source_vec[p1]
        g2 = source_vec[p2]
        g3 = source_vec[p3]

        # Calculate contributions using the element mass matrix formula
        BVector[p1] += (area / 12.0) * (2*g1 + g2 + g3)
        BVector[p2] += (area / 12.0) * (g1 + 2*g2 + g3)
        BVector[p3] += (area / 12.0) * (g1 + g2 + 2*g3)

    return AMatrix, BVector

def calculate_gradient(nodes, triangles, p_vector):
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

def build_advection_matrix(nodes, triangles, u):
    N = nodes.shape[0]
    A_adv = np.zeros((N, N), dtype=np.float64)
        

    for tri in triangles:
        p1_idx, p2_idx, p3_idx = tri
        global_indices = [p1_idx, p2_idx, p3_idx]

        x1, y1 = nodes[p1_idx]
        x2, y2 = nodes[p2_idx]
        x3, y3 = nodes[p3_idx]

        u_val1 = u[p1_idx]
        u_val2 = u[p2_idx]
        u_val3 = u[p3_idx]
        u_avg = (u_val1 + u_val2 + u_val3) / 3

        A_element = np.zeros((3, 3), dtype=np.float64)

        A_det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        area = 0.5 * A_det

        y_diffs = [y2 - y3, y3 - y1, y1 - y2]
        x_diffs = [x3 - x2, x1 - x3, x2 - x1]

        for j in range(3):
            """
            A_element[i, j] = -(u_avg * ∇φj) * area/3
            """
            # calculate the components of the gradient of the j-th basis function
            grad_phi_j_x = y_diffs[j] / A_det
            grad_phi_j_y = x_diffs[j] / A_det

            u_dot_grad_phi = u_avg[0] * grad_phi_j_x + u_avg[1] * grad_phi_j_y

            # changed from: column_value = -u_dot_grad_phi * (area / 3.0)
            column_value = u_dot_grad_phi * (area / 3.0)

            A_element[:, j] = column_value

        for i in range(3):
            for j in range(3):
                global_row = global_indices[i]
                global_column = global_indices[j]

                A_adv[global_row, global_column] += A_element[i, j]
                
    return A_adv


H = 1.0
tol = 1e-6
DT = 0.0001
NUM_STEPS = 1000
v = 0.1 # viscosity
rho = 1.0 # density

nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
N = nodes_coords.shape[0]
triangles = readEle(ELE_FILEPATH)


def g_source_fun(x, y):
    return 50 * np.sin(3 * y)

# b = np.zeros((N, 2), dtype=np.float64)

# apply PERIODIC boundary conditions to the spatial matrix.
pairs = find_boundary_pairs(nodes_coords, L=1.0)
filtered_pairs = []
for master_idx, slave_idx in pairs:
    master_y = nodes_coords[master_idx, 1]
    if not (np.abs(master_y - 0.0) < tol or np.abs(master_y - H) < tol):
        filtered_pairs.append((master_idx, slave_idx))


A_base, _ = buildFemSystem(nodes_coords, triangles, g_source=0.0)
# b = np.zeros((N, 2))


b = np.zeros((N, 2))

A_vel = A_base.copy() # A_velocities
apply_periodic_bc(A_vel, np.zeros(N), filtered_pairs) 



A_new = np.eye(N) + v * DT * A_vel


A_p = A_base.copy()
apply_periodic_bc(A_p, np.zeros(N), filtered_pairs)
A_p[0, :] = 0.0
A_p[0, 0] = 1.0


u = np.zeros((N, 2), dtype=np.float64)
u = apply_periodic_u(u)
u = apply_dirchlect_u(u)


p = np.zeros(N)


plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))



for step in range(NUM_STEPS):
    # System matrix for the velocity update
    A_new = np.eye(N) + v * DT * A_vel

    # Right-hand side for the velocity update
    rhs_x = u[:, 0] + DT * b[:, 0] 
    rhs_y = u[:, 1] + DT * b[:, 1]

    
    for i in range(N):
        marker = nodes_boundary_markers[i]
        x_coord = nodes_coords[i, 0]
        y_coord = nodes_coords[i, 1]
        is_wall = np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol
        is_inner = (marker == INNER_BOUNDARY_MARKER)
        
        if is_wall or is_inner:
            # Modify the matrix row to enforce the condition
            A_new[i, :] = 0.0
            A_new[i, i] = 1.0
            
            # Set the corresponding RHS value to the desired boundary value (0 for no-slip)
            rhs_x[i] = 0.0
            rhs_y[i] = 0.0
            if is_inner:
                theta = np.arctan2(y_coord, x_coord)
                rhs_x[i] = np.sin(theta)
                rhs_y[i] = np.cos(-theta)
   

    # viscous solve  
    u_star_x = np.linalg.solve(A_new, rhs_x)
    u_star_y = np.linalg.solve(A_new, rhs_y)
    u_star = np.stack([u_star_x, u_star_y], axis=1)

    # pressure solve
    div = calculate_divergence(nodes_coords, triangles, u_star)
    b_p = (rho / DT) * div 
    b_p[0] = 0.0 # Apply BC to the RHS vector
    p = np.linalg.solve(A_p, b_p)

    # projection
    grad_p = calculate_gradient(nodes_coords, triangles, p)
    u = u_star - DT * grad_p

    
    # apply boundary conditions
    u = apply_dirchlect_u(u)
    u = apply_periodic_u(u)
  

    # plot stuff
    if step > 0: 
        print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Max P: {np.max(np.abs(p)):.2e}")
        ax.clear()

        # velocity Magnitude
        u_magnitude = np.linalg.norm(u, axis=1)
        current_max_u = np.max(u_magnitude)
        vmax = current_max_u if current_max_u > 1e-9 else 0.1
        tpc = ax.tripcolor(triang, u_magnitude, shading="gouraud", cmap="viridis", vmin=0, vmax=vmax)

        # quiver
        skip = 25
        ax.quiver(nodes_coords[::skip, 0], nodes_coords[::skip, 1], u[::skip, 0], u[::skip, 1],
                  color='white', scale=vmax * 20, width=0.004)

        # streamlines
        interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
        interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])
        u_x_grid = interpolator_x(grid_x, grid_y)
        u_y_grid = interpolator_y(grid_x, grid_y)
        ax.streamplot(grid_x, grid_y, u_x_grid, u_y_grid, color='black', linewidth=1, density=0.7)

        ax.set_aspect("equal")
        ax.set_title(f"Time Evolution: Step {step}/{NUM_STEPS}")

        fig.canvas.draw_idle()
        plt.pause(0.0001)


print("Animation finished.")
plt.ioff()
skip = 25
# Redraw final state for a clean plot with a colorbar
ax.clear()
u_magnitude = np.linalg.norm(u, axis=1)
vmax = np.max(u_magnitude)
tpc = ax.tripcolor(triang, u_magnitude, shading="gouraud", cmap="viridis", vmin=0, vmax=vmax)
fig.colorbar(tpc, ax=ax, label="Velocity Magnitude")
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
