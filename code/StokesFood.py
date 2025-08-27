from os import wait
import numpy as np
import numpy.linalg as la
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List, Set
from jax.experimental import sparse
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
import matplotlib.animation as animation
import scipy.linalg as sla
jax.config.update("jax_enable_x64", True)
from scipy.spatial import KDTree
import numpy as np

# ==============================================================================
# --- CONSTANTS ---
# ==============================================================================

NODE_FILEPATH = "./mesh/mesh.1.node"
ELE_FILEPATH = "./mesh/mesh.1.ele"
POLY_FILEPATH = "./mesh/mesh.1.poly"


# --- Boundary Conditions ---
OUTER_BOUNDARY_MARKER = 1
INNER_BOUNDARY_MARKER = 2
OUTER_BOUNDARY_VALUE = [0.0,0.0]
B1 = -2.0
B2 = 0.0

# --- Domain and Physics Parameters ---
L = 1.0  # Domain width
H = 1.0  # Domain height
D = 1e-3
v = 1.0 # Kinematic viscosity
tol = 1e-6 # Tolerance for coordinate comparisons

# --- Simulation Parameters ---
DT = 0.01
STEPS = 6000

# --- Visualization ---
PLOT_GRID_DENSITY = 100
FIXED_VMAX = 2.0 # Fixed max value for color bar

# --- Squirmer feeding parameters ---
SQUIRMER_RADIUS = 0.25
CAPTURE_RADIUS = SQUIRMER_RADIUS + 0.03 
SQUIRMER_CENTER = np.array([0.5, 0.5])

# ==============================================================================
# --- FUNCTIONS AND HELPER-FUNCTIONS ---
# ==============================================================================

def readNode(filepath):
    """
    Prepares the data for processing
    Returns:
        nodes_coords: Array over the coordinates (x, y) of
        nodes with the index being the node number()
    """
    with open(filepath, "r") as nodeFile:
        header = nodeFile.readline().strip().split()
        number_of_nodes = header[0]
        nodes_coords = np.zeros((int(number_of_nodes), 2), dtype=np.float64)
        nodes_boundary_markers = np.zeros(int(number_of_nodes), dtype=np.int32)

        for i in range(int(number_of_nodes)):
            line = nodeFile.readline().strip().split()
            index = int(line[0]) - 1

            nodes_coords[index, 0] = float(line[1])
            nodes_coords[index, 1] = float(line[2])

            # boundary marker
            if int(line[3]) != 0:
                nodes_boundary_markers[index] = int(line[3])
        
        return nodes_coords, nodes_boundary_markers



def readEle(filepath):
    with open(filepath, "r") as eleFile:
        header = eleFile.readline().strip().split()
        number_of_triangles = int(header[0])
        nodes_per_triangle = int(header[1])
        elements = np.zeros((number_of_triangles, 3), dtype=np.int32)

        for i in range(int(number_of_triangles)):
            line = eleFile.readline().strip().split()

            elements[i, 0] = int(line[1]) - 1
            elements[i, 1] = int(line[2]) - 1
            elements[i, 2] = int(line[3]) - 1
        return elements


def buildStiffnessMatrix(nodes, triangles, g_source=1.0):
    N = nodes.shape[0]
    AMatrix = np.zeros((N, N), dtype=np.float64)
    BVector = np.zeros(N, dtype=np.float64)

    for tri in triangles:
        p1_idx, p2_idx, p3_idx = tri
        p1, p2, p3 = nodes[p1_idx], nodes[p2_idx], nodes[p3_idx]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Calculate the signed determinant (2 * signed area)
        ADet = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

        if abs(ADet) < 1e-14:
            print("skip!")
            continue

        y_diffs = [y2 - y3, y3 - y1, y1 - y2]
        x_diffs = [x3 - x2, x1 - x3, x2 - x1]

        for i in range(3):
            for j in range(3):
                numerator = (y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j])
                denominator = 2 * abs(ADet)
                integral_val = numerator / denominator
                
                AMatrix[tri[i], tri[j]] += integral_val

    return AMatrix, -BVector

def calculate_divergence(nodes, triangles, u_star):
    """
    u_star : shape (N, 2)
    returns: (N,) nodal divergence (area–weighted average)
    """
    N        = nodes.shape[0]
    div_sum  = np.zeros(N)
    area_sum = np.zeros(N)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]

        det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(det) < 1e-14:
            continue
            
        inv2A = 1.0 / det
        area  = 0.5 * abs(det)

        # divergence calculation 
        ux1, uy1 = u_star[p1]
        ux2, uy2 = u_star[p2]
        ux3, uy3 = u_star[p3]
        
        # ∑_i=1^3 u_xi ∂Ni/∂x × 2A
        d_ux_dx = (ux1*(y2-y3) + ux2*(y3-y1) + ux3*(y1-y2)) * inv2A
        # ∑_i=1^3 u_yi ∂Ni/∂y × 2A
        d_uy_dy = (uy1*(x3-x2) + uy2*(x1-x3) + uy3*(x2-x1)) * inv2A
        div_tri = d_ux_dx + d_uy_dy

        # lumping sheme         
        lump = div_tri * (area / 3.0)
        for p in tri:
            div_sum[p]  += lump
            area_sum[p] += area / 3.0
    
    # final nodal divergence / total accumulated area (and small number to prevent division by zero)       
    return div_sum / (area_sum + 1e-12)



def find_boundary_pairs(nodes_coords, L=1.0, tol=1e-6):
    # 1. Get the indices of all nodes on the left and right boundaries
    left_indices = np.where(np.abs(nodes_coords[:, 0]) < tol)[0]
    right_indices = np.where(np.abs(nodes_coords[:, 0] - L) < tol)[0]

    # Return early if one of the boundaries has no nodes
    if len(left_indices) == 0 or len(right_indices) == 0:
        print("Warning: One or both boundaries have no nodes.")
        return []

    # 2. Get the coordinates of the boundary nodes
    left_coords = nodes_coords[left_indices]
    right_coords = nodes_coords[right_indices]

    # 3. Use a KDTree for a very fast nearest-neighbor search.
    # We build the tree using the y-coordinates of the right-boundary nodes.
    # KDTree needs a 2D array, so we reshape the y-coordinates.
    right_y_coords_for_tree = right_coords[:, 1].reshape(-1, 1)
    kdtree = KDTree(right_y_coords_for_tree)

    pairs = []
    # 4. For each node on the left boundary, find its closest partner on the right
    for i, left_idx in enumerate(left_indices):
        left_y = left_coords[i, 1]

        # Query the tree to find the closest y-value on the right boundary.
        dist, right_array_idx = kdtree.query([left_y])

        # The `right_array_idx` corresponds to the position in the `right_indices`
        # array. We use it to get the original node index from the main `nodes_coords`.
        right_node_idx = right_indices[right_array_idx]

        pairs.append((left_idx, right_node_idx))

    return pairs


def apply_periodic_bc(A, pairs):
    """
    Modifies the system matrix A to enforce periodic boundary conditions
    using the PENALTY METHOD, which preserves symmetry.
    """
    # The penalty should be a large number, e.g., 1e10 or 1e12
    penalty = 1.0e10

    for master_idx, slave_idx in pairs:
        # Add the penalty to the diagonal elements
        A[master_idx, master_idx] += penalty
        A[slave_idx, slave_idx] += penalty   

        # Subtract the penalty from the off-diagonal elements
        A[master_idx, slave_idx] -= penalty
        A[slave_idx, master_idx] -= penalty


def calculate_gradiant(nodes, triangles, p_scalar):
    """ 
    Calculates nodal gradient using a 1/3 area lumping scheme.
    """
    N = nodes.shape[0]
    grad_px_sum = np.zeros(N)
    grad_py_sum = np.zeros(N)
    area_sum = np.zeros(N)
    for tri in triangles:
        p_nodes = nodes[tri]
        p_local = p_scalar[tri]
        det = p_nodes[0,0]*(p_nodes[1,1]-p_nodes[2,1]) + \
              p_nodes[1,0]*(p_nodes[2,1]-p_nodes[0,1]) + \
              p_nodes[2,0]*(p_nodes[0,1]-p_nodes[1,1])
              
        if abs(det) < 1e-14: continue
        
        inv2A = 1.0 / det
        area = 0.5 * abs(det)

        # constant gradient of the scalar field 
        grads = np.array([
            [p_nodes[1,1]-p_nodes[2,1], p_nodes[2,0]-p_nodes[1,0]],
            [p_nodes[2,1]-p_nodes[0,1], p_nodes[0,0]-p_nodes[2,0]],
            [p_nodes[0,1]-p_nodes[1,1], p_nodes[1,0]-p_nodes[0,0]]
        ], dtype=np.float64) * inv2A
        
        gp_element = grads.T @ p_local
        
        # lumping scheme, assuming each node has equal  (to avoid the integration)
        lump_px = gp_element[0] * (area / 3.0)
        lump_py = gp_element[1] * (area / 3.0)
        area_lump = area / 3.0
        for i in range(3):
            node_idx = tri[i]
            grad_px_sum[node_idx] += lump_px
            grad_py_sum[node_idx] += lump_py
            area_sum[node_idx] += area_lump
          
    grad_px = grad_px_sum / (area_sum + 1e-12)
    grad_py = grad_py_sum / (area_sum + 1e-12)
    return grad_px, grad_py


def buildLumpedMassMatrix(nodes_coords, triangles):
    """
    Builds a diagonal (lumped) mass matrix M_L.
    Each diagonal entry M_ii is the sum of 1/3 of the area of all
    triangles connected to node i.
    Returns a 1D array representing the diagonal.
    """
    N = nodes_coords.shape[0]
    lumped_mass = np.zeros(N)
    for tri in triangles:
        p_nodes = nodes_coords[tri]
        det = p_nodes[0,0]*(p_nodes[1,1]-p_nodes[2,1]) + \
              p_nodes[1,0]*(p_nodes[2,1]-p_nodes[0,1]) + \
              p_nodes[2,0]*(p_nodes[0,1]-p_nodes[1,1])
        
        area = 0.5 * abs(det)
        for i in range(3):
            lumped_mass[tri[i]] += area / 3.0
    return lumped_mass

def build_mass_and_convection(nodes, triangles, u):
    N   = nodes.shape[0]
    M   = np.zeros((N, N))
    C   = np.zeros((N, N))

    for tri in triangles:
        idx      = tri
        coords   = nodes[idx]
        x1,y1, x2,y2, x3,y3 = coords.flatten()
        det      = x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)
        if abs(det) < 1e-14:  continue
        area     = 0.5*abs(det)

        # --- local mass (lumped formula) ---
        for i in range(3):
            for j in range(3):
                M[idx[i], idx[j]] += (area/12.0)*(1.0 if i!=j else 2.0)

        # --- local convection ---
        u_c   = u[idx].mean(axis=0) 
        grads = np.array([[y2-y3, x3-x2],
                          [y3-y1, x1-x3],
                          [y1-y2, x2-x1]]) / (2*abs(det)) 
        for i in range(3):
            for j in range(3):
                C[idx[i], idx[j]] += (area/3) * np.dot(u_c, grads[j])
    return M, C

def makeDirBCU(u):
    u[wall_node_indices] = OUTER_BOUNDARY_VALUE
    # --- SQUIRMER BOUNDARY CONDITION ---
    center_x, center_y = 0.5, 0.5

    for idx in inner_boundary_indices:
        # 1. Get the coordinates and calculate the angle (theta)
        node_x, node_y = nodes_coords[idx]
        radius_vec_x = node_x - center_x
        radius_vec_y = node_y - center_y
        theta = np.arctan2(radius_vec_y, radius_vec_x)
#
        # 2. Calculate the tangential velocity on the surface using the squirmer model
        #    This is the 2D equivalent of the formula in the article.
        v_tangential = B1 * np.sin(theta) + B2 * np.sin(2 * theta)
       
        # 3. Get the unit tangential vector at this point
        unit_tangential_x = -np.sin(theta)
        unit_tangential_y =  np.cos(theta)
       
        # 4. Set the velocity of the node
        u[idx, 0] = v_tangential * unit_tangential_x
        u[idx, 1] = v_tangential * unit_tangential_y 

def makePerBCU(u):
    for master_idx, slave_idx in pairs:
        u[slave_idx] = u[master_idx]
# ==============================================================================
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================

# --- Load Mesh Data ---
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
triangles = readEle(ELE_FILEPATH)
N = nodes_coords.shape[0]

# --- Identify Periodic Boundary Nodes ---
all_pairs = find_boundary_pairs(nodes_coords, L=1.0)
for pair in all_pairs:
    xcoord = nodes_coords[pair[0]]
    ycoord = nodes_coords[pair[1]]
    print(f"{xcoord}, {ycoord}")

# Filter out corner/wall nodes from periodic pairs
non_wall_pairs = []
for master_idx, slave_idx in all_pairs:
    master_y = nodes_coords[master_idx, 1]
    if not (np.abs(master_y - 0.0) < tol or np.abs(master_y - H) < tol):
        non_wall_pairs.append((master_idx, slave_idx))


print(f"Original periodic pairs: {len(all_pairs)}, Filtered pairs (excluding walls): {len(non_wall_pairs)}")
pairs = non_wall_pairs


# --- Identify Dirichlet (Fixed-Value) Boundary Nodes ---
wall_node_indices = np.where(np.isclose(nodes_coords[:, 1], 0.0, atol=tol) | np.isclose(nodes_coords[:, 1], H, atol=tol))[0]
inner_boundary_indices = np.where(nodes_boundary_markers == INNER_BOUNDARY_MARKER)[0]
dirichlet_node_indices = np.union1d(wall_node_indices, inner_boundary_indices)
interior = np.setdiff1d(np.arange(N), dirichlet_node_indices)

# --- Build Base Stiffness Matrix (Laplacian) ---
A_stiffness, _ = buildStiffnessMatrix(nodes_coords, triangles, g_source=0.0)
M_lumped_diag = buildLumpedMassMatrix(nodes_coords, triangles)

# --- Setup Viscosity Matrix (for Velocity) ---
A_visc = np.eye(N) + DT * v * A_stiffness
# Enforce u=value on walls and inner body
A_visc[dirichlet_node_indices, :] = 0.0
A_visc[:, dirichlet_node_indices] = 0.0
A_visc[dirichlet_node_indices, dirichlet_node_indices] = 1.0

# --- Setup Pressure Matrix (for Pressure Correction) ---
A_pressure = A_stiffness / (M_lumped_diag[:, np.newaxis] + 1e-12)
apply_periodic_bc(A_pressure, pairs) 

# --- Initial Velocity Field and Body Forces ---
u = np.zeros((N, 2))
makeDirBCU(u)

b_force = np.zeros((N, 2))
b_force[:, 0] = 0.0


# --- Setup2 Visualization ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
grid_x, grid_y = np.meshgrid(
    np.linspace(0.01, 0.99, PLOT_GRID_DENSITY),
    np.linspace(0.01, 0.99, PLOT_GRID_DENSITY)
)
colorbar = None


# --- Initialize Passive Tracers ("Paint") ---
grid_density = 25 
xx = np.linspace(0.05, L - 0.05, grid_density)
yy = np.linspace(0.05, H - 0.05, grid_density)
grid_x, grid_y = np.meshgrid(xx, yy)
all_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

# Filter out points inside the cylinder
distances = np.linalg.norm(all_points - SQUIRMER_CENTER, axis=1)
tracer_points = all_points[distances > SQUIRMER_RADIUS]
num_tracers = tracer_points.shape[0]

# --- Initialize tracer status (0 = purple/uneaten, 1 = red/eaten) ---
tracer_status = np.zeros(num_tracers, dtype=int)
tracer_colors = np.array(['blue', 'red'])

print(f"Initialized {num_tracers} tracer particles.")



print("\nStarting simulation...")
for step in range(STEPS):
    # --- Step 1: Tentative Velocity ---
    # We solve (I + Δt*ν*A)u* = u^n + Δt*F
    rhs_x = u[:, 0] + DT * b_force[:, 0]
    rhs_y = u[:, 1] + DT * b_force[:, 1]

    u_star = np.zeros((N,2))
    u_star[:,0] = np.linalg.solve(A_visc, rhs_x)
    u_star[:,1] = np.linalg.solve(A_visc, rhs_y)
    makePerBCU(u_star) 
    makeDirBCU(u_star)

    # --- Step 2: Pressure Correction (Poisson Equation) ---
    # a) Calculate the FVM-style divergence of the tentative velocity.
    div_u_star = calculate_divergence(nodes_coords, triangles, u_star)

    # The lumped mass matrix M_lumped_diag bridges the FEM and FVM operators.
    b_p = -(1.0 / DT) * div_u_star
    p = np.linalg.solve(A_pressure, b_p)
    
    # --- Step 3: Velocity Update ---
    # Use the FVM-style gradient for the velocity correction
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p)

    u[:,0] = u_star[:,0] - DT*grad_px
    u[:,1] = u_star[:,1] - DT*grad_py
    makePerBCU(u)
    makeDirBCU(u)

    # Add one more projection to decrease divergence
    div_u = calculate_divergence(nodes_coords, triangles, u)
    b_p2 = -(1.0/DT) * div_u 
    p2 = np.linalg.solve(A_pressure, b_p2)
    grad2_x, grad2_y = calculate_gradiant(nodes_coords, triangles, p2)

    u[interior,0] -= DT * grad2_x[interior]
    u[interior,1] -= DT * grad2_y[interior]

    final_div = calculate_divergence(nodes_coords, triangles, u)

        # --- Step 5: Passive Tracers ---
    interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
    interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])

    tracer_ux = interpolator_x(tracer_points[:, 0], tracer_points[:, 1])
    tracer_uy = interpolator_y(tracer_points[:, 0], tracer_points[:, 1])

    tracer_points[:, 0] += tracer_ux * DT
    tracer_points[:, 1] += tracer_uy * DT
    tracer_points[:, 0] = np.mod(tracer_points[:, 0], L)

    # --- Check for "eaten" tracers and update their status ---
    # Calculate distance from each tracer to the squirmer's center
    distances_to_center = np.linalg.norm(tracer_points - SQUIRMER_CENTER, axis=1)
    # Find indices of points within the capture radius
    eaten_indices = np.where(distances_to_center <= CAPTURE_RADIUS)[0]
    # Update the status of these tracers to 1 (eaten)
    if len(eaten_indices) > 0:
        tracer_status[eaten_indices] = 1
    
    # --- Count eaten vs. uneaten tracers ---
    num_eaten = np.sum(tracer_status)
    num_uneaten = num_tracers - num_eaten

    print(f"Step: {step}, Div(u*): {np.max(np.abs(div_u_star)):.2e}, Final Div(u): {np.max(np.abs(final_div)):.2e}, Eaten (Red): {num_eaten}, Uneaten (Blue): {num_uneaten}")              

    # --- VISUALIZATION ---
    ax.clear()
    
    # Plot velocity magnitude as background
    u_magnitude = np.linalg.norm(u, axis=1)
    tpc = ax.tripcolor(triang, u_magnitude, shading='gouraud', cmap='viridis', vmin=0, vmax=FIXED_VMAX)

    # Plot velocity vectors
    skip_interior = 30
    interior_indices = np.where(nodes_boundary_markers == 0)[0][::skip_interior]
    quiver_indices = np.union1d(inner_boundary_indices, interior_indices)
    ax.quiver(nodes_coords[quiver_indices, 0], nodes_coords[quiver_indices, 1],
                u[quiver_indices, 0], u[quiver_indices, 1],
                color='white', scale=40.0)

    # --- Plot tracer particles with their corresponding color ---
    ax.scatter(tracer_points[:, 0], tracer_points[:, 1], 
                c=tracer_colors[tracer_status], s=20, zorder=5, alpha=0.9)

    if colorbar is None:
        colorbar = fig.colorbar(tpc, ax=ax)
        colorbar.set_label("Velocity Magnitude")
    else:
        colorbar.update_normal(tpc)

    ax.set_aspect('equal')
    ax.set_title(f"Squirmer Flow Field: Step {step}/{STEPS}")
    ax.set_facecolor('black') 
    fig.canvas.draw_idle()
    plt.pause(0.0001)

print("\nSimulation finished")
plt.ioff()
plt.show()

