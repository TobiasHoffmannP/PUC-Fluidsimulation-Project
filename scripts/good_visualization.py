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
from scipy.interpolate import griddata
jax.config.update("jax_enable_x64", True)

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
                # The formula for the element stiffness is (1 / (4 * Area)) * (...)
                # Since Area = 0.5 * abs(ADet), then 4 * Area = 2 * abs(ADet).
                numerator = (y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j])
                denominator = 2 * abs(ADet)
                integral_val = numerator / denominator
                
                AMatrix[tri[i], tri[j]] += integral_val

    # The g_source part is for the right-hand side vector, which we are not using for A_pressure.
    # It can be ignored for this debugging process.
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
            
        # Use the SIGNED determinant for calculating derivatives.
        inv2A = 1.0 / det
        # Use the ABSOLUTE area for physical lumping.
        area  = 0.5 * abs(det)

        ux1, uy1 = u_star[p1]
        ux2, uy2 = u_star[p2]
        ux3, uy3 = u_star[p3]
 
        d_ux_dx = (ux1*(y2-y3) + ux2*(y3-y1) + ux3*(y1-y2)) * inv2A
        d_uy_dy = (uy1*(x3-x2) + uy2*(x1-x3) + uy3*(x2-x1)) * inv2A
        div_tri = d_ux_dx + d_uy_dy

        # Your lumping scheme was also correct.
        lump = div_tri * (area / 3.0)
        for p in tri:
            div_sum[p]  += lump
            area_sum[p] += area / 3.0
            
    return div_sum / (area_sum + 1e-12)



def find_boundary_pairs(nodes_coords, L=1.0, tol=1e-6):
    # Use np.isclose for robust floating-point comparison
    left_indices = np.where(np.isclose(nodes_coords[:, 0], 0.0, atol=tol))[0]
    right_indices = np.where(np.isclose(nodes_coords[:, 0], L, atol=tol))[0]

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
        # It returns the distance and the index within the `right_y_coords_for_tree` array.
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


def g_source_fun(x, y):
    return 50 * np.sin(3 * y)


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
        
        
        # Use the SIGNED determinant for calculating derivatives.
        inv2A = 1.0 / det
        # Use the ABSOLUTE area for physical lumping.
        area = 0.5 * abs(det)
        
        # This part of your original code was correct.
        grads = np.array([
            [p_nodes[1,1]-p_nodes[2,1], p_nodes[2,0]-p_nodes[1,0]],
            [p_nodes[2,1]-p_nodes[0,1], p_nodes[0,0]-p_nodes[2,0]],
            [p_nodes[0,1]-p_nodes[1,1], p_nodes[1,0]-p_nodes[0,0]]
        ], dtype=np.float64) * inv2A
        
        gp_element = grads.T @ p_local
        
        # Your lumping scheme was also correct.
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

def calculate_consistent_rhs(nodes_coords, triangles, u_star):
    """
    Calculates the RHS for the pressure Poisson equation in a way that is
    consistent with the FEM stiffness matrix `A`.
    
    Computes b_i = - ∫ ∇φ_i ⋅ u* dV for all nodes i.
    """
    N = nodes_coords.shape[0]
    rhs = np.zeros(N)

    for tri in triangles:
        # Get coordinates and local velocity values
        p_nodes = nodes_coords[tri]
        u_local = u_star[tri] # Shape (3, 2)

        # Element-constant gradient of basis functions (∇φ_1, ∇φ_2, ∇φ_3)
        det = p_nodes[0,0]*(p_nodes[1,1]-p_nodes[2,1]) + \
              p_nodes[1,0]*(p_nodes[2,1]-p_nodes[0,1]) + \
              p_nodes[2,0]*(p_nodes[0,1]-p_nodes[1,1])
        if abs(det) < 1e-14: continue
        area = 0.5 * abs(det)
        
        grad_phi_matrix = (1.0 / det) * np.array([
            [p_nodes[1,1]-p_nodes[2,1], p_nodes[2,0]-p_nodes[1,0]],
            [p_nodes[2,1]-p_nodes[0,1], p_nodes[0,0]-p_nodes[2,0]],
            [p_nodes[0,1]-p_nodes[1,1], p_nodes[1,0]-p_nodes[0,0]]
        ]) # Shape (3, 2)

        # Approximate u* on the element as the average of the nodal velocities
        u_avg_on_element = np.mean(u_local, axis=0) # Shape (2,)

        # Calculate integral for each of the 3 local nodes (i=0,1,2)
        for i in range(3):
            # The integral is approx. (∇φ_i ⋅ u_avg) * Area
            integral = np.dot(grad_phi_matrix[i], u_avg_on_element) * area
            
            # Assemble into the global RHS vector
            # The formula has a minus sign, so we subtract
            rhs[tri[i]] -= integral
            
    return rhs

def calculate_vorticity(nodes, triangles, u):
    """Calculates nodal vorticity (curl of the velocity field)."""
    N = nodes.shape[0]
    vorticity_sum = np.zeros(N)
    area_sum = np.zeros(N)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]

        det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if abs(det) < 1e-14:
            continue
        
        inv2A = 1.0 / det
        area = 0.5 * abs(det)

        ux1, uy1 = u[p1]
        ux2, uy2 = u[p2]
        ux3, uy3 = u[p3]

        # Element-constant derivatives
        d_uy_dx = (uy1 * (y2 - y3) + uy2 * (y3 - y1) + uy3 * (y1 - y2)) * inv2A
        d_ux_dy = (ux1 * (x3 - x2) + ux2 * (x1 - x3) + ux3 * (x2 - x1)) * inv2A
        
        vorticity_tri = d_uy_dx - d_ux_dy

        # Lump to nodes
        lump = vorticity_tri * (area / 3.0)
        for p_idx in tri:
            vorticity_sum[p_idx] += lump
            area_sum[p_idx] += area / 3.0
            
    return vorticity_sum / (area_sum + 1e-12)


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
        u_c   = u[idx].mean(axis=0)          # centroid velocity (2,)
        grads = np.array([[y2-y3, x3-x2],
                          [y3-y1, x1-x3],
                          [y1-y2, x2-x1]]) / (2*abs(det))   # ∇φ_j rows
        for i in range(3):
            for j in range(3):
                C[idx[i], idx[j]] += (area/3) * np.dot(u_c, grads[j])
                # 0.3333 is ∫φ_i over K  (=area/3) divided again by area (because of mean φ_i)
    return M, C


# ==============================================================================
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================

## 1. Constants and Parameters
# --- File Paths ---
#NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh_fine.1.node"
#ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh_fine.1.ele"
#POLY_FILEPATH = "/home/tobias/FluidMixing/resources/mesh_fine.1.poly"

NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh5.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh5.1.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/resources/mesh5.1.poly"


# --- Boundary Conditions ---
OUTER_BOUNDARY_MARKER = 1
INNER_BOUNDARY_MARKER = 2
OUTER_BOUNDARY_VALUE = [0.0,0.0]
INNER_BOUNDARY_VALUE = [0.0,0.0]

# --- Domain and Physics Parameters ---
L = 1.0  # Domain width
H = 1.0  # Domain height
v = 0.1  # Kinematic viscosity
tol = 1e-6 # Tolerance for coordinate comparisons
D = 1e-3

# --- Simulation Parameters ---
DT = 0.001
STEPS = 6000

# --- Visualization ---
PLOT_GRID_DENSITY = 100
FIXED_VMAX = 2.0 # Fixed max value for color bar


# --- Load Mesh Data ---
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
triangles = readEle(ELE_FILEPATH)
N = nodes_coords.shape[0]

#debug
print("\n--- Running Mesh Quality Analysis ---")
areas = []
min_edge_lengths = []
for tri in triangles:
    p1, p2, p3 = nodes_coords[tri]
    area = 0.5 * np.abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    if area > 1e-14:
        areas.append(area)
        l1 = np.linalg.norm(p1-p2)
        l2 = np.linalg.norm(p2-p3)
        l3 = np.linalg.norm(p3-p1)
        min_edge_lengths.append(min(float(l1), float(l2), float(l3)))

print(f"Mesh has {len(triangles)} triangles.")
print(f"Triangle Area: Min={np.min(areas):.2e}, Max={np.max(areas):.2e}, Avg={np.mean(areas):.2e}")
if np.min(areas) < 1e-10:
    print("WARNING: Very small triangles detected, which may cause numerical issues.")

# Check CFL-like condition based on mesh size
cfl_limit = np.min(min_edge_lengths)**2 / (2 * v)
print(f"Viscous CFL Time Step Limit (for stability reference): {cfl_limit:.2e}")
if DT > cfl_limit:
    print(f"WARNING: Your time step DT={DT:.2e} is large compared to the mesh size. While implicit methods are stable, this can affect accuracy.")

#debug
# --- Add this diagnostic check after loading the mesh ---
print("\n--- Checking Triangle Orientations ---")
positive_dets = 0
negative_dets = 0
for tri in triangles:
    p1, p2, p3 = nodes_coords[tri]
    det = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
    if det > 0:
        positive_dets += 1
    else:
        negative_dets += 1
print(f"Triangles with positive determinant (CCW): {positive_dets}")
print(f"Triangles with negative determinant (CW): {negative_dets}")
if negative_dets > 0:
    print("WARNING: Mesh contains clockwise-oriented triangles. This is the cause of the instability.")


# --- Identify Periodic Boundary Nodes ---
all_pairs = find_boundary_pairs(nodes_coords, L=1.0)

# Filter out corner/wall nodes from periodic pairs
non_wall_pairs = []
for master_idx, slave_idx in all_pairs:
    master_y = nodes_coords[master_idx, 1]
    if not (np.abs(master_y - 0.0) < tol or np.abs(master_y - H) < tol):
        non_wall_pairs.append((master_idx, slave_idx))

print(f"Original periodic pairs: {len(all_pairs)}, Filtered pairs (excluding walls): {len(non_wall_pairs)}")
pairs = non_wall_pairs


# --- Identify Dirichlet (Fixed-Value) Boundary Nodes ---
# Use np.isclose for robust floating-point comparison on walls
wall_node_indices = np.where(np.isclose(nodes_coords[:, 1], 0.0, atol=tol) | np.isclose(nodes_coords[:, 1], H, atol=tol))[0]
inner_boundary_indices = np.where(nodes_boundary_markers == INNER_BOUNDARY_MARKER)[0]
dirichlet_node_indices = np.union1d(wall_node_indices, inner_boundary_indices)


# --- Identify Pressure Reference Node ---
# Pin pressure at the first interior node to ensure a unique solution
pressure_ref_node = np.where(nodes_boundary_markers == 0)[0][0]



## 3. System Matrix Assembly
# --- Build Base Stiffness Matrix (Laplacian) ---
A_stiffness, _ = buildStiffnessMatrix(nodes_coords, triangles, g_source=0.0)
M_lumped_diag = buildLumpedMassMatrix(nodes_coords, triangles)

# --- Setup Viscosity Matrix (for Velocity) ---
# This matrix needs strong Dirichlet BCs for the velocity field.
A_visc = np.eye(N) + DT * v * A_stiffness
#apply_periodic_bc(A_visc, pairs)
# Enforce u=value on walls and inner body
A_visc[dirichlet_node_indices, :] = 0.0
A_visc[:, dirichlet_node_indices] = 0.0
A_visc[dirichlet_node_indices, dirichlet_node_indices] = 1.0

# --- Setup Pressure Matrix (for Pressure Correction) ---
print("\n--- Building Correct Lumped Laplacian L = inv(M)*A ---")
A_pressure = A_stiffness / (M_lumped_diag[:, np.newaxis] + 1e-12)

# Apply boundary conditions as before
apply_periodic_bc(A_pressure, pairs)
A_pressure[pressure_ref_node, :] = 0.0
A_pressure[:, pressure_ref_node] = 0.0
A_pressure[pressure_ref_node, pressure_ref_node] = 1.0


# Check eigenvalues for stability
# Note: Pinning the pressure node makes the matrix non-symmetric, so we check general eigenvalues.
try:
    eigvals = np.linalg.eigvals(A_pressure)
    print(f"Pressure matrix eigenvalues: min real part = {eigvals.real.min():.2e}, max real part = {eigvals.real.max():.2e}")
    if eigvals.real.min() < -1e-6:
         print("WARNING: Negative eigenvalues detected in pressure matrix!")
except np.linalg.LinAlgError:
    print("Could not compute eigenvalues for the pressure matrix.")


## 4. Initialization
# --- Initial Velocity Field and Body Forces ---
u = np.zeros((N, 2))
b_force = np.zeros((N, 2))
# b_force[:, 0] = 1.0 # Constant body force in x-direction


# --- Apply Initial Boundary Conditions on u ---
# Enforce periodic BCs by copying master node values to slave nodes
for master_idx, slave_idx in pairs:
    u[slave_idx] = u[master_idx]

# Enforce Dirichlet BCs using direct vectorized assignment 
u[wall_node_indices] = OUTER_BOUNDARY_VALUE
u[inner_boundary_indices] = INNER_BOUNDARY_VALUE

# ==============================================================================
# --- VISUALIZATION SETUP ---
# ==============================================================================
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.tight_layout(pad=3.0)

# --- 1. Setup for Concentration (Dye) and Streamlines Plot ---
triang = mtri.Triangulation(nodes_coords[:, 0],
                           nodes_coords[:, 1],
                           triangles)

# **MODIFICATION**: Inject a circular blob of dye
c = np.zeros(N)
blob_center = np.array([0.5, 0.75])
blob_radius = 0.1
node_distances = np.linalg.norm(nodes_coords - blob_center, axis=1)
c[node_distances < blob_radius] = 1.0

# Draw concentration background (the dye)
tpc_c = ax1.tripcolor(triang, c, shading="gouraud", cmap="plasma",
                      vmin=0.0, vmax=1.0)
fig.colorbar(tpc_c, ax=ax1, label="Concentration", shrink=0.8)
ax1.set_title("Dye Advection with Velocity Streamlines")
ax1.set_aspect('equal')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Prepare for streamlines by creating a regular grid
x_coords = np.linspace(0, 1, 100)
y_coords = np.linspace(0, 1, 100)
# Then, use np.meshgrid which is the standard for functions like streamplot
grid_x, grid_y = np.meshgrid(x_coords, y_coords)

stream_plot_object = None # Handle for the streamlines

# --- 2. Setup for Vorticity Plot ---
vorticity = calculate_vorticity(nodes_coords, triangles, u)
vort_max = np.max(np.abs(vorticity)) + 1e-9 # Initial max for color scale
tpc_vort = ax2.tripcolor(triang, vorticity, shading="gouraud", cmap="coolwarm",
                         vmin=-vort_max, vmax=vort_max)
fig.colorbar(tpc_vort, ax=ax2, label="Vorticity", shrink=0.8)
ax2.set_title("Vorticity (Flow Curl)")
ax2.set_aspect('equal')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)


# --- CORRECTED: Calculate the constant Mass Matrix ONCE before the loop ---
M, _ = build_mass_and_convection(nodes_coords, triangles, u) # u is zero, but M doesn't depend on it
apply_periodic_bc(M, pairs)

## 5. Time-Stepping Loop (Projection Method)
print("\nStarting simulation...")

for step in range(STEPS):
    # --- Step 1: Tentative Velocity with CORRECT Boundary Conditions ---

    # a) Start with the RHS from the previous step's solution
    rhs_x = u[:, 0] + DT * b_force[:, 0]
    rhs_y = u[:, 1] + DT * b_force[:, 1]

    # b) --- THIS IS THE FINAL CORRECTION ---
    #    Overwrite the RHS for all Dirichlet nodes with their specified values.
    #    This explicitly tells the solver what the velocity SHOULD BE on the boundaries.
    
    # Set RHS for stationary outer walls
    rhs_x[wall_node_indices] = OUTER_BOUNDARY_VALUE[0]
    rhs_y[wall_node_indices] = OUTER_BOUNDARY_VALUE[1]

    # Set RHS for the rotating inner cylinder (can be combined with a soft start)
    center_x, center_y = 0.5, 0.5
    target_angular_velocity = 5.0
    ramp_up_steps = 200 # Optional: Ramp up the speed to avoid a "hammer start"

    if step < ramp_up_steps:
        current_angular_velocity = target_angular_velocity * (step + 1) / ramp_up_steps
    else:
        current_angular_velocity = target_angular_velocity

    
    # c) Now, solve the system. A_visc is already set up to enforce u_star[i] = rhs[i] for any Dirichlet node i.
    u_star = np.zeros((N,2))
    u_star[:,0] = np.linalg.solve(A_visc, rhs_x)
    u_star[:,1] = np.linalg.solve(A_visc, rhs_y)

    # d) Enforce periodic BCs on the result
    for master_idx, slave_idx in pairs:
        u_star[slave_idx] = u_star[master_idx]


    # --- Step 2: STABILIZED Pressure (v1) ---
    div_u_star = calculate_divergence(nodes_coords, triangles, u_star)
    b_p = -(1.0 / DT) * div_u_star
    b_p -= b_p.mean()
    b_p[pressure_ref_node] = 0.0
    p_raw = np.linalg.solve(A_pressure, b_p)
    alpha_smooth = 0.01 
    smoothing_matrix = np.eye(N) + alpha_smooth * A_stiffness
    smoothing_matrix[pressure_ref_node, :] = 0.0
    smoothing_matrix[:, pressure_ref_node] = 0.0
    smoothing_matrix[pressure_ref_node, pressure_ref_node] = 1.0
    p_raw[pressure_ref_node] = 0.0
    p = np.linalg.solve(smoothing_matrix, p_raw)
    p -= p.mean()




    # --- Step 3: Velocity Update (Your code is correct here) ---
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p)
    u[:,0] = u_star[:,0] - DT * grad_px
    u[:,1] = u_star[:,1] - DT * grad_py
    
    final_div = calculate_divergence(nodes_coords, triangles, u)
    
    # --- Step 4: Enforce Boundary Conditions ---


    # Enforce outer walls (stationary)
    u[wall_node_indices] = OUTER_BOUNDARY_VALUE
#
    # Enforce periodicity
    for master_idx, slave_idx in pairs:
        u[slave_idx] = u[master_idx]
#
    # --- SQUIRMER BOUNDARY CONDITION ---
    center_x, center_y = 0.5, 0.5
#
    # --- Define Squirmer Parameters Here ---
    # B1 controls the swimming speed. A negative value swims left.
    B1 = -2.0 
   
    # B2 controls the type of swimmer.
    # B2 < 0 for a "pusher" (e.g., bacteria)
    # B2 > 0 for a "puller" (e.g., algae)
    # B2 = 0 for a neutral squirmer
    B2 = 5.0
#
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

    # --- Step 5: Update Dye Concentration ---
    # Note: M is already calculated. We only need the changing C_u matrix here.
    _, C_u  = build_mass_and_convection(nodes_coords, triangles, u)
    
    div_u_nodes = calculate_divergence(nodes_coords, triangles, u)
    G_diag = DT * (M_lumped_diag * div_u_nodes)

    for master, slave in pairs:
        G_diag[slave] = G_diag[master]

    A_c_step = M + DT * (C_u + D * A_stiffness) + np.diag(G_diag)
    apply_periodic_bc(A_c_step, pairs)

    rhs_c = M @ c
    c     = np.linalg.solve(A_c_step, rhs_c)

    #np.clip(c, 0.0, 1.0, out=c)
    for master_idx, slave_idx in pairs:
        c[slave_idx] = c[master_idx]


    # --- Step 6: Update Plots ---
    if step > 0:
        print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Max P: {np.max(np.abs(p)):.2e}, Final Div(u): {np.max(np.abs(final_div)):.2e}")
        
        # 1. Update Concentration (Dye) Plot
        tpc_c.set_array(c)
        
        # 2. Update Streamlines
        grid_u = griddata(nodes_coords, u[:, 0], (grid_x, grid_y), method='cubic', fill_value=0)
        grid_v = griddata(nodes_coords, u[:, 1], (grid_x, grid_y), method='cubic', fill_value=0)
        
        if stream_plot_object is not None:
            for artist in ax1.artists:
                artist.remove()
            for patch in ax1.patches:
                patch.remove()

        stream_plot_object = ax1.streamplot(grid_x, grid_y, grid_u, grid_v, 
                                            color='white', linewidth=0.8, density=1.2,
                                            arrowstyle='->', arrowsize=1.0)

        # 3. Update Vorticity Plot
        vorticity = calculate_vorticity(nodes_coords, triangles, u)
        tpc_vort.set_array(vorticity)
        vort_max = np.max(np.abs(vorticity)) + 1e-9
        tpc_vort.set_clim(vmin=-vort_max, vmax=vort_max)

        fig.suptitle(f"Step {step}", fontsize=16)
        plt.pause(0.001)

plt.ioff()
plt.show()
print("\nSimulation finished")
