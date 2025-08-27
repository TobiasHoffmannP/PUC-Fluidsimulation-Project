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
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]


        ADet = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2

        if ADet < 1e-14:
            print("skip!")
            continue

        # print(triangles.shape[1])
        y_diffs = [y2 - y3, y3 - y1, y1 - y2]
        x_diffs = [x3 - x2, x1 - x3, x2 - x1]

        for i in range(3):
            for j in range(3):
                # 1/2A [(2i − y3i)(y2j − y3j) + (x3i − x2i)(x3j − x2j)]
                integral_val = (y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j]) / ((ADet*2))

                # assemble to global matrix (AMatrix)
                # += because it can get values from other triangles as well
                AMatrix[tri[i], tri[j]] += integral_val

        area = 0.5 * abs(ADet)

        # pyramid over triangle is 1 / 3 of the area
        # bj = ∫Ω g(x, y) ϕj(x, y) dΩ = g * Area / 3
        if callable(g_source):
            g = g_source((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)
        else:
            g = g_source

        sourceIntergralValue = g * (area / 3)

        BVector[p1] += sourceIntergralValue
        BVector[p2] += sourceIntergralValue
        BVector[p3] += sourceIntergralValue

    return AMatrix, -BVector

def calculate_divergence(nodes, triangles, u_star):
    """
    u_star : shape (2, N)   – u_star[0,:] = u_x  , u_star[1,:] = u_x
    returns: (N,) nodal divergence  (area–weighted average)
    """
    N   = nodes.shape[0]
    div_sum  = np.zeros(N)
    area_sum = np.zeros(N)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]

        det   = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)   # 2×triangle-area
        EPS = 1e-14                # ~ machine epsilon * 10
        if abs(det) < EPS:
            continue               # skip the sliver / degenerate triangle
        inv2A = 1.0 / det          # = 1/(2·area)
        area  = 0.5 * abs(det)

        # ---- 1 take component-at-node correctly -------------------------
        # u_star has shape (2,N) → component first, *node* second
        ux1, uy1 = u_star[p1]
        ux2, uy2 = u_star[p2]
        ux3, uy3 = u_star[p3]


        # ---- 2 use proper gradient factors (1/(2A)) ---------------------
        d_ux_dx = (ux1*(y2-y3) + ux2*(y3-y1) + ux3*(y1-y2)) * inv2A
        d_uy_dy = (uy1*(x3-x2) + uy2*(x1-x3) + uy3*(x2-x1)) * inv2A
        div_tri = d_ux_dx + d_uy_dy          # constant on the element

        # ---- 3 area-weighted lumping to the three vertices --------------
        lump = div_tri * area / 3.0

        for p in tri:
            div_sum[p]  += lump
            area_sum[p] += area / 3.0        # same share
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
        # It returns the distance and the index within the `right_y_coords_for_tree` array.
        dist, right_array_idx = kdtree.query([left_y])

        # The `right_array_idx` corresponds to the position in the `right_indices`
        # array. We use it to get the original node index from the main `nodes_coords`.
        right_node_idx = right_indices[right_array_idx]
        pairs.append((left_idx, right_node_idx))

    return pairs


def apply_periodic_bc(A, b, pairs):
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


def calculate_gradiant(nodes_coords, triangles, p):
    # Calculate gradient of pressure p at nodes via lumping
    grad_px = np.zeros(N)
    grad_py = np.zeros(N)
    for tri in triangles:
        p_local = p[tri]
        x1, y1 = nodes_coords[tri[0]]
        x2, y2 = nodes_coords[tri[1]]
        x3, y3 = nodes_coords[tri[2]]
        
        det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        if abs(det) < 1e-14: continue
        
        inv2A = 1.0 / det
        area = 0.5 * abs(det)
        
        # Basis function gradients (shape 3x2)
        grads = np.array([
            [y2-y3, x3-x2],
            [y3-y1, x1-x3],
            [y1-y2, x2-x1]
        ], dtype=np.float64) * inv2A
        
        gp_element = grads.T @ p_local # Field gradient on element (shape 2,)
        lump = gp_element * (area / 3.0) # Lump to nodes
        
        grad_px[tri] += lump[0]
        grad_py[tri] += lump[1]

    return grad_px, grad_py


# ==============================================================================
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================

## 1. Constants and Parameters
# --- File Paths ---
NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.poly"

# --- Boundary Conditions ---
OUTER_BOUNDARY_MARKER = 1
INNER_BOUNDARY_MARKER = 2
OUTER_BOUNDARY_VALUE = [1.0,0.0]
INNER_BOUNDARY_VALUE = [0.0,0.0]

# --- Domain and Physics Parameters ---
L = 1.0  # Domain width
H = 1.0  # Domain height
v = 0.01 # Kinematic viscosity
tol = 1e-6 # Tolerance for coordinate comparisons

# --- Simulation Parameters ---
DT = 0.000001
STEPS = 600

# --- Visualization ---
PLOT_GRID_DENSITY = 100
FIXED_VMAX = 2.0 # Fixed max value for color bar


# --- Load Mesh Data ---
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
triangles = readEle(ELE_FILEPATH)
N = nodes_coords.shape[0]


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
wall_node_indices = np.where((np.abs(nodes_coords[:, 1] - 0.0) < tol) | (np.abs(nodes_coords[:, 1] - H) < tol))[0]
inner_boundary_indices = np.where(nodes_boundary_markers == INNER_BOUNDARY_MARKER)[0]
dirichlet_node_indices = np.union1d(wall_node_indices, inner_boundary_indices)


# --- Identify Pressure Reference Node ---
# Pin pressure at the first interior node to ensure a unique solution
pressure_ref_node = np.where(nodes_boundary_markers == 0)[0][0]



## 3. System Matrix Assembly
# --- Build Base Stiffness Matrix (Laplacian) ---
A_stiffness, _ = buildStiffnessMatrix(nodes_coords, triangles, g_source=0.0)


# --- Setup Viscosity Matrix (Implicit Scheme) ---
# A1 = (I + Δt * ν * A)
A_visc = np.eye(N) + DT * v * A_stiffness
apply_periodic_bc(A_visc, np.zeros(N), pairs)

# Enforce Dirichlet BCs: u_i = value --> row i becomes [0,...,1,...,0]
A_visc[dirichlet_node_indices, :] = 0.0
A_visc[:, dirichlet_node_indices] = 0.0
A_visc[dirichlet_node_indices, dirichlet_node_indices] = 1.0


# --- Setup Pressure Matrix (Poisson Equation) ---
A_pressure = A_stiffness.copy()
apply_periodic_bc(A_pressure, np.zeros(N), pairs)

fix = np.where(nodes_boundary_markers == 0)[0][0]   # first interior node
# Enforce pressure reference point p_ref = 0
A_pressure[pressure_ref_node, :] = 0.0
A_pressure[:, pressure_ref_node] = 0.0
A_pressure[pressure_ref_node, pressure_ref_node] = 1.0 

# Check eigenvalues for stability
eigvals = np.linalg.eigvalsh(A_pressure)
print(f"Pressure matrix eigenvalues: min λ = {eigvals.min():.2e}, max λ = {eigvals.max():.2e}")



## 4. Initialization
# --- Initial Velocity Field and Body Forces ---
u = np.zeros((N, 2))
b_force = np.zeros((N, 2))
b_force[:, 0] = 0.1 # Constant body force in x-direction


# --- Apply Initial Boundary Conditions ---
# Enforce periodic BCs by copying master node values to slave nodes
for master_idx, slave_idx in pairs:
    u[slave_idx] = u[master_idx]

# Enforce Dirichlet BCs using direct vectorized assignment 
u[wall_node_indices] = OUTER_BOUNDARY_VALUE
u[inner_boundary_indices] = INNER_BOUNDARY_VALUE


# --- Setup Visualization ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
grid_x, grid_y = np.meshgrid(
    np.linspace(0.01, 0.99, PLOT_GRID_DENSITY),
    np.linspace(0.01, 0.99, PLOT_GRID_DENSITY)
)
colorbar = None


## 5. Time-Stepping Loop (Projection Method)
print("\nStarting simulation...")

for step in range(STEPS):
    # --- Step 1: Tentative Velocity (Advection-Diffusion) ---
    # We solve (I + Δt*ν*A)u* = u^n + Δt*F
    rhs_x = u[:, 0] + DT * b_force[:, 0]
    rhs_y = u[:, 1] + DT * b_force[:, 1]

    # Enforce Dirichlet BC on the right-hand side
    rhs_x[dirichlet_node_indices] = 0.0
    rhs_y[dirichlet_node_indices] = 0.0

    # Assemble u* 
    u_star = np.zeros((N,2))
    u_star[:,0] = np.linalg.solve(A_visc, rhs_x) # u_x*
    u_star[:,1] = np.linalg.solve(A_visc, rhs_y) # u_y*


    # --- Step 2: Pressure Correction (Poisson Equation) ---
    # We solve A*p = (1/Δt)∇·u*
    div_u_star = calculate_divergence(nodes_coords, triangles, u_star)
    
    b_p = - (div_u_star / DT)
    b_p -= b_p.mean() # Ensure compatibility
    b_p[pressure_ref_node] = 0.0 # Enforce reference on RHS    

    p   = np.linalg.solve(A_pressure, b_p)

    # --- Step 3: Velocity Update ---
    # We update u^(n+1) = u* - Δt * ∇p
    # Calculate gradient of pressure p at nodes via lumping
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p) 

    u[:,0] = u_star[:,0] - DT * grad_px
    u[:,1] = u_star[:,1] - DT * grad_py

    
    final_div = calculate_divergence(nodes_coords, triangles, u)
    
    # --- Step 4: Enforce Boundary Conditions ---
    # Enforce periodic BCs by copying master node values to slave nodes
    for master_idx, slave_idx in pairs:
        u[slave_idx] = u[master_idx]

    # Enforce Dirichlet BCs using direct vectorized assignment
    u[wall_node_indices] = OUTER_BOUNDARY_VALUE
    u[inner_boundary_indices] = INNER_BOUNDARY_VALUE

    
    if step > 0:
        # final_div = calculate_divergence(nodes_coords, triangles, u)
        print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Max P: {np.max(np.abs(p)):.2e}, Div(u*): {np.max(np.abs(div_u_star)):.2e}, Final Div(u): {np.max(np.abs(final_div)):.2e}")        

       # Update plot
        ax.clear()
        u_magnitude = np.linalg.norm(u, axis=1)
        tpc = ax.tripcolor(triang, u_magnitude, shading='gouraud', cmap='viridis', vmin=0, vmax=FIXED_VMAX)


       # Draw streamlines
        interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
        interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])
        u_x_grid = interpolator_x(grid_x, grid_y)
        u_y_grid = interpolator_y(grid_x, grid_y)
        ax.streamplot(grid_x, grid_y, u_x_grid, u_y_grid, color='black', linewidth=1, density=0.7)

        if colorbar is None:
            colorbar = fig.colorbar(tpc, ax=ax)
            colorbar.set_label("Velocity Magnitude")
        else:
            colorbar.update_normal(tpc)        

        ax.set_aspect('equal')
        ax.set_title(f"Time Evolution: Step {step}/{STEPS}")
        fig.canvas.draw_idle()
        plt.pause(0.01)


print("\nSimulation finished") 
