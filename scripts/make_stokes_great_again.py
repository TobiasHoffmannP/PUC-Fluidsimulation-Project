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
                # --- THE DEFINITIVE FIX IS HERE ---
                # The denominator MUST use the absolute value of the determinant to be
                # consistent with the physics of diffusion and with our other functions.
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
            
        # --- CORRECTION ---
        # Use the SIGNED determinant for calculating derivatives.
        inv2A = 1.0 / det
        # Use the ABSOLUTE area for physical lumping.
        area  = 0.5 * abs(det)

        ux1, uy1 = u_star[p1]
        ux2, uy2 = u_star[p2]
        ux3, uy3 = u_star[p3]

        # --- REVERTED TO YOUR ORIGINAL, CORRECT FORMULAS ---
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
        
        # --- CORRECTION ---
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

# ==============================================================================
# --- TESTS ---
# ==============================================================================


def run_checkerboard_test():
    print("\n--- Running LBB 'Checkerboard' Test ---")

    # 1. Create a synthetic velocity field u_star
    # This field is designed to have a high-frequency divergence
    u_star_test = np.zeros((N, 2))
    x_coords = nodes_coords[:, 0]
    y_coords = nodes_coords[:, 1]

    # A divergence pattern of sin(k*pi*x) * sin(k*pi*y)
    # Velocity u_x = cos(k*pi*x)*sin(k*pi*y), u_y = sin(k*pi*x)*cos(k*pi*y)
    # Note: This is just one way to create a provocative divergence field
    k = 8 # Wave number - higher k means higher frequency
    u_star_test[:, 0] = np.cos(k * np.pi * x_coords) * np.sin(k * np.pi * y_coords)
    u_star_test[:, 1] = np.sin(k * np.pi * x_coords) * np.cos(k * np.pi * y_coords)

    # 2. Calculate its divergence
    div_u_star_test = calculate_divergence(nodes_coords, triangles, u_star_test)

    # 3. Solve the pressure equation
    b_p_test = div_u_star_test.copy()
    b_p_test[pressure_ref_node] = 0.0 # Enforce reference

    p_test = np.linalg.solve(A_pressure, b_p_test)

    # 4. Visualize the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot the divergence we fed in
    tpc1 = ax1.tripcolor(triang, div_u_star_test, shading='gouraud', cmap='coolwarm')
    ax1.set_title("Input: High-Frequency Divergence")
    ax1.set_aspect('equal')
    fig.colorbar(tpc1, ax=ax1)

    # Plot the resulting pressure field
    tpc2 = ax2.tripcolor(triang, p_test, shading='gouraud', cmap='viridis')
    ax2.set_title("Output: Pressure Response")
    ax2.set_aspect('equal')
    fig.colorbar(tpc2, ax=ax2)

    plt.suptitle("P1/P1 Checkerboard Instability Test")
    plt.show()
    plt.pause(10)


def test_gradient_calculation():
    """
    Tests if the gradient calculation is correct by feeding it a simple
    linear pressure field p = 2x + 3y. The expected gradient is (2, 3).
    """
    print("\n--- Running Test A: Gradient Calculation ---")
    # Create a simple, known pressure field
    p_test = 2 * nodes_coords[:, 0] + 3 * nodes_coords[:, 1]

    # Calculate the gradient using your functiograd_px, grad_py = calculate_gradiant_fixed(nodes_coords, triangles, p_test)
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p_test)
    # Check the results
    print(f"Expected Gradient: (2.0, 3.0)")
    print(f"Calculated Gradient X: Mean={np.mean(grad_px):.4f}, StdDev={np.std(grad_px):.4f}")
    print(f"Calculated Gradient Y: Mean={np.mean(grad_py):.4f}, StdDev={np.std(grad_py):.4f}")
    
    if not (np.isclose(np.mean(grad_px), 2.0, atol=0.1) and np.isclose(np.mean(grad_py), 3.0, atol=0.1)):
        print("WARNING: Gradient calculation appears to be inaccurate.")
    else:
        print("SUCCESS: Gradient calculation is behaving as expected.")


def test_divergence_calculation():
    """
    Tests if the divergence calculation is correct by feeding it a simple
    velocity field u = (2x, 3y). The expected divergence is 2 + 3 = 5.
    """
    print("\n--- Running Test B: Divergence Calculation ---")
    # Create a simple, known velocity field
    u_test = np.zeros((N, 2))
    u_test[:, 0] = 2 * nodes_coords[:, 0]
    u_test[:, 1] = 3 * nodes_coords[:, 1]

    # Calculate the divergence using your function
    div_u = calculate_divergence(nodes_coords, triangles, u_test)

    # Check the results
    print(f"Expected Divergence: 5.0")
    print(f"Calculated Divergence: Mean={np.mean(div_u):.4f}, StdDev={np.std(div_u):.4f}")

    if not np.isclose(np.mean(div_u), 5.0, atol=0.1):
        print("WARNING: Divergence calculation appears to be inaccurate.")
    else:
        print("SUCCESS: Divergence calculation is behaving as expected.")


def test_projection_consistency():
    """
    Tests the core assumption of the projection method.
    """
    print("\n--- Running Test C: Projection Consistency ---")
    # Step 1: Run a single step of the solver to get a u_star and p
    u_test = np.zeros((N, 2))
    b_force_test = np.zeros((N, 2))
    b_force_test[:, 0] = 0.1

    rhs_x = u_test[:, 0] + DT * b_force_test[:, 0]
    rhs_y = u_test[:, 1] + DT * b_force_test[:, 1]
    
    u_star = np.zeros((N, 2))
    u_star[:, 0] = np.linalg.solve(A_visc, rhs_x)
    u_star[:, 1] = np.linalg.solve(A_visc, rhs_y)

    # --- THIS IS THE KEY CHANGE ---
    # Calculate the RHS of the pressure equation using the consistent method.
    b_p_vector = calculate_consistent_rhs(nodes_coords, triangles, u_star)
    
    # Scale by 1/DT and apply boundary conditions
    b_p = b_p_vector / DT
    b_p -= b_p.mean() # Ensure solvability
    b_p[pressure_ref_node] = 0.0
    p = np.linalg.solve(A_pressure, b_p)

    # Step 2: Check the relationship
    # The relationship is: A_pressure * p = b_p = (1/DT) * consistent_div(u_star)
    # So, we expect DT * (A_pressure * p) to be equal to the original vector.
    # We can check the correlation between them.
    
    from scipy.stats import pearsonr
    vec1 = DT * (A_pressure @ p)
    vec2 = b_p_vector

    active_nodes = np.where(np.abs(vec2) > 1e-9)[0]
    if len(active_nodes) > 1:
        corr, _ = pearsonr(vec1[active_nodes], vec2[active_nodes])
        print(f"Correlation between DT*(A_pressure*p) and consistent RHS: {corr:.6f}")
        if corr > 0.999:
            print("SUCCESS: The projection relationship holds with consistent operators.")
        else:
            print("WARNING: The projection relationship still does not hold.")
    else:
        print("Test inconclusive: RHS vector is zero.")


def test_laplacian_vs_divgrad(A_pressure, nodes_coords, triangles):
    """
    Directly compares the discrete Laplacian matrix (A_pressure) against the
    composite operator -div(grad(...)) to check for consistency.
    """
    print("\n--- Running Test D: Laplacian vs. Div(Grad) ---")
    
    # Create a non-trivial test pressure field, like a Gaussian blob
    p_test = np.exp(-20 * ((nodes_coords[:, 0] - 0.5)**2 + (nodes_coords[:, 1] - 0.5)**2))
    
    # Method 1: Apply the discrete Laplacian matrix from the FEM stiffness matrix
    # This is the 'consistent' Laplacian for your system
    laplacian_p_matrix = A_pressure @ p_test

    # Method 2: Apply the composite operator -div(grad(p)) using your lumping functions
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p_test)
    grad_p_vector = np.vstack((grad_px, grad_py)).T
    div_grad_p_composite = calculate_divergence(nodes_coords, triangles, grad_p_vector)

    # We expect laplacian_p_matrix to be proportional to -div_grad_p_composite
    from scipy.stats import pearsonr
    # Use only non-zero values to avoid spurious correlations at the boundaries
    active_nodes = np.where(np.abs(laplacian_p_matrix) > 1e-9)[0]
    corr, _ = pearsonr(laplacian_p_matrix[active_nodes], -div_grad_p_composite[active_nodes])
    print(f"Correlation between (A_pressure*p) and -div(grad(p)): {corr:.4f}")

    if corr < 0.99:
        print("CONFIRMED: The discrete Laplacian A_pressure and the composite operator -div(grad(...)) are not equivalent.")
    else:
        print("SUCCESS: The discrete Laplacian and -div(grad(...)) are highly correlated.")

    # Plot for visual inspection
    plt.figure(figsize=(8, 8))
    # We plot the negative of the div(grad) because the Laplacian is defined as -div(grad)
    plt.scatter(laplacian_p_matrix, -div_grad_p_composite, alpha=0.5, label='Nodal Values')
    plt.xlabel("Laplacian from Matrix (A_pressure * p)")
    plt.ylabel("Laplacian from Composition (-div(grad(p)))")
    plt.title("Comparison of Discrete Laplacian Operators")
    plt.grid(True)
    
    # Add a y=x line for reference, showing what a perfect match would look like
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match (y=x)')
    plt.legend()
    plt.axis('equal') # Ensure the scale is the same on both axes
    plt.show()
    plt.pause(60) # Pause to ensure the plot is displayed


def test_adjointness(nodes_coords, triangles, nodes_boundary_markers):
    """
    Tests if the discrete divergence and gradient operators are negative adjoints.
    This is the fundamental property required for a stable pressure solve.
    It checks if <grad(p), u> ≈ -<p, div(u)>, where <,> is the L2 inner product.
    """
    print("\n--- Running Test E: Adjoint Relationship ---")
    N = nodes_coords.shape[0]

    # 1. Create two random fields that are zero on the boundary.
    # This mimics functions in the correct space and avoids boundary term issues.
    np.random.seed(0)
    p_test = np.random.rand(N)
    u_test = np.random.rand(N, 2)
    
    boundary_nodes = np.where(nodes_boundary_markers != 0)[0]
    p_test[boundary_nodes] = 0.0
    u_test[boundary_nodes, :] = 0.0

    # 2. Calculate the lumped mass at each node (the integral of the basis function).
    lumped_mass = np.zeros(N)
    for tri in triangles:
        p_nodes = nodes_coords[tri]
        det = p_nodes[0,0]*(p_nodes[1,1]-p_nodes[2,1]) + \
              p_nodes[1,0]*(p_nodes[2,1]-p_nodes[0,1]) + \
              p_nodes[2,0]*(p_nodes[0,1]-p_nodes[1,1])
        if abs(det) < 1e-14: continue
        area = 0.5 * abs(det)
        for i in range(3):
            lumped_mass[tri[i]] += area / 3.0
    
    # 3. Calculate LHS of the equation: <grad(p), u>
    # <v1, v2> = integral(v1 . v2 dV) ≈ sum( (v1_i . v2_i) * mass_i )
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p_test)
    grad_p_vector = np.vstack((grad_px, grad_py)).T
    
    # Dot product at each node: (grad_p_x * u_x) + (grad_p_y * u_y)
    dot_product = np.sum(grad_p_vector * u_test, axis=1)
    lhs = np.sum(dot_product * lumped_mass)

    # 4. Calculate RHS of the equation: -<p, div(u)>
    # -<f1, f2> = - integral(f1 * f2 dV) ≈ - sum( f1_i * f2_i * mass_i )
    div_u = calculate_divergence(nodes_coords, triangles, u_test)
    rhs = -np.sum(p_test * div_u * lumped_mass)

    # 5. Compare LHS and RHS
    print(f"LHS <grad(p), u> = {lhs:.6f}")
    print(f"RHS -<p, div(u)> = {rhs:.6f}")
    
    # Use relative error for a robust comparison
    relative_error = np.abs(lhs - rhs) / (0.5 * (np.abs(lhs) + np.abs(rhs)) + 1e-9)
    print(f"Relative Error: {relative_error:.4f}")

    if relative_error < 1e-6:
        print("SUCCESS: Gradient and Divergence operators appear to be adjoints.")
        return True
    else:
        print("FAILURE: Gradient and Divergence operators are NOT adjoints.")
        print("This is the definitive root cause of the negative eigenvalues.")
        return False

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
OUTER_BOUNDARY_VALUE = [1.0,0.0]
INNER_BOUNDARY_VALUE = [0.0,0.0]

# --- Domain and Physics Parameters ---
L = 1.0  # Domain width
H = 1.0  # Domain height
v = 1.0 # Kinematic viscosity
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
apply_periodic_bc(A_visc, pairs)
# Enforce u=value on walls and inner body
A_visc[dirichlet_node_indices, :] = 0.0
A_visc[:, dirichlet_node_indices] = 0.0
A_visc[dirichlet_node_indices, dirichlet_node_indices] = 1.0

# --- Setup Pressure Matrix (for Pressure Correction) ---
print("\n--- Building Correct Lumped Laplacian L = inv(M)*A ---")
A_pressure = A_stiffness / (M_lumped_diag[:, np.newaxis] + 1e-12)

# Apply the boundary conditions for pressure
apply_periodic_bc(A_pressure, pairs)
A_pressure[pressure_ref_node, :] = 0.0
A_pressure[:, pressure_ref_node] = 0.0 # This makes the matrix non-symmetric but is a common way to pin a value.
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
b_force[:, 0] = 0.1 # Constant body force in x-direction


# --- Apply Initial Boundary Conditions on u ---
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

#debug
# Visualize the boundary conditions
# plt.figure(figsize=(8, 8))
# plt.tripcolor(triang, facecolors=np.ones(len(triangles)), edgecolors='lightgray', linewidth=0.5)

# Plot node types
# plt.scatter(nodes_coords[:, 0], nodes_coords[:, 1], c='cyan', label='Interior Nodes', s=5)
# plt.scatter(nodes_coords[dirichlet_node_indices, 0], nodes_coords[dirichlet_node_indices, 1], c='red', label='Dirichlet Nodes', s=10)


#test_gradient_calculation()
#test_divergence_calculation()
#test_projection_consistency()
#test_laplacian_vs_divgrad(A_pressure, nodes_coords, triangles)
#test_adjointness(nodes_coords, triangles, nodes_boundary_markers)
#run_checkerboard_test()
#import sys
#sys.exit("\nExiting after diagnostic test.")

## 5. Time-Stepping Loop (Projection Method)
print("\nStarting simulation...")

for step in range(STEPS):
    # --- Step 1: Tentative Velocity (Advection-Diffusion) ---
    # We solve (I + Δt*ν*A)u* = u^n + Δt*F
    rhs_x = u[:, 0] + DT * b_force[:, 0]
    rhs_y = u[:, 1] + DT * b_force[:, 1]
    u_star = np.zeros((N,2))
    u_star[:,0] = np.linalg.solve(A_visc, rhs_x)
    u_star[:,1] = np.linalg.solve(A_visc, rhs_y)


    # --- Step 2: Pressure Correction (Poisson Equation) ---
    # Use the consistent RHS for the FEM pressure solve
    b_p_vector = calculate_consistent_rhs(nodes_coords, triangles, u_star)
    b_p = - (b_p_vector / DT)
    b_p -= b_p.mean()
    b_p[pressure_ref_node] = 0.0
    p = np.linalg.solve(A_pressure, b_p)


    # --- Step 3: Velocity Update ---
    # Use the FVM-style gradient for the velocity correction
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p)
    u[:,0] = u_star[:,0] - DT * grad_px
    u[:,1] = u_star[:,1] - DT * grad_py
    
    final_div = calculate_divergence(nodes_coords, triangles, u)
    
    # --- Step 4: Enforce Boundary Conditions ---
    # This step is correct
    for master_idx, slave_idx in pairs:
        u[slave_idx] = u[master_idx]

    u[wall_node_indices] = OUTER_BOUNDARY_VALUE
    u[inner_boundary_indices] = INNER_BOUNDARY_VALUE
    
    if step > 0:
        # final_div = calculate_divergence(nodes_coords, triangles, u)
        print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Max P: {np.max(np.abs(p)):.2e}, Div(u*): {np.max(np.abs(b_p_vector)):.2e}, Final Div(u): {np.max(np.abs(final_div)):.2e}")        

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


