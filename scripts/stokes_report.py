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
            
        inv2A = 1.0 / det
        area  = 0.5 * abs(det)

        ux1, uy1 = u_star[p1]
        ux2, uy2 = u_star[p2]
        ux3, uy3 = u_star[p3]
 
        d_ux_dx = (ux1*(y2-y3) + ux2*(y3-y1) + ux3*(y1-y2)) * inv2A
        d_uy_dy = (uy1*(x3-x2) + uy2*(x1-x3) + uy3*(x2-x1)) * inv2A
        div_tri = d_ux_dx + d_uy_dy

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
        
        
        inv2A = 1.0 / det
        area = 0.5 * abs(det)
        
        grads = np.array([
            [p_nodes[1,1]-p_nodes[2,1], p_nodes[2,0]-p_nodes[1,0]],
            [p_nodes[2,1]-p_nodes[0,1], p_nodes[0,0]-p_nodes[2,0]],
            [p_nodes[0,1]-p_nodes[1,1], p_nodes[1,0]-p_nodes[0,0]]
        ], dtype=np.float64) * inv2A
        
        gp_element = grads.T @ p_local
        
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
    plt.pause(60)


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

def test_laplacian_blind_spot(A_pressure, nodes_coords, triangles):
    """
    Tests if the discrete Laplacian has a weak response to a checkerboard
    pattern, which is a key symptom of LBB instability.
    """
    print("\n--- Running Test F: The Laplacian's Blind Spot ---")
    
    # 1. Create a perfect, high-frequency checkerboard pressure field.
    k = 25 # Wave number - higher k means higher frequency
    x_coords = nodes_coords[:, 0]
    y_coords = nodes_coords[:, 1]
    p_checkerboard = np.sin(k * np.pi * x_coords) * np.sin(k * np.pi * y_coords)

    # 2. Apply the Laplacian matrix to this checkerboard field.
    laplacian_response = A_pressure @ p_checkerboard

    # 3. Analyze the result.
    # A stable operator would have a strong response. An unstable one will barely react.
    norm_input = np.linalg.norm(p_checkerboard)
    norm_response = np.linalg.norm(laplacian_response)
    print(f"Norm of Input Checkerboard: {norm_input:.4f}")
    print(f"Norm of Laplacian Response: {norm_response:.4f}")
    if norm_response < norm_input * 0.1: # If the response is less than 10% of the input
        print("CONFIRMED: The Laplacian operator is weak against the checkerboard mode.")
    else:
        print("WARNING: The Laplacian has a strong response. The instability may lie elsewhere.")

    # 4. Visualize the fields.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)

    tpc1 = ax1.tripcolor(triang, p_checkerboard, shading='gouraud', cmap='coolwarm')
    ax1.set_title("Input: Perfect Checkerboard Pressure")
    ax1.set_aspect('equal')
    fig.colorbar(tpc1, ax=ax1)

    # Note: We plot the absolute value of the response to see its structure
    tpc2 = ax2.tripcolor(triang, np.abs(laplacian_response), shading='gouraud', cmap='viridis')
    ax2.set_title("Output: Laplacian Response (Magnitude)")
    ax2.set_aspect('equal')
    fig.colorbar(tpc2, ax=ax2)
    
    plt.suptitle("Test F: Laplacian Operator Nullspace Test")
    plt.show()
    plt.pause(100)

def test_gradient_of_checkerboard(nodes_coords, triangles):
    """
    Visualizes the gradient of a checkerboard pressure field to show
    why LBB instability is destructive.
    """
    print("\n--- Running Test G: Gradient of a Checkerboard (The Destructor) ---")

    # 1. Create the same checkerboard pressure field.
    k = 25
    x_coords = nodes_coords[:, 0]
    y_coords = nodes_coords[:, 1]
    p_checkerboard = np.sin(k * np.pi * x_coords) * np.sin(k * np.pi * y_coords)

    # 2. Calculate its gradient using your operator.
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p_checkerboard)
    
    # 3. Visualize the gradient field using a quiver plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)

    tpc1 = ax1.tripcolor(triang, p_checkerboard, shading='gouraud', cmap='coolwarm')
    ax1.set_title("Input: Perfect Checkerboard Pressure")
    ax1.set_aspect('equal')
    fig.colorbar(tpc1, ax=ax1)

    # Use a quiver plot to show the vector field of the gradient.
    # We only plot a subset of vectors for clarity.
    skip = 20 # Plot every 20th node
    ax2.quiver(nodes_coords[::skip, 0], nodes_coords[::skip, 1], grad_px[::skip], grad_py[::skip])
    ax2.set_title("Output: 'Garbage' Gradient Field")
    ax2.set_aspect('equal')
    
    plt.suptitle("Test G: Why the Checkerboard is Destructive")
    plt.show()
    plt.pause(100)

def test_rhs_handling(A_stiffness, nodes_coords, triangles, inner_boundary_indices, dirichlet_node_indices):
    """
    Tests the difference between the flawed and corrected methods for handling
    time-dependent Dirichlet BCs in the implicit viscosity solve.
    """
    print("\n--- Running Test H: Boundary Condition Propagation ---")
    N = nodes_coords.shape[0]
    DT = 0.00001 # Use the small DT for the fine mesh
    v = 0.1

    # --- Setup for the Test ---
    # We create a simple, constant velocity BC for the cylinder for clarity
    test_boundary_velocity = [1.0, 0.0] 
    
    # The A_visc matrix with strong Dirichlet BCs (used by both methods)
    A_visc_test = np.eye(N) + DT * v * A_stiffness
    A_visc_test[dirichlet_node_indices, :] = 0.0
    A_visc_test[:, dirichlet_node_indices] = 0.0
    A_visc_test[dirichlet_node_indices, dirichlet_node_indices] = 1.0

    # The RHS is based on the previous velocity field, u^n.
    u_n = np.zeros((N, 2))
    # The BC was applied to u^n at the end of the last step.
    u_n[inner_boundary_indices] = test_boundary_velocity
    rhs_x_flawed = u_n[:, 0]
    rhs_y_flawed = u_n[:, 1]
    
    u_star_flawed = np.zeros((N, 2))
    u_star_flawed[:, 0] = np.linalg.solve(A_visc_test, rhs_x_flawed)
    u_star_flawed[:, 1] = np.linalg.solve(A_visc_test, rhs_y_flawed)

    # The RHS is built explicitly with the desired boundary values for u_star.
    rhs_x_correct = np.zeros(N) # Starts from a zero field
    rhs_y_correct = np.zeros(N)
    rhs_x_correct[inner_boundary_indices] = test_boundary_velocity[0]
    rhs_y_correct[inner_boundary_indices] = test_boundary_velocity[1]

    u_star_correct = np.zeros((N, 2))
    u_star_correct[:, 0] = np.linalg.solve(A_visc_test, rhs_x_correct)
    u_star_correct[:, 1] = np.linalg.solve(A_visc_test, rhs_y_correct)

    # --- 3. Visualize the Comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
    
    u_mag_flawed = np.linalg.norm(u_star_flawed, axis=1)
    tpc1 = ax1.tripcolor(triang, u_mag_flawed, shading='gouraud', cmap='viridis')
    ax1.set_title("Flawed Method: u_star result")
    ax1.set_aspect('equal')
    fig.colorbar(tpc1, ax=ax1)

    u_mag_correct = np.linalg.norm(u_star_correct, axis=1)
    tpc2 = ax2.tripcolor(triang, u_mag_correct, shading='gouraud', cmap='viridis')
    ax2.set_title("Corrected Method: u_star result")
    ax2.set_aspect('equal')
    fig.colorbar(tpc2, ax=ax2)

    plt.suptitle("Test H: Comparison of RHS Handling for Viscosity Solve")
    plt.show()
    plt.pause(100)

def run_singlestep_ustar_diagnostic(A_visc, u, b_force, DT, pairs, nodes_coords, triangles):
    """Runs a single step and visualizes the intermediate u_star field."""
    print("\n--- Running Test I: Single-Step u_star Diagnostic ---")
    
    # --- Perform Step 1 ---
    rhs_x = u[:, 0] + DT * b_force[:, 0]
    rhs_y = u[:, 1] + DT * b_force[:, 1]
    u_star = np.zeros((u.shape[0], 2))
    u_star[:, 0] = np.linalg.solve(A_visc, rhs_x)
    u_star[:, 1] = np.linalg.solve(A_visc, rhs_y)
    for master_idx, slave_idx in pairs:
        u_star[slave_idx] = u_star[master_idx]
        
    # --- Analyze the result ---
    div_u_star = calculate_divergence(nodes_coords, triangles, u_star)
    vort_u_star = calculate_vorticity(nodes_coords, triangles, u_star)
    
    print(f"Max u_star magnitude: {np.max(np.linalg.norm(u_star, axis=1)):.2e}")
    print(f"Max div(u_star): {np.max(np.abs(div_u_star)):.2e}")

    # --- Visualize ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
    
    ax1.set_title("u_star Magnitude")
    tpc1 = ax1.tripcolor(triang, np.linalg.norm(u_star, axis=1), shading='gouraud', cmap='viridis')
    fig.colorbar(tpc1, ax=ax1)
    
    ax2.set_title("div(u_star)")
    tpc2 = ax2.tripcolor(triang, div_u_star, shading='gouraud', cmap='coolwarm')
    fig.colorbar(tpc2, ax=ax2)

    ax3.set_title("vorticity(u_star)")
    tpc3 = ax3.tripcolor(triang, vort_u_star, shading='gouraud', cmap='seismic')
    fig.colorbar(tpc3, ax=ax3)

    for ax in [ax1, ax2, ax3]: ax.set_aspect('equal')
    plt.suptitle("Test I: Inspecting the Tentative Velocity Field")
    plt.show()
    plt.pause(100)


def run_singlestep_pressure_diagnostic(A_pressure, u_star, DT, nodes_coords, triangles, pressure_ref_node):
    """Runs a single pressure solve and visualizes the input and output."""
    print("\n--- Running Test J: Single-Step Pressure Solve Diagnostic ---")
    
    # --- Perform Step 2 (without stabilization) ---
    div_u_star = calculate_divergence(nodes_coords, triangles, u_star)
    b_p = -(1.0 / DT) * div_u_star
    b_p -= b_p.mean()
    b_p[pressure_ref_node] = 0.0
    p_raw = np.linalg.solve(A_pressure, b_p)

    # --- Analyze the result ---
    print(f"Max input value to solver (div_u_star): {np.max(np.abs(div_u_star)):.2e}")
    print(f"Max output value from solver (p_raw): {np.max(np.abs(p_raw)):.2e}")

    # --- Visualize ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
    
    ax1.set_title("Input to Pressure Solve: div(u_star)")
    tpc1 = ax1.tripcolor(triang, div_u_star, shading='gouraud', cmap='coolwarm')
    fig.colorbar(tpc1, ax=ax1)
    
    ax2.set_title("Output from Pressure Solve: p_raw")
    tpc2 = ax2.tripcolor(triang, p_raw, shading='gouraud', cmap='viridis')
    fig.colorbar(tpc2, ax=ax2)

    for ax in [ax1, ax2]: ax.set_aspect('equal')
    plt.suptitle("Test J: Inspecting the Pressure Solve")
    plt.show()
    plt.pause(100)

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

# === Visualization === 
## --- Setup1 Visualization ---
#plt.ion()
## Create a figure with 3 side-by-side subplots
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
#triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
#grid_x, grid_y = np.meshgrid(
#    np.linspace(0.01, 0.99, PLOT_GRID_DENSITY),
#    np.linspace(0.01, 0.99, PLOT_GRID_DENSITY)
#)
## Initialize colorbar objects so we can update them later
#cb1, cb2, cb3 = None, None, None

# --- Setup2 Visualization ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
grid_x, grid_y = np.meshgrid(
    np.linspace(0.01, 0.99, PLOT_GRID_DENSITY),
    np.linspace(0.01, 0.99, PLOT_GRID_DENSITY)
)
colorbar = None


#debug (visual line mode :'<,'>s/^/# / to commit out :'<,'>s/^# // to uncommit)
## --- Visualize all boundary conditions ---
#plt.figure(figsize=(9, 8))
#ax = plt.gca() # Get current axes
#
## Plot the mesh triangles as a background
#plt.tripcolor(triang, facecolors=np.ones(len(triangles)), edgecolors='lightgray', linewidth=0.5)
#
# ## Plot interior nodes
#plt.scatter(nodes_coords[:, 0], nodes_coords[:, 1], c='cyan', label='Interior Nodes', s=10)
#
## Plot Dirichlet nodes (walls and cylinder)
#plt.scatter(nodes_coords[dirichlet_node_indices, 0], nodes_coords[dirichlet_node_indices, 1], 
#            c='red', label='Dirichlet Nodes', s=25, marker='x')
#
## --- Plot Periodic Nodes ---
#if pairs: # Check that the pairs list is not empty
#    master_indices = [p[0] for p in pairs]
#    slave_indices = [p[1] for p in pairs]
#
#    # Plot master nodes (left side)
#    plt.scatter(nodes_coords[master_indices, 0], nodes_coords[master_indices, 1], 
#                c='lime', label='Periodic Master', s=40, edgecolors='k', marker='o')
#
#    # Plot slave nodes (right side)
#    plt.scatter(nodes_coords[slave_indices, 0], nodes_coords[slave_indices, 1], 
#                c='magenta', label='Periodic Slave', s=40, edgecolors='k', marker='o')
#
#    # (Optional but Recommended) Draw lines connecting the pairs
#    for master_idx, slave_idx in pairs:
#        master_coord = nodes_coords[master_idx]
#        slave_coord = nodes_coords[slave_idx]
#        plt.plot([master_coord[0], slave_coord[0]], 
#                 [master_coord[1], slave_coord[1]], 
#                 'k--', linewidth=0.7, alpha=0.6)
#
#plt.title("Boundary Condition Visualization")
#plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
#plt.axis('equal')
#plt.tight_layout()
#plt.show()
#plt.pause(10)


# --- TESTS ---

#test_gradient_calculation()
#test_divergence_calculation()
#test_projection_consistency()
#test_laplacian_vs_divgrad(A_pressure, nodes_coords, triangles)
#test_adjointness(nodes_coords, triangles, nodes_boundary_markers)
#run_checkerboard_test()
#test_laplacian_blind_spot(A_pressure, nodes_coords, triangles)
#test_gradient_of_checkerboard(nodes_coords, triangles)
#test_rhs_handling(A_stiffness, nodes_coords, triangles, inner_boundary_indices, dirichlet_node_indices)
#import sys
#sys.exit("\nExiting after diagnostic test.")

## 1. Enforce the initial boundary conditions to set up the starting state `u`
#u[wall_node_indices] = OUTER_BOUNDARY_VALUE
#for master_idx, slave_idx in pairs:
#    u[slave_idx] = u[master_idx]
#
#center_x, center_y = 0.5, 0.5
#angular_velocity = 5.0
#for idx in inner_boundary_indices:
#    # Vector from circle center to the boundary node
#    node_x, node_y = nodes_coords[idx]
#    radius_vec_x = node_x - center_x
#    radius_vec_y = node_y - center_y
#    # The tangential velocity vector is (-dy, dx)
#    tangential_vec_x = -radius_vec_y
#    tangential_vec_y =  radius_vec_x
#    # Set the velocity of the node
#    u[idx, 0] = tangential_vec_x * angular_velocity
#    u[idx, 1] = tangential_vec_y * angular_velocity
#
## 2. Run the diagnostic test on this initial state
#run_singlestep_ustar_diagnostic(A_visc, u, b_force, DT, pairs, nodes_coords, triangles)
#
## 3. Exit the script
#import sys
#sys.exit("\nExiting after Test I.")

## 1. Enforce the initial boundary conditions to set up the starting state `u`
#u[wall_node_indices] = OUTER_BOUNDARY_VALUE
#for master_idx, slave_idx in pairs:
#    u[slave_idx] = u[master_idx]
#
#center_x, center_y = 0.5, 0.5
#angular_velocity = 5.0
#for idx in inner_boundary_indices:
#    node_x, node_y = nodes_coords[idx]
#    radius_vec_x = node_x - center_x
#    radius_vec_y = node_y - center_y
#    tangential_vec_x = -radius_vec_y
#    tangential_vec_y =  radius_vec_x
#    u[idx, 0] = tangential_vec_x * angular_velocity
#    u[idx, 1] = tangential_vec_y * angular_velocity
#
## 2. We need to generate a `u_star` to test the pressure solve.
##    So we run the logic from Step 1 of the simulation.
#print("\n--- Generating u_star to be used as input for Test J ---")
#DT = 0.00001 # Make sure to use the small DT for the fine mesh
#rhs_x = u[:, 0] + DT * b_force[:, 0]
#rhs_y = u[:, 1] + DT * b_force[:, 1]
#u_star = np.zeros((N,2))
#u_star[:,0] = np.linalg.solve(A_visc, rhs_x)
#u_star[:,1] = np.linalg.solve(A_visc, rhs_y)
#for master_idx, slave_idx in pairs:
#    u_star[slave_idx] = u_star[master_idx]
#print("--- u_star generated successfully. ---")
#
## 3. Now, run Test J using the generated u_star
#run_singlestep_pressure_diagnostic(A_pressure, u_star, DT, nodes_coords, triangles, pressure_ref_node)
#
## 4. Exit the script
#import sys
#sys.exit("\nExiting after Test J.")

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

# --- NEW: Initialize tracer status (0 = purple/uneaten, 1 = red/eaten) ---
tracer_status = np.zeros(num_tracers, dtype=int)
tracer_colors = np.array(['blue', 'red'])

print(f"Initialized {num_tracers} tracer particles.")

## 5. Time-Stepping Loop (Projection Method)
print("\nStarting simulation...")

for step in range(STEPS):
    # --- Step 1: Tentative Velocity  ---

    # a) Start with the RHS from the previous step's solution
    rhs_x = u[:, 0] + DT * b_force[:, 0]
    rhs_y = u[:, 1] + DT * b_force[:, 1]

    
    # Set RHS for stationary outer walls
    #rhs_x[wall_node_indices] = OUTER_BOUNDARY_VALUE[0]
    #rhs_y[wall_node_indices] = OUTER_BOUNDARY_VALUE[1]

    # Set RHS for the rotating inner cylinder 
    center_x, center_y = 0.5, 0.5
    target_angular_velocity = 5.0
    ramp_up_steps = 200 

    if step < ramp_up_steps:
        current_angular_velocity = target_angular_velocity * (step + 1) / ramp_up_steps
    else:
        current_angular_velocity = target_angular_velocity

#    for idx in inner_boundary_indices:
#        node_x, node_y = nodes_coords[idx]
#        radius_vec_x = node_x - center_x
#        radius_vec_y = node_y - center_y
#        tangential_vec_x = -radius_vec_y
#        tangential_vec_y =  radius_vec_x
#        rhs_x[idx] = tangential_vec_x * current_angular_velocity
#        rhs_y[idx] = tangential_vec_y * current_angular_velocity
    
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

#    # --- Step 2: Pressure Correction (Poisson Equation) (v2) ---
#    # This is the corrected, consistent formulation.
#    
#    # a) Calculate the FVM-style divergence of the tentative velocity.
#    div_u_star = calculate_divergence(nodes_coords, triangles, u_star)
#    
#    # b) Build the RHS vector: b = (1/DT) * M_lumped * div(u_star)
#    # The lumped mass matrix M_lumped_diag bridges the FEM and FVM operators.
#    b_p = -(1.0 / DT) * div_u_star    
#
#    # c) Enforce the pressure reference condition on the RHS vector.
#    b_p[pressure_ref_node] = 0.0
#    
#    # d) Solve the FEM system Kp = b
#    p = np.linalg.solve(A_pressure, b_p)
#    p -= p.mean() # Remove any pressure drift for stability



    # --- Step 3: Velocity Update ---
    grad_px, grad_py = calculate_gradiant(nodes_coords, triangles, p)
    u[:,0] = u_star[:,0] - DT * grad_px
    u[:,1] = u_star[:,1] - DT * grad_py
    
    final_div = calculate_divergence(nodes_coords, triangles, u)
    
    # --- Step 4: Enforce Boundary Conditions ---
#    # Rotating squirmer
#    u[wall_node_indices] = OUTER_BOUNDARY_VALUE
#
#    for master_idx, slave_idx in pairs:
#        u[slave_idx] = u[master_idx]
#
#    for idx in inner_boundary_indices:
#        node_x, node_y = nodes_coords[idx]
#        radius_vec_x = node_x - center_x
#        radius_vec_y = node_y - center_y
#        tangential_vec_x = -radius_vec_y
#        tangential_vec_y =  radius_vec_x
#        u[idx, 0] = tangential_vec_x * current_angular_velocity
#        u[idx, 1] = tangential_vec_y * current_angular_velocity 


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


    final_div = calculate_divergence(nodes_coords, triangles, u)

    if step > 0:
        # final_div = calculate_divergence(nodes_coords, triangles, u)
        #print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Final div: {final_div}, Eaten (Red): {num_eaten}, Uneaten (Purple): {num_uneaten}")            
        print(f"Step: {step}, Max U: {np.max(np.abs(u)):.2e}, Max P: {np.max(np.abs(p)):.2e}, Div(u*): {np.max(np.abs(div_u_star)):.2e}, Final Div(u): {np.max(np.abs(final_div)):.2e}, Eaten (Red): {num_eaten}, Uneaten (Blue): {num_uneaten}")
    

#        # === Setup1 ===
#        # --- Clear all axes for the new frame ---
#        ax1.clear()
#        ax2.clear()
#        ax3.clear()
#
#        # === Plot 1: Velocity Magnitude and Streamlines ===
#        u_magnitude = np.linalg.norm(u, axis=1)
#        tpc1 = ax1.tripcolor(triang, u_magnitude, shading='gouraud', cmap='viridis', vmin=0, vmax=FIXED_VMAX)
#        interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
#        interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])
#        u_x_grid = interpolator_x(grid_x, grid_y)
#        u_y_grid = interpolator_y(grid_x, grid_y)
#        ax1.streamplot(grid_x, grid_y, u_x_grid, u_y_grid, color='black', linewidth=0.7, density=1.0)
#        ax1.set_title("Velocity (how fast and in what direction)")
#        ax1.set_aspect('equal')
#        # Create colorbar if it doesn't exist, otherwise update it
#        if cb1 is None:
#            cb1 = fig.colorbar(tpc1, ax=ax1, label="Velocity Magnitude (Speed)")
#        else:
#            cb1.update_normal(tpc1)
#
#        # === Plot 2: Pressure Field ===
#        tpc2 = ax2.tripcolor(triang, p, shading='gouraud', cmap='coolwarm')
#        ax2.set_title("Pressure (pressure gradient)")
#        ax2.set_aspect('equal')
#        # Create colorbar if it doesn't exist, otherwise update it
#        if cb2 is None:
#            cb2 = fig.colorbar(tpc2, ax=ax2, label="Pressure")
#        else:
#            cb2.update_normal(tpc2)
#
#        # === Plot 3: Vorticity Field ===
#        vorticity = calculate_vorticity(nodes_coords, triangles, u)
#        vort_max = np.max(np.abs(vorticity)) if np.max(np.abs(vorticity)) > 1e-9 else 1.0
#        tpc3 = ax3.tripcolor(triang, vorticity, shading='gouraud', cmap='seismic', vmin=-vort_max, vmax=vort_max)
#        ax3.set_title("Vorticity (local rotation or 'spin' of the fluid)")
#        ax3.set_aspect('equal')
#        # Create colorbar if it doesn't exist, otherwise update it
#        if cb3 is None:
#            cb3 = fig.colorbar(tpc3, ax=ax3, label="Vorticity (Curl)")
#        else:
#            cb3.update_normal(tpc3)
#
#        # --- Finalize and draw the figure ---
#        fig.suptitle(f"Time Evolution: Step {step}/{STEPS}", fontsize=16)
#        fig.canvas.draw_idle()
#        if step == 5:
#            plt.pause(1.0)
#        plt.pause(0.01)
        
#        # === Setup2 ===
#        # Update plot
#        ax.clear()
#        u_magnitude = np.linalg.norm(u, axis=1)
#        tpc = ax.tripcolor(triang, u_magnitude, shading='gouraud', cmap='viridis', vmin=0, vmax=FIXED_VMAX)
#
#
#        # Draw streamlines
#        interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
#        interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])
#        u_x_grid = interpolator_x(grid_x, grid_y)
#        u_y_grid = interpolator_y(grid_x, grid_y)
#        ax.streamplot(grid_x, grid_y, u_x_grid, u_y_grid, color='black', linewidth=1, density=0.7)
#
#        if colorbar is None:
#            colorbar = fig.colorbar(tpc, ax=ax)
#            colorbar.set_label("Velocity Magnitude")
#        else:
#            colorbar.update_normal(tpc)        
#
#        ax.set_aspect('equal')
#        ax.set_title(f"Time Evolution: Step {step}/{STEPS}")
#        fig.canvas.draw_idle()
#        plt.pause(0.01)

#        # === Setup3 (2) ====
#        ax.clear()
#
#        # 1. Calculate Vorticity and find a symmetric color range
#        vorticity = calculate_vorticity(nodes_coords, triangles, u)
#        vort_max = np.max(np.abs(vorticity)) if np.max(np.abs(vorticity)) > 1e-9 else 1.0
#        
#        # 2. Plot Vorticity as the colored background
#        #    'seismic' is a great colormap for this (blue for negative, red for positive)
#        tpc = ax.tripcolor(triang, vorticity, shading='gouraud', cmap='seismic', 
#                          vmin=-vort_max, vmax=vort_max)
#
#        # 3. Interpolate Velocity onto a grid for the streamlines
#        interpolator_x = mtri.LinearTriInterpolator(triang, u[:, 0])
#        interpolator_y = mtri.LinearTriInterpolator(triang, u[:, 1])
#        u_x_grid = interpolator_x(grid_x, grid_y)
#        u_y_grid = interpolator_y(grid_x, grid_y)
#
#        # 4. Overlay the Streamlines
#        #    'density' controls how many streamlines are drawn.
#        ax.streamplot(grid_x, grid_y, u_x_grid, u_y_grid, color='black', 
#                      linewidth=1.0, density=1.5)
#
#        # 5. Manage the Colorbar
#        if colorbar is None:
#            colorbar = fig.colorbar(tpc, ax=ax)
#            colorbar.set_label("Vorticity (Fluid Spin)")
#        else:
#            colorbar.update_normal(tpc)
#
#        # 6. Final Touches
#        ax.set_aspect('equal')
#        ax.set_title(f"Squirmer Flow Field: Step {step}/{STEPS}")
#        fig.canvas.draw_idle()
#        plt.pause(0.01)

#        # === Setup 4 (2) ===
#        ax.clear()
#
#        # 1. Plot the velocity magnitude as a background color field
#        u_magnitude = np.linalg.norm(u, axis=1)
#        tpc = ax.tripcolor(triang, u_magnitude, shading='gouraud', cmap='viridis', 
#                          vmin=0, vmax=FIXED_VMAX)
#
#        # 2. Prepare the node indices for the quiver plot
#        # We'll plot vectors for all surface nodes and a sparse subset of interior nodes.
#        skip_interior = 30 # Plot a vector for every 30th interior node.
#        interior_indices = np.where(nodes_boundary_markers == 0)[0][::skip_interior]
#        
#        # Combine the inner boundary nodes and the selected interior nodes
#        quiver_indices = np.union1d(inner_boundary_indices, interior_indices)
#        
#        # 3. Overlay the quiver plot (the velocity vectors)
#        #    The 'scale' parameter adjusts arrow length; larger scale = shorter arrows.
#        ax.quiver(nodes_coords[quiver_indices, 0], nodes_coords[quiver_indices, 1],
#                  u[quiver_indices, 0], u[quiver_indices, 1],
#                  color='white', scale=40.0)
#
#        # 4. Manage the Colorbar
#        if colorbar is None:
#            colorbar = fig.colorbar(tpc, ax=ax)
#            colorbar.set_label("Velocity Magnitude")
#        else:
#            colorbar.update_normal(tpc)
#
#        # 5. Final Touches
#        ax.set_aspect('equal')
#        ax.set_title(f"Squirmer Velocity Field: Step {step}/{STEPS}")
#        # A dark background helps the white arrows stand out
#        ax.set_facecolor('black') 
#        fig.canvas.draw_idle()
#        plt.pause(0.01)

        # --- UPDATE VISUALIZATION ---
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

        # --- MODIFIED: Plot tracer particles with their corresponding color ---
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
