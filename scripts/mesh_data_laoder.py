import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, List, Set
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

# --- Constants and Filepaths ---
NODE_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.poly"

OUTER_BOUNDARY_MARKER = 2 # The PDF uses '2' for the outer boundary
INNER_BOUNDARY_MARKER = 1 # The PDF uses '1' for the inner boundary

OUTER_BOUNDARY_VALUE = 0.0
INNER_BOUNDARY_VALUE = 1.0

# --- File Reading Functions (Unchanged) ---
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
            
            # X coord
            nodes_coords[index, 0] = float(line[1])

            # Y coord
            nodes_coords[index, 1] = float(line[2])

            # boundary marker 
            if int(line[3]) != 0:
                nodes_boundary_markers[index] = int(line[3])

        return nodes_coords, nodes_boundary_markers

def readEle(filepath):
    with open(filepath, "r") as eleFile:
        header = eleFile.readline().strip().split()
        number_of_triangles = int(header[0])
        
        elements = np.zeros((number_of_triangles, 3), dtype=np.int32)

        for i in range(int(number_of_triangles)):
            line = eleFile.readline().strip().split()
            
            elements[i, 0] = int(line[1]) - 1
            elements[i, 1] = int(line[2]) - 1
            elements[i, 2] = int(line[3]) - 1

        return elements

# --- FEM System Builder ---
def build_fem_system(nodes, triangles, g_source=1.0):
    """
    Builds the stiffness matrix A and load vector b, following the formulas
    from the source material as closely as possible.
    """
    N = nodes.shape[0]
    A_matrix = np.zeros((N, N), dtype=np.float64)
    b_vector = np.zeros(N, dtype=np.float64)

    # Loop over each triangle element in the mesh
    for tri_nodes in triangles:
        
        # Define vertex coordinates p1, p2, p3 for the current triangle
        p1_idx, p2_idx, p3_idx = tri_nodes
        x1, y1 = nodes[p1_idx]
        x2, y2 = nodes[p2_idx]
        x3, y3 = nodes[p3_idx]
        
        # Calculate the determinant 'A', which is twice the triangle's area 
        A_det = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        
        if A_det == 0: continue

        # The final FEM system is of the form -Af = b_g, where A contains the integrals
        # of the basis function gradients and b_g contains the integrals of the source term.
        # This is equivalent to Af = -b_g. We will calculate the positive integral for b_g
        # and apply the negative sign at the end. 

        # Assemble the local stiffness matrix K_local for this triangle
        # The terms in these lists correspond to (y_2 - y_3), (x_3 - x_2), etc.
        y_diffs = [y2 - y3, y3 - y1, y1 - y2]
        x_diffs = [x3 - x2, x1 - x3, x2 - x1]

        for i in range(3):
            for j in range(3):
                # This calculates the integral of the dot product of the gradients of
                # basis functions phi_i and phi_j over the current triangle. 
                # Note: The PDF's 'A' is our 'A_det'. The formula's 1/(2A) is 1/(2 * A_det).
                # The code calculates K_local for Af = b_g, and the minus sign from Eq. 4.64
                # is handled later.

                numerator = y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j]
                integral_val = numerator / (2.0 * A_det)
                
                # Assemble into the global matrix A
                global_row_idx = tri_nodes[i]
                global_col_idx = tri_nodes[j]
                A_matrix[global_row_idx, global_col_idx] += integral_val
        
        # Assemble the source term vector b for this triangle
        # This approximates the integral of g*phi_j over the triangle 
        # For a constant g, this is (g * Area) / 3 for each node.
        area = 0.5 * A_det
        source_integral_val = g_source * area / 3.0
        
        # Add the contribution to each of the three nodes
        b_vector[p1_idx] += source_integral_val
        b_vector[p2_idx] += source_integral_val
        b_vector[p3_idx] += source_integral_val
    
    # Per Eq. 4.64, the final system is -Af = b, which is equivalent to Af = -b
    return A_matrix, -b_vector


# --- Main Execution Logic ---

# 1. Read Mesh Data
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
triangles = readEle(ELE_FILEPATH)
N = nodes_coords.shape[0]

# 2. Build the FEM system A and b
A, b = build_fem_system(nodes_coords, triangles, g_source=-4.0) # Using g=-4 as an example

# 3. Apply Dirichlet Boundary Conditions
for i in range(N):
    marker = nodes_boundary_markers[i]
    if marker != 0:
        # For boundary nodes, enforce the condition f_i = value
        # Set the row to 0, with 1 on the diagonal
        A[i, :] = 0
        A[i, i] = 1.0
        
        # Set the b vector to the desired boundary value
        if marker == INNER_BOUNDARY_MARKER:
            b[i] = INNER_BOUNDARY_VALUE
        elif marker == OUTER_BOUNDARY_MARKER:
            b[i] = OUTER_BOUNDARY_VALUE

# 4. Solve the Linear System using JAX
Ajp = jnp.array(A)
bjp = jnp.array(b)
f = jnp.linalg.solve(Ajp, bjp)

print(f"System solved successfully: {jnp.allclose(Ajp @ f, bjp, rtol=1e-5, atol=1e-5)}")

# 5. Plot the Result
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
plt.figure(figsize=(6, 6))
tpc = plt.tripcolor(triang, f, shading='gouraud', cmap='viridis') # Gouraud shading looks smoother
plt.colorbar(tpc, label='Solution f(x, y)')
plt.gca().set_aspect('equal')
plt.title("FEM Solution to Poisson's Equation (∇²f = g)")
plt.show()
