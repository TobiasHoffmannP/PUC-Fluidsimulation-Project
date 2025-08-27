import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# --- You will need to copy these four functions from your script into this new file ---
# readNode, readEle, buildFemSystem, apply_dirichlet_to_all_boundaries
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

def apply_dirichlet_to_all_boundaries(A, rhs_x, rhs_y, nodes_coords, nodes_boundary_markers, H=1.0, tol=1e-6):
    """
    Modifies a system matrix and RHS vectors to enforce no-slip (u=0)
    Dirichlet conditions on all four outer walls and the inner boundary.
    """
    N = nodes_coords.shape[0]
    
    for i in range(N):
        # Get the node's properties
        x_coord, y_coord = nodes_coords[i]
        marker = nodes_boundary_markers[i]

        # Check if the node is on any boundary
        is_outer_wall = (
            np.abs(x_coord - 0.0) < tol or
            np.abs(x_coord - 1.0) < tol or
            np.abs(y_coord - 0.0) < tol or
            np.abs(y_coord - H) < tol
        )
        is_inner_wall = (marker == INNER_BOUNDARY_MARKER)

        # If the node is on any wall, apply the Dirichlet condition
        if is_outer_wall or is_inner_wall:
            # Modify the matrix row to enforce the condition (e.g., 1 * u_i = 0)
            A[i, :] = 0.0
            A[i, i] = 1.0
            
            # Set the corresponding RHS value to the desired boundary value (0 for no-slip)
            rhs_x[i] = 0.0
            rhs_y[i] = 0.0

def apply_dirichlet_u_to_all_walls(u, nodes_coords, nodes_boundary_markers, H=1.0, tol=1e-6):
    """Strongly enforces u=0 on all boundary nodes."""
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


# --- Script to solve a simple Poisson equation ---

# 1. Load the mesh
OUTER_BOUNDARY_MARKER = 1
INNER_BOUNDARY_MARKER = 2

OUTER_BOUNDARY_VALUE = 1.0
INNER_BOUNDARY_VALUE = 0.0

NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
triangles = readEle(ELE_FILEPATH)
N = nodes_coords.shape[0]

# 2. Define a simple source term (f = -1 everywhere)
# For the equation -∇²u = f, this will be the 'b' vector.
# We pass g_source=1 to buildFemSystem, which calculates ∫fφ, giving us the RHS vector.
A_matrix, b_vector = buildFemSystem(nodes_coords, triangles, g_source=1.0)


# 3. Apply Zero Dirichlet BCs on all boundaries
# This solves for u=0 on all walls.
# We create dummy RHS vectors because the function expects them, but we only need the modified A_matrix.
apply_dirichlet_to_all_boundaries(A_matrix, np.zeros(N), np.zeros(N), nodes_coords, nodes_boundary_markers)

# Set the RHS for boundary nodes to 0, consistent with the matrix modification
for i in range(N):
    x_coord, y_coord = nodes_coords[i]
    marker = nodes_boundary_markers[i]
    is_outer_wall = (np.abs(x_coord-0.0)<1e-6 or np.abs(x_coord-1.0)<1e-6 or np.abs(y_coord-0.0)<1e-6 or np.abs(y_coord-1.0)<1e-6)
    is_inner_wall = (marker == 2) # Assuming INNER_BOUNDARY_MARKER is 2
    if is_outer_wall or is_inner_wall:
        b_vector[i] = 0.0


# 4. Solve the simple system Au = b
try:
    u_solution = np.linalg.solve(A_matrix, b_vector)
    print("Poisson equation solved successfully.")

    # 5. Plot the result
    fig, ax = plt.subplots(figsize=(8, 8))
    triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
    tpc = ax.tripcolor(triang, u_solution, shading='gouraud', cmap='viridis')
    fig.colorbar(tpc, label="Solution u")
    ax.set_title("Solution to Poisson Equation ∇²u = -1")
    ax.set_aspect('equal')
    plt.show()

except np.linalg.LinAlgError as e:
    print(f"CRITICAL ERROR: Could not solve the simple Poisson system.")
    print(f"The matrix A_matrix is likely singular or ill-formed. Error: {e}")
