import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List, Set
from jax.experimental import sparse
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

# read the .ele, node, poly files

NODE_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.poly"

OUTER_BOUNDARY_MARKER = 2
INNER_BOUNDARY_MARKER = 1

OUTER_BOUNDARY_VALUE = 0.0
INNER_BOUNDARY_VALUE = 1.0

FIRST_EQ = False
POISSON = True


def readNode(filepath):
    """ 
        Prepares the data for processing

        Returns:
            nodes_coords: Array over the coordinates (x, y) of 
            nodes with the index being the node number()
    """

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
    """
    Read a .poly file
    Returns a N*2 matrix, each line is a segments containg 2 0-indexed indicies from the coordinate matrix
    Also returns a boundary marker for each segment
    """
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

def identify_boundary_marker(nodes, boundaryMarkers):
    N = nodes.shape[0]
    innerBoundaries = np.zeros(N, dtype=np.float64)


def buildFemSystem(nodes, triangles, g_source=1.0):
    N = nodes.shape[0]
    AMatrix = np.zeros((N,N), dtype=np.float64)
    BVector = np.zeros(N, dtype=np.float64)

    for triangle_nodes in triangles:
        p1, p2, p3 = triangle_nodes
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
                AMatrix[triangle_nodes[i], triangle_nodes[j]] += integral_val

        area = 0.5 * ADet
        
        # pyramid over triangle is 1 / 3 of the area
        # bj = ∫Ω g(x, y) ϕj(x, y) dΩ = g * Area / 3
        sourceIntergralValue = g_source * area / 3
        
        BVector[p1] += sourceIntergralValue
        BVector[p2] += sourceIntergralValue
        BVector[p3] += sourceIntergralValue

    return AMatrix, -BVector

def buildMassMatrix(nodes, triangles):
    N = nodes.shape[0]
    mass_matrix = np.zeros((N, N), dtype=np.float64)
    
    local_mass_matrix_template = np.array([[2, 1, 1],
                                           [1, 2, 1],
                                           [1, 1, 2]])

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]

        A_det = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2
        area = 0.5 * A_det

        if area == 0:
            continue

        local_mass = (area / 12) * local_mass_matrix_template 
        
        for i in range(3):
            for j in range(3):
                mass_matrix[tri[i], tri[j]] += local_mass[i, j]

    return mass_matrix

def buildAdvectionMatrix(nodes, triangles, c_velocity):
    N = nodes.shape[0]
    C_Matrix = np.zeros((N, N), dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]

        # ∇
        A_det = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2

        area = 0.5 * A_det
         
        if A_det == 0: 
            continue

        # print(triangles.shape[1])

        y_diffs = np.array([y2 - y3, y3 - y1, y1 - y2])
        x_diffs = np.array([x3 - x2, x1 - x3, x2 - x1])

        # ∇ϕ_j (rate of change)
        grad = (1.0 / A_det) * np.vstack((y_diffs, x_diffs)).T

        # c⋅∇ϕ_j 
        c_dot_grad = np.dot(grad, c_velocity)

        integral_phi = area / 3.0
        

        for i in range(3):
            for j in range(3):
                # C_ij^e = (c ⋅ ∇ϕ_j) Area / 3 
                C_Matrix[tri[i], tri[j]] += c_dot_grad[j] * integral_phi

    return C_Matrix

def get_boundary_indices(nodes_boundary_markers, inner_marker, outer_marker):
    """
    Finds the indices of nodes that lie on the inner and outer boundaries.

    Args:
        nodes_boundary_markers (np.ndarray): The array of boundary markers for each node.
        inner_marker (int): The integer value marking the inner boundary.
        outer_marker (int): The integer value marking the outer boundary.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                       (inner_boundary_indices, outer_boundary_indices)
    """
    inner_indices = np.where(nodes_boundary_markers == inner_marker)[0]
    outer_indices = np.where(nodes_boundary_markers == outer_marker)[0]
    
    return inner_indices, outer_indices



# -- Simulation Parametersdt = 0.005                # Time step size
dt = 0.001
T_final = 1.0            # Final simulation time
nu = 0.01                # Diffusion coefficient (controls spreading)
c_velocity = np.array([5.0, 5.0]) # Advection velocity [cx, cy]



nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
N = nodes_coords.shape[0] 
triangles = readEle(ELE_FILEPATH)
print(nodes_boundary_markers)

inner_bnd_indices, outer_bnd_indices = get_boundary_indices(
    nodes_boundary_markers, 
    INNER_BOUNDARY_MARKER, 
    OUTER_BOUNDARY_MARKER
)

# -- DEFINE THE SOURCE TERM --
# 1. Find all nodes on the outer boundary to act as the source
outer_boundary_indices = np.where(nodes_boundary_markers == OUTER_BOUNDARY_MARKER)[0]
if len(outer_boundary_indices) == 0:
    raise ValueError("No nodes found for the outer boundary marker.")

# 2. Create the source vector 'b'
source_vector = np.zeros(N, dtype=np.float64)
source_strength = 50.0  # Controls how much is injected per second from each node
source_vector[outer_boundary_indices] = source_strength
print(f"Source is applied to {len(outer_boundary_indices)} nodes on the outer boundary.")


# -- Build the constant matrices
# Note: K is the stiffness matrix, so we ignore the source vector 'b' for now.
K, _ = buildFemSystem(nodes_coords, triangles, g_source=0.0) 
M = buildMassMatrix(nodes_coords, triangles)
C = buildAdvectionMatrix(nodes_coords, triangles, c_velocity)

# -- Set Initial Condition
# Start with f=0 everywhere, except for a small "blob" in the middle.
f_current = np.zeros(N, dtype=np.float64)
center_node_index = np.argmin(np.linalg.norm(nodes_coords - np.array([0.80, 0.50]), axis=1))
f_current[center_node_index] = 10.0 # High concentration at one point

center_node_index2 = np.argmin(np.linalg.norm(nodes_coords - np.array([0.75, 0.75]), axis=1))
f_current[center_node_index2] = 10.0 # High concentration at one point





# Pre-build the constant part of the system matrix
SystemMatrix = M + dt * (nu * K + C)

# =============================================================================
# --- 2. TIME-STEPPING AND VISUALIZATION ---
# =============================================================================

# --- Plotting Setup ---
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
plt.ion() # Turn on interactive mode
fig, ax = plt.subplots(figsize=(7, 7))

# Create the plot and colorbar ONCE before the loop
# The plot is initialized with the starting state (f_current)
tpc = ax.tripcolor(triang, f_current, shading='gouraud', cmap='viridis', vmin=0, vmax=10)
fig.colorbar(tpc, ax=ax, label='Concentration f(x, y)')
ax.set_aspect('equal')


# --- Main Loop ---
for t in np.arange(0, T_final, dt):
    print(f"Solving for time t = {t:.4f}")
    center_node_index2 = np.argmin(np.linalg.norm(nodes_coords - np.array([0.0, 0.20]), axis=1))
    f_current[center_node_index2] = 10.0 # High concentration at one point

    center_node_index2 = np.argmin(np.linalg.norm(nodes_coords - np.array([0.20, 0.0]), axis=1))
    f_current[center_node_index2] = 10.0 # High concentration at one point

    
    # 1. Assemble the system for this time step
    A_system = SystemMatrix.copy()
    b_system = M @ f_current
    
    # 2. Apply Dirichlet Boundary Conditions
    for i in range(N):
        if nodes_boundary_markers[i] != 0:
            A_system[i, :] = 0
            A_system[i, i] = 1.0
            b_system[i] = 0.0
    
    # 3. Solve for the next time step
    A_jax = jnp.array(A_system)
    b_jax = jnp.array(b_system)
    f_next = jnp.linalg.solve(A_jax, b_jax)
    
    # 4. Update the current state and convert back to NumPy
    f_current = np.array(f_next)
    
    # 5. UPDATE the plot data instead of redrawing the whole thing
    tpc.set_array(f_current)
    ax.set_title(f"Solution at t = {t+dt:.2f} s")
    fig.canvas.draw()  # Redraw the canvas
    plt.pause(0.01)


# --- Finalize Plotting ---
ax.set_title(f"Final Solution at t = {T_final:.2f} s")
plt.ioff() # Turn off interactive mode
plt.show() # Display the final result

print("\nSimulation finished!")


