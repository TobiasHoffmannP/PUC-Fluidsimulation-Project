import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List, Set
from jax.experimental import sparse
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

# read the .ele, node, poly files

NODE_FILEPATH = "/home/tobias/FluidMixing/mesh2.2.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/mesh2.2.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/mesh2.2.poly"

OUTER_BOUNDARY_MARKER = 2
INNER_BOUNDARY_MARKER = 1

OUTER_BOUNDARY_VALUE = 0.0
INNER_BOUNDARY_VALUE = 1.0




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

def readEle_P2(filepath): # Renamed to be specific
    with open(filepath, "r") as eleFile:
        header = eleFile.readline().strip().split()
        number_of_triangles = int(header[0])
        
        # Each triangle now has 6 nodes for P2 elements
        elements = np.zeros((number_of_triangles, 6), dtype=np.int32)

        for i in range(number_of_triangles):
            line = eleFile.readline().strip().split()
            # The node indices are columns 1 through 6
            elements[i, 0] = int(line[1]) - 1
            elements[i, 1] = int(line[2]) - 1
            elements[i, 2] = int(line[3]) - 1
            elements[i, 3] = int(line[4]) - 1
            elements[i, 4] = int(line[5]) - 1
            elements[i, 5] = int(line[6]) - 1

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



def get_dof_indices(node_indexes, num_nodes):

    if not (0 <= node_indexes < num_nodes):
        raise ValueError("node_idx is out of bounds.")
        
    u_x_dof = node_indexes
    u_y_dof = num_nodes + node_indexes
    p_dof = 2 * num_nodes + node_indexes
    
    return u_x_dof, u_y_dof, p_dof



def build_stokes_stiffness_matrix(nodes, triangles, nu, get_dof_indices_func):
    N = nodes.shape[0]
    total_dof = 3 * N
    K_Matrix = np.zeros((total_dof, total_dof), dtype=np.float64)

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

        y_diffs = np.array([y2 - y3, y3 - y1, y1 - y2])
        x_diffs = np.array([x3 - x2, x1 - x3, x2 - x1])

        for i in range(3):
            for j in range(3):

                scalar_stiffness = (y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j]) / (2.0 * A_det)

                viscous_term = nu * scalar_stiffness

                u_x_i_dof, u_y_i_dof, _ = get_dof_indices_func(tri[i], N)
                u_x_j_dof, u_y_j_dof, _ = get_dof_indices_func(tri[j], N)

                K_Matrix[u_x_i_dof, u_x_j_dof] += viscous_term
                K_Matrix[u_y_i_dof, u_y_j_dof] += viscous_term

    return K_Matrix

    
def build_pressure_coupling_matrix(nodes, triangles, get_dof_indices_func):
    N = nodes.shape[0]
    total_dof = 3 * N
    Coupling_Matrix = np.zeros((total_dof, total_dof), dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri
        x1, y1 = nodes[p1]
        x2, y2 = nodes[p2]
        x3, y3 = nodes[p3]

        y_diffs = np.array([y2 - y3, y3 - y1, y1 - y2])
        x_diffs = np.array([x3 - x2, x1 - x3, x2 - x1])

        for i in range(3):  # This index is for the pressure basis function, phi_i
            for j in range(3):  # This index is for the velocity basis function, phi_j
                
                # This calculation comes from the integral of -(p * div(u))
                # which simplifies to the gradient components divided by 6.
                b_x_val = -y_diffs[j] / 6.0
                b_y_val = -x_diffs[j] / 6.0

                              
                _, _, p_i_dof = get_dof_indices_func(tri[i], N)
                u_x_j_dof, u_y_j_dof, _ = get_dof_indices_func(tri[j], N)
                
                # Assemble into the B block (pressure rows, velocity columns)
                Coupling_Matrix[p_i_dof, u_x_j_dof] += b_x_val
                Coupling_Matrix[p_i_dof, u_y_j_dof] += b_y_val
                
                # Assemble into the B^T block (velocity rows, pressure columns)
                Coupling_Matrix[u_x_j_dof, p_i_dof] += b_x_val
                Coupling_Matrix[u_y_j_dof, p_i_dof] += b_y_val
                
    return Coupling_Matrix


# =============================================================================
# --- MAIN EXECUTION ---
# =============================================================================

# --- Parameters ---
nu = 1.0          # Viscosity
U0 = 1.0          # Squirmer swimming speed

# --- Load Mesh and DoF Info ---
nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
num_nodes = nodes_coords.shape[0]
triangles = readEle(ELE_FILEPATH)
total_dofs = 3 * num_nodes

# --- Build and Assemble the System Matrix ---
print("Building matrices...")
K_matrix = build_stokes_stiffness_matrix(nodes_coords, triangles, nu, get_dof_indices)
Coupling_Matrix = build_pressure_coupling_matrix(nodes_coords, triangles, get_dof_indices)

# The full Stokes matrix A is the sum of the two parts
A_stokes = K_matrix + Coupling_Matrix

# --- Build the Right-Hand-Side (RHS) Vector and Apply BCs ---
print("Applying boundary conditions...")
b_stokes = np.zeros(total_dofs, dtype=np.float64)

# Get the indices of the boundary nodes
inner_bnd_indices, outer_bnd_indices = get_boundary_indices(
    nodes_boundary_markers,
    INNER_BOUNDARY_MARKER,
    OUTER_BOUNDARY_MARKER
)

# Apply "no-slip" (u=0, v=0) condition on the outer boundary
for node_idx in outer_bnd_indices:
    u_x_dof, u_y_dof, _ = get_dof_indices(node_idx, num_nodes)
    
    # Enforce u_x = 0
    A_stokes[u_x_dof, :] = 0
    A_stokes[u_x_dof, u_x_dof] = 1.0
    b_stokes[u_x_dof] = 0.0
    
    # Enforce u_y = 0
    A_stokes[u_y_dof, :] = 0
    A_stokes[u_y_dof, u_y_dof] = 1.0
    b_stokes[u_y_dof] = 0.0

# Apply "squirmer" velocity condition on the inner boundary
for node_idx in inner_bnd_indices:
    u_x_dof, u_y_dof, _ = get_dof_indices(node_idx, num_nodes)
    x, y = nodes_coords[node_idx]
    
    # Normalize position to be on a unit circle for clean tangential velocity
    r = np.sqrt(x**2 + y**2)
    if r == 0: r = 1.0 # Avoid division by zero at origin
    
    # Tangential velocity u = U0 * (-y/r, x/r)
    u_x_squirmer = -U0 * y / r
    u_y_squirmer =  U0 * x / r
    
    # Enforce u_x = u_x_squirmer
    A_stokes[u_x_dof, :] = 0
    A_stokes[u_x_dof, u_x_dof] = 1.0
    b_stokes[u_x_dof] = u_x_squirmer
    
    # Enforce u_y = u_y_squirmer
    A_stokes[u_y_dof, :] = 0
    A_stokes[u_y_dof, u_y_dof] = 1.0
    b_stokes[u_y_dof] = u_y_squirmer

# Pin the pressure at the first node to 0 to ensure a unique solution
_, _, p_dof_0 = get_dof_indices(0, num_nodes)
A_stokes[p_dof_0, :] = 0
A_stokes[p_dof_0, p_dof_0] = 1.0
b_stokes[p_dof_0] = 0.0


# If you get to here, the system is fully built!
print("System built successfully!")


# =============================================================================
# --- 5. SOLVE AND VISUALIZE ---
# =============================================================================
print("Solving the linear system...")
# Convert the final system to JAX arrays for the solve step
A_jax = jnp.array(A_stokes)
b_jax = jnp.array(b_stokes)

# Solve the system A*X = b to find the solution vector X
solution_vector = jnp.linalg.solve(A_jax, b_jax)

# --- Unpack the Solution Vector ---
# The solution vector contains all DoFs, so we need to slice it
# to get our separate physical fields.
u_x = solution_vector[0:num_nodes]
u_y = solution_vector[num_nodes:2*num_nodes]
p = solution_vector[2*num_nodes:3*num_nodes]
print("System solved. Unpacking results for plotting.")


# --- Create the plots ---
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Pressure field (scalar plot)
ax1.set_title("Pressure Field (p)")
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang, p, shading='gouraud', cmap='viridis')
fig.colorbar(tpc, ax=ax1, label='Pressure')

# Plot 2: Velocity field (vector plot)
ax2.set_title("Velocity Field (u)")
ax2.set_aspect('equal')
# Use quiver to plot vectors (arrows) at each node
# To avoid clutter, we can plot a vector at every Nth node using a stride.
stride = 5 
ax2.quiver(nodes_coords[::stride, 0], nodes_coords[::stride, 1], 
           u_x[::stride], u_y[::stride])
# Also plot the inner and outer boundaries for context
ax2.triplot(triang, 'k-', lw=0.5, alpha=0.5)


# Set the main title for the entire figure
fig.suptitle("Stokes Flow Solution for a Squirmer")

# Adjust layout to prevent titles and labels from overlapping.
# The `rect` parameter leaves space at the top for the suptitle.
plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

# Display the final plot
plt.show()

print("\nSimulation and visualization complete!")






