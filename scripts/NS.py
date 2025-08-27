import numpy as np
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

# read the .ele, .node, .poly files

NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.poly"


OUTER_BOUNDARY_MARKER = 1

# RIGHT_BOUNDARY_MARKER = 1
# LEFT_BOUNDARY_MARKER = 2
# TOP_BOTTOM_BOUNDARY_MARKER = 3
# INNER_BOUNDARY_MARKER = 4
INNER_BOUNDARY_MARKER = 2


OUTER_BOUNDARY_VALUE = [1.0,0.0]
# TOP_BOTTOM_BOUNDARY_VALUE = 1.0
INNER_BOUNDARY_VALUE = [0.0,0.0]


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


def buildFemSystem(nodes, triangles, g_source=1.0):
    N = nodes.shape[0]
    AMatrix = np.zeros((N, N), dtype=np.float64)
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
                integral_val = (y_diffs[i] * y_diffs[j] + x_diffs[i] * x_diffs[j]) / (
                    2.0 * ADet
                )

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

        BVector[p1] += sourceIntergralValue
        BVector[p2] += sourceIntergralValue
        BVector[p3] += sourceIntergralValue

    return AMatrix, -BVector

def calculate_divergence(nodes, triangles, u_star):
    """
    u_star : shape (2, N)   – u_star[0,:] = u_x  , u_star[1,:] = u_y
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
        inv2A = 1.0 / det
        inv2A = 1.0 / det          # = 1/(2·area)
        area  = 0.5 * det

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
    Modifies the system matrix A and vector b in-place to enforce
    periodic boundary conditions.

    For each pair (master_idx, slave_idx) in `pairs`, this function
    enforces the constraint x[slave_idx] = x[master_idx].

    Args:
        A (np.ndarray): The system matrix (e.g., stiffness matrix).
        b (np.ndarray): The right-hand side vector (e.g., load vector).
        pairs (list[tuple[int, int]]): A list of (master_idx, slave_idx)
                                       node index pairs.
    """
    #   left set    right set
    for master_idx, slave_idx in pairs:
        # 1. Add the slave's row contributions to the master's row
        A[master_idx, :] += A[slave_idx, :]
        A[:, master_idx] += A[:, slave_idx]
        b[master_idx] += b[slave_idx]

        # 2. Clear the slave's row to create the constraint equation
        A[slave_idx, :] = 0.0

        # 3. Enforce the constraint: 1*x_slave - 1*x_master = 0
        A[slave_idx, slave_idx] = 1.0
        A[slave_idx, master_idx] = -1.0
        b[slave_idx] = 0.0


# TUBE_WALL_MARKER = 3 # A NEW marker for top/bottom walls
# Assuming a value for the wall, usually 0.0
# WALL_VALUE = 0.0


nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
N = nodes_coords.shape[0]
triangles = readEle(ELE_FILEPATH)
pairs = find_boundary_pairs(nodes_coords, L=1.0)
for pair in pairs:
    xcoord = nodes_coords[pair[0]]
    ycoord = nodes_coords[pair[1]]
    print(f"{xcoord}, {ycoord}")


pairs = find_boundary_pairs(nodes_coords, L=1.0)

# Define domain height and a tolerance for coordinate checks
H = 1.0
tol = 1e-6


def g_source_fun(x, y):
    return 50 * np.sin(3 * y)


A, _ = buildFemSystem(nodes_coords, triangles, g_source=g_source_fun)
Ap = A.copy()
b = np.zeros((N,2))

# filter out the corner points.
filtered_pairs = []
for master_idx, slave_idx in pairs:
    master_y = nodes_coords[master_idx, 1]
    if not (np.abs(master_y - 0.0) < tol or np.abs(master_y - H) < tol):
        filtered_pairs.append((master_idx, slave_idx))

print(
    f"Original pairs: {len(pairs)}, Filtered pairs (excluding walls): {len(filtered_pairs)}"
)


apply_periodic_bc(A, b, filtered_pairs)

apply_periodic_bc(Ap, np.zeros(N), filtered_pairs)

ref = 0                                    # remove null-space
Ap[ref,:] = 0.0;  Ap[ref,ref] = 1.0

WALL_VALUE = 0.0  # Define the value for the top/bottom walls

for i in range(N):
    marker = nodes_boundary_markers[i]
    x_coord, y_coord = nodes_coords[i]

    # A node is a wall node if its y-coordinate is 0 or H
    is_wall = np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol

    # A node is an inner boundary node if it has the marker
    is_inner_boundary = marker == INNER_BOUNDARY_MARKER

    # Apply fixed-value conditions if the node is on any Dirichlet boundary
    if is_wall or is_inner_boundary:
        # Set the matrix row to enforce f_i = value
        A[i, :] = 0.0
        A[i, i] = 1.0

        # Set the corresponding value in the b vector
        if is_inner_boundary:
            b[i] = INNER_BOUNDARY_VALUE
        elif is_wall:
            b[i] = OUTER_BOUNDARY_VALUE


def reapply_dirchlect_u(u):
    for i in range(N):
        marker = nodes_boundary_markers[i]
        _, y_coord = nodes_coords[i]

        is_wall = np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol
        is_inner_boundary = marker == INNER_BOUNDARY_MARKER

        if is_wall or is_inner_boundary:
            if is_inner_boundary:
                u[i] = INNER_BOUNDARY_VALUE
            elif is_wall:
                u[i] = OUTER_BOUNDARY_VALUE
    return u


def reapply_periodic_u(u):
    for master, slave in pairs:
        u[slave] = u[master]
    return u


DT = 0.0001
v = 0.01
A1 = np.eye(N) + DT * v * A # Viscosity is multiplied here

# Initial u
u = np.zeros((N,2))
u = reapply_periodic_u(u)
u = reapply_dirchlect_u(u)

steps = 600

plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

skip      = 3                             # decimate so the plot stays readable
mask      = np.arange(nodes_coords.shape[0])[::skip]
xq, yq    = nodes_coords[mask].T
uq, vq    = u[mask].T                     # initial field (all zeros right now)

quiv = ax.quiver(xq, yq, uq, vq, angles='xy',
                 scale_units='xy', scale=10.0, headlength=4, headaxislength=4)
ax.set_aspect('equal')
ax.set_xlim(0,1); ax.set_ylim(0,1)
def unit_vectors(v):
    mag = np.linalg.norm(v, axis=1, keepdims=True)
    mag[mag == 0] = 1.0          # keep zeros at zero, avoid div-by-zero
    return v / mag               # every row now length 1

for step in range(steps):
    rhs = u + DT * b 
    u_star = np.zeros((N,2))
    
    u_star[:,0] = np.linalg.solve(A1, rhs[:,0]) # u_x*
    u_star[:,1] = np.linalg.solve(A1, rhs[:,1]) # u_y*
    

    div = calculate_divergence(nodes_coords, triangles, u_star)
    b_p = div/DT
    b_p -= b_p.mean()
    b_p[ref]   = 0
    p   = np.linalg.solve(Ap, b_p)
    grad_px = np.zeros(N)
    grad_py = np.zeros(N)
    for tri in triangles:
        p_local = p[tri]
        x1, y1 = nodes_coords[tri[0]]
        x2, y2 = nodes_coords[tri[1]]
        x3, y3 = nodes_coords[tri[2]]
        area   = 0.5 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        inv2A  = 1.0 / (2.0*area)
        grads  = np.array([[y2-y3, x3-x2],
                           [y3-y1, x1-x3],
                           [y1-y2, x2-x1]], dtype=np.float64) * inv2A
        gp     = grads.T @ p_local      # (2,)
        lump   = gp * area/3.0
        grad_px[tri] += lump[0]
        grad_py[tri] += lump[1]

    u[:,0] = u_star[:,0] - grad_px*DT
    u[:,1] = u_star[:,1] - grad_py*DT

    # 5. Re-enforce periodic / Dirichlet BCs component-wise
    u = reapply_periodic_u(u)
    u = reapply_dirchlect_u(u)
    if step % 5 == 0:
        div_after = calculate_divergence(nodes_coords, triangles, u)
        print(f"step {step:4d}  max|div u*|={div.max():8.2e}   "
      f"after proj={div_after.max():8.2e}")
        uq, vq = unit_vectors(u)[mask].T               # sample the latest velocity
        quiv.set_UVC(uq, vq)             # fast in-place update
        ax.set_title(f"step {step}")
        plt.pause(0.001)                 # let the GUI breathe

plt.ioff()
plt.show()
