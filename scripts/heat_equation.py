import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List, Set
from jax.experimental import sparse
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
jax.config.update("jax_enable_x64", True)

# read the .ele, .node, .poly files

NODE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/resources/mesh2.1.poly"

OUTER_BOUNDARY_MARKER = 1

#RIGHT_BOUNDARY_MARKER = 1
#LEFT_BOUNDARY_MARKER = 2
#TOP_BOTTOM_BOUNDARY_MARKER = 3
#INNER_BOUNDARY_MARKER = 4
INNER_BOUNDARY_MARKER = 2


OUTER_BOUNDARY_VALUE = 1.0
#TOP_BOTTOM_BOUNDARY_VALUE = 1.0
INNER_BOUNDARY_VALUE = 0.0


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


def readPoly(path: str):
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
        print(sourceIntergralValue)

        BVector[p1] += sourceIntergralValue
        BVector[p2] += sourceIntergralValue
        BVector[p3] += sourceIntergralValue

    return AMatrix, BVector


def find_boundary_pairs(nodes_coords, L=1.0, tol=1e-6):

    # get the indices of all nodes on the left and right boundaries
    left_indices = np.where(np.abs(nodes_coords[:, 0]) < tol)[0]
    right_indices = np.where(np.abs(nodes_coords[:, 0] - L) < tol)[0]

    # Return early if one of the boundaries has no nodes
    if len(left_indices) == 0 or len(right_indices) == 0:
        print("Warning: One or both boundaries have no nodes.")
        return []
 
    left_coords = nodes_coords[left_indices]
    right_coords = nodes_coords[right_indices]


    right_y_coords_for_tree = right_coords[:, 1].reshape(-1, 1)
    kdtree = KDTree(right_y_coords_for_tree)

    pairs = []
    for i, left_idx in enumerate(left_indices):
        left_y = left_coords[i, 1]
        
        dist, right_array_idx = kdtree.query([left_y])
        
        right_node_idx = right_indices[right_array_idx]
        
        pairs.append((left_idx, right_node_idx))

    return pairs


def apply_periodic_bc(A, b, pairs):
    #   left set    right set
    for master_idx, slave_idx in pairs:
        # add the slave's row contributions to the master's row
        A[master_idx, :] += A[slave_idx, :]
        b[master_idx] += b[slave_idx]

        # clear the slave's row to create the constraint equation
        A[slave_idx, :] = 0.0

        # enforce the constraint: 1*x_slave - 1*x_master = 0
        A[slave_idx, slave_idx] = 1.0
        A[slave_idx, master_idx] = -1.0
        b[slave_idx] = 0.0    

# TUBE_WALL_MARKER = 3 # A NEW marker for top/bottom walls

# WALL_VALUE = 0.0

def apply_dirchlect_u(u):
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

def apply_periodic_u(u):
    for master, slave in pairs:
        u[slave] = u[master]
    return u

H = 1.0
tol = 1e-6


nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
N = nodes_coords.shape[0]
triangles = readEle(ELE_FILEPATH)


def g_source_fun(x, y):
    return 50 * np.sin(3 * y)

A, b = buildFemSystem(nodes_coords, triangles, g_source=-5)

# 2. Apply PERIODIC boundary conditions to the spatial matrix.
pairs = find_boundary_pairs(nodes_coords, L=1.0)
filtered_pairs = []
for master_idx, slave_idx in pairs:
    master_y = nodes_coords[master_idx, 1]
    if not (np.abs(master_y - 0.0) < tol or np.abs(master_y - H) < tol):
        filtered_pairs.append((master_idx, slave_idx))

apply_periodic_bc(A, b, filtered_pairs) 


# apply the (Dirichlet) boundary conditions
for i in range(N):
    marker = nodes_boundary_markers[i]
    y_coord = nodes_coords[i, 1]
    is_wall = np.abs(y_coord - 0.0) < tol or np.abs(y_coord - H) < tol
    is_inner = (marker == INNER_BOUNDARY_MARKER)
    if is_wall or is_inner:
        A[i, :] = 0.0
        A[i, i] = 1.0
        if is_inner:
            b[i] = INNER_BOUNDARY_VALUE
        elif is_wall:
            b[i] = OUTER_BOUNDARY_VALUE



DT = 0.01
NUM_STEPS = 1000

A_new = np.eye(len(A)) + DT * A

u = np.zeros(N, dtype=np.float64)
u = apply_periodic_u(u)
u = apply_dirchlect_u(u)


plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)

tpc = ax.tripcolor(triang, u, shading="gouraud", cmap="viridis")
fig.colorbar(tpc, label="Solution f(x, y)")
ax.set_aspect("equal")


b = np.zeros(N, dtype=np.float64)


for step in range(NUM_STEPS):
    rhs = u - DT * b * 0 

    u = np.linalg.solve(A_new, rhs)
    u = apply_periodic_u(u)
    u = apply_dirchlect_u(u) 

    if step % 5 == 0: 
        print(f"Completed step {step}/{NUM_STEPS}")
        tpc.set_array(u)
        tpc.autoscale()
        ax.set_title(f"Time Evolution: Step {step}/{NUM_STEPS}")
        fig.canvas.draw_idle()
        plt.pause(0.001)

plt.ioff()
tpc.set_array(u)
ax.set_title(f"Final State at {NUM_STEPS}")
plt.show()


