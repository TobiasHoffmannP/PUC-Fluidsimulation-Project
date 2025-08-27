import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List, Set
from jax.experimental import sparse
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

# read the .ele, .node, .poly files

NODE_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.node"
ELE_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.ele"
POLY_FILEPATH = "/home/tobias/FluidMixing/mesh2.1.poly"

OUTER_BOUNDARY_MARKER = 1

#RIGHT_BOUNDARY_MARKER = 1
#LEFT_BOUNDARY_MARKER = 2
#TOP_BOTTOM_BOUNDARY_MARKER = 3
#INNER_BOUNDARY_MARKER = 4
INNER_BOUNDARY_MARKER = 2


OUTER_BOUNDARY_VALUE = 1.0
# TOP_BOTTOM_BOUNDARY_VALUE = 1.0
INNER_BOUNDARY_VALUE = 0.0

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


def buildFemSystem(nodes, triangles, g_source=1.0):
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
        sourceIntergralValue = g_source * area / 3
        
        BVector[p1] += sourceIntergralValue
        BVector[p2] += sourceIntergralValue
        BVector[p3] += sourceIntergralValue

    return AMatrix, -BVector

nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
N = nodes_coords.shape[0] 
triangles = readEle(ELE_FILEPATH)



A, b = buildFemSystem(nodes_coords, triangles, g_source=-4.0)


# add Dirichlet BC
for i in range(N):
    marker = nodes_boundary_markers[i]
    if marker != 0:
        # For boundary nodes, enforce the condition f_i = value
        # Set the row to 0, with 1 on the diagonal
        A[i, :] = 0
        A[i, i] = 1.0
        
        # Set the b vector to the desired boundary value
        if marker == OUTER_BOUNDARY_MARKER:
            b[i] = OUTER_BOUNDARY_VALUE
        elif marker == INNER_BOUNDARY_MARKER:
            b[i] = INNER_BOUNDARY_VALUE


# 4. Solve the Linear System using JAX
Ajp = jnp.array(A)
bjp = jnp.array(b)
f = jnp.linalg.solve(Ajp, bjp)



print(f"System solved: {jnp.allclose(Ajp @ f, bjp, rtol=1e-5, atol=1e-5)}")

# 5. Plot the Result
triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
plt.figure(figsize=(6, 6))
tpc = plt.tripcolor(triang, f, cmap='viridis') # Gouraud shading looks smoother
plt.colorbar(tpc, label='Solution f(x, y)')
plt.gca().set_aspect('equal')
plt.title("FEM Solution")
plt.show()

