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

FIRST_EQ = True


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


def getNeighbors(N: int, triangles: np.ndarray):
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(N)}
    for tri in triangles:
        v0, v1, v2 = tri
        neighbors[v0].add(v1)
        neighbors[v0].add(v2)
        neighbors[v1].add(v0)
        neighbors[v1].add(v2)
        neighbors[v2].add(v0)
        neighbors[v2].add(v1)

    return {node_id: sorted(list(neigh)) for node_id, neigh in neighbors.items()}



if (FIRST_EQ):
    nodes_coords, nodes_boundary_markers = readNode(NODE_FILEPATH)
    N = nodes_coords.shape[0] 
    triangles = readEle(ELE_FILEPATH)
    
    neighbors = getNeighbors(N, triangles)

    # print(neighbors)


    A = np.zeros((N,N), dtype=float)

    def buildAverageMatrix():
        # Af = b
        for i in range(N):
            A[i,i] = 1.0
            if nodes_boundary_markers[i] == 0:
                neigh = neighbors[i]
                amount_of_neighbors = len(neigh)
                for j in neigh:
                    A[i, j] = -1/amount_of_neighbors

    b = np.zeros((N), dtype=float)


    for i in range(N):
        if nodes_boundary_markers[i] == 0:
            b[i] = 0
        elif nodes_boundary_markers[i] == 1:
            b[i] = INNER_BOUNDARY_VALUE
        elif nodes_boundary_markers[i] == 2:
            b[i] = OUTER_BOUNDARY_VALUE

    buildAverageMatrix()

    # Convert to jnp arrays
    Ajp = jnp.array(A)
    bjp = jnp.array(b)

    f = jnp.linalg.solve(Ajp, bjp)

    # print(f)

    print(jnp.allclose(Ajp @ f, bjp, rtol=1e-12, atol=1e-12))

    triang = mtri.Triangulation(nodes_coords[:, 0], nodes_coords[:, 1], triangles)
    plt.figure(figsize=(5, 5))
    tpc = plt.tripcolor(triang, f, shading='flat', cmap='viridis')
    plt.colorbar(tpc, label='f value')
    plt.gca().set_aspect('equal')
    plt.title('Solution field')


    plt.show()


