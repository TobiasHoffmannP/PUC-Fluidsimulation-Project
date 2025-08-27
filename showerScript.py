# view_triangle.py
import numpy as np
import matplotlib.pyplot as plt

def read_node(fn):
    data = np.loadtxt(fn, skiprows=1)
    return data[:,1:3]

def read_ele(fn):
    data = np.loadtxt(fn, skiprows=1, dtype=int)
    return data[:,1:4]  # triangle vertex indices

nodes = read_node('/home/tobias/FluidMixing/resources/mesh5.1.node')
ele = read_ele('/home/tobias/FluidMixing/resources/mesh5.1.ele')

for tri in ele:
    pts = nodes[tri - 1]  # indices are 1-based
    tri_pts = np.vstack([pts, pts[0]])
    plt.plot(tri_pts[:,0], tri_pts[:,1], 'k-')

plt.gca().set_aspect('equal')
plt.show()

