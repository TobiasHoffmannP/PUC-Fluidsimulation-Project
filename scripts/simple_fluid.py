import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import map_coordinates

# --- Simulation Parameters ---
GRID_SIZE = 200  # Resolution of the simulation grid
TIME_STEP = 0.1  # Time step for the simulation
VISCOSITY = 0.0001 # How thick the fluid is
DIFFUSION = 0.0001 # How quickly the dye spreads
INFLOW_DURATION = 100 # How long the dye inflow lasts
INFLOW_RADIUS = 100 # Radius of the dye inflow stream

# --- Obstacle Parameters ---
OBSTACLE_CENTER = (GRID_SIZE // 2, GRID_SIZE // 2)
OBSTACLE_BASE_RADIUS = 20
OBSTACLE_SQUIRM_AMPLITUDE = 2
OBSTACLE_SQUIRM_SPEED = 0.1

def create_obstacle_mask(t):
    """
    Creates a boolean mask for the circular obstacle.
    The radius of the circle oscillates over time to create a "squirming" effect.
    """
    y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    radius = OBSTACLE_BASE_RADIUS + OBSTACLE_SQUIRM_AMPLITUDE * np.sin(t * OBSTACLE_SQUIRM_SPEED)
    dist_from_center = np.sqrt((x - OBSTACLE_CENTER[0])**2 + (y - OBSTACLE_CENTER[1])**2)
    return dist_from_center <= radius

def set_boundaries(b, x):
    """
    Sets boundary conditions for a given 2D array `x`.
    The `b` parameter determines the type of boundary condition.

    b = 0: Fixed value boundary (walls).
    b = 1: Horizontal wrap-around (left/right).
    b = 2: Vertical wrap-around (top/bottom).
    """
    
    # Walls
    x[0, :] = -x[1, :] if b == 2 else x[1, :]
    x[-1, :] = -x[-2, :] if b == 2 else x[-2, :]
    x[:, 0] = -x[:, 1] if b == 1 else x[:, 1]
    # Set the right wall to be a zero-gradient outflow for all quantities
    x[:, -1] = x[:, -2]    

    # Corners
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

def linear_solve(b, x, x0, a, c, iters=20):
    """
    Iterative linear solver using Gauss-Seidel method.
    Solves for `x` in the equation `x = (x0 + a * neighbors(x)) / c`.
    This is used for diffusion and pressure calculations.
    """
    c_recip = 1.0 / c
    for _ in range(iters):
        # Calculate sum of neighbors
        neighbors_sum = x[1:-1, :-2] + x[1:-1, 2:] + x[:-2, 1:-1] + x[2:, 1:-1]
        # Update interior points
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * neighbors_sum) * c_recip
        set_boundaries(b, x)

def diffuse(b, x, x0, diff, dt):
    """
    Simulates the diffusion process, where substances spread out over time.
    """
    a = dt * diff * (GRID_SIZE - 2) * (GRID_SIZE - 2)
    linear_solve(b, x, x0, a, 1 + 4 * a)

def project(velocX, velocY, p, div):
    """
    Enforces mass conservation (incompressibility).
    It adjusts the velocity field to ensure that the net flow into any
    grid cell is zero. This is done by solving a Poisson equation for pressure.
    """
    # Calculate divergence of the velocity field
    div[1:-1, 1:-1] = -0.5 * (
        velocX[1:-1, 2:] - velocX[1:-1, :-2] +
        velocY[2:, 1:-1] - velocY[:-2, 1:-1]
    ) / GRID_SIZE
    
    p.fill(0)
    set_boundaries(0, div)
    set_boundaries(0, p)
    linear_solve(0, p, div, 1, 4)

    # Subtract the pressure gradient from the velocity field
    velocX[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * GRID_SIZE
    velocY[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * GRID_SIZE
    set_boundaries(1, velocX)
    set_boundaries(2, velocY)

def advect(b, d, d0, velocX, velocY, dt):
    """
    Moves quantities (like density or velocity) along the velocity field.
    This function uses semi-Lagrangian advection, which traces velocities
    backward in time to find where the quantity came from.
    """
    dtx = dt * (GRID_SIZE - 2)
    dty = dt * (GRID_SIZE - 2)

    # Create coordinate grids
    ix, iy = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))

    # Trace back coordinates
    x = ix - dtx * velocX
    y = iy - dty * velocY
    
    # Clamp coordinates to stay within grid boundaries
    np.clip(x, 0.5, GRID_SIZE - 1.5, out=x)
    np.clip(y, 0.5, GRID_SIZE - 1.5, out=y)

    # Interpolate the values from the source grid (d0) at the new coordinates
    # map_coordinates is a fast way to do this interpolation.
    coords = np.stack([y.ravel(), x.ravel()])
    d[:] = map_coordinates(d0, coords, order=1, mode='nearest').reshape(d.shape)
    
    set_boundaries(b, d)


class FluidSimulation:
    """
    A class to manage the state and steps of the fluid simulation.
    """
    def __init__(self, size, viscosity, diffusion, dt):
        self.size = size
        self.dt = dt
        self.visc = viscosity
        self.diff = diffusion
        
        # Velocity fields
        self.vx = np.zeros((size, size))
        self.vy = np.zeros((size, size))
        self.vx0 = np.zeros((size, size))
        self.vy0 = np.zeros((size, size))
        
        # Density fields
        self.density = np.zeros((size, size))
        self.density0 = np.zeros((size, size))
        
        # Helper fields for projection
        self.p = np.zeros((size, size))
        self.div = np.zeros((size, size))
        
        self.t = 0
        self.frame_count = 0

    def add_source(self):
        """Adds a constant inflow of dye and velocity from the left."""
        # The 'if' statement is removed for continuous flow
        center_y = self.size // 2
        start_y = center_y - INFLOW_RADIUS
        end_y = center_y + INFLOW_RADIUS

        self.vx[start_y:end_y, 1:3] = 5.0 # Add horizontal velocity
        self.density[start_y:end_y, 1:3] = 1.0 # Add dye

    def step(self):
        """Advances the simulation by one time step."""
        # --- Handle Obstacle ---
        obstacle = create_obstacle_mask(self.t)
        self.vx[obstacle] = 0
        self.vy[obstacle] = 0

        # --- Velocity Step ---
        # 1. Diffuse velocity (viscosity)
        self.vx0, self.vy0 = self.vx.copy(), self.vy.copy()
        diffuse(1, self.vx0, self.vx, self.visc, self.dt)
        diffuse(2, self.vy0, self.vy, self.visc, self.dt)
        
        # 2. Project to enforce incompressibility
        project(self.vx0, self.vy0, self.p, self.div)
        
        # 3. Advect velocity field along itself
        self.vx, self.vy = self.vx0.copy(), self.vy0.copy()
        advect(1, self.vx, self.vx0, self.vx0, self.vy0, self.dt)
        advect(2, self.vy, self.vy0, self.vx0, self.vy0, self.dt)
        
        # 4. Project again to clean up any errors
        project(self.vx, self.vy, self.p, self.div)

        # --- Density Step ---
        # 1. Add new dye source
        self.add_source()
        
        # 2. Diffuse density
        self.density0 = self.density.copy()
        diffuse(0, self.density0, self.density, self.diff, self.dt)
        
        # 3. Advect density along the final velocity field
        self.density = self.density0.copy()
        advect(0, self.density, self.density0, self.vx, self.vy, self.dt)
        
        # Apply obstacle mask to density as well
        self.density[obstacle] = 0.1 # Give obstacle a slight color

        self.t += self.dt
        self.frame_count += 1
        return self.density

# --- Main Setup and Animation ---
if __name__ == '__main__':
    sim = FluidSimulation(GRID_SIZE, VISCOSITY, DIFFUSION, TIME_STEP)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Use imshow to display the density. 'magma' is a good colormap for this.
    im = ax.imshow(sim.density, cmap='magma', interpolation='bilinear', vmin=0, vmax=1)
    
    ax.set_title("Fluid Simulation with Pulsating Obstacle", color='white')
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        """Animation update function."""
        density_field = sim.step()
        im.set_array(density_field)
        # Add a subtle flicker to the obstacle color to enhance the squirm effect
        obstacle = create_obstacle_mask(sim.t)
        masked_density = np.ma.masked_where(~obstacle, density_field)
        ax.imshow(masked_density, cmap='bone_r', vmin=0, vmax=1, interpolation='none')
        return [im]

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=400, blit=True, interval=20)
    
    # To save the animation, you might need ffmpeg installed.
    # ani.save('fluid_simulation.mp4', writer='ffmpeg', fps=30)
    
    plt.show()

