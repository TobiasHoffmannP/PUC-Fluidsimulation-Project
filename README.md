# PUC Fluid Simulation Project
## Description
    This projects aimts to create a 2D Stokes solver without using existing simulation libraries like FEniCSx. 
    To create the domain, we used the open source software Triangle to create triangle mesh containing a squirmer in the middle. 

## Goals
 - Create a 2D Fluid simulation (Stokes solver) from the ground up
 - Investigate most optimal squirmer boundary conditions to stir surrounding water to enable most consumption of food

## Workflow
 1. Create Triangle mesh 
 2. Solve Poisson equation (∇²f(x, y) = g(x, y))
 3. Solve Heat equation (∂u/∂t = ∇²u) 
 4. Solve Stokes equation (∂u/∂t = v∇² _u_ - ∇p, ∇ • _u_ = 0)

 ![Mesh](https://imgur.com/a/izxPIQ2 "Triangle Mesh")
