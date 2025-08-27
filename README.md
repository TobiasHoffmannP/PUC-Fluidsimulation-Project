# PUC Fluid Simulation Project
## Description
    This projects aimts to create a 2D Stokes solver without using existing simulation libraries like FEniCSx. 
    To create the domain, we used the open source software Triangle to create triangle mesh containing a squirmer in the middle. 

## Goals
 - Create a 2D Fluid simulation (Stokes solver) from the ground up
 - Investigate most optimal squirmer boundary conditions to stir surrounding water to enable most consumption of food

## Workflow
 1. Create Triangle mesh 
 2. Solve Poisson's equation (∇²f(x, y) = g(x, y))
 3. Solve Heat equation (∂u/∂t = ∇²u) 
 4. Solve Stokes equation (∂u/∂t = v∇²**u** - ∇p, ∇ • **u** = 0)

## Triangle Mesh
<img src="images/Mesh.png" alt="Alt text" width="300" height="300">

## Poisson's Equation solver
![Description of image](images/Poisson.png)

## Heat Equation solver
![Description of image](images/Heat.png)
### Demo video
[![Watch the video](https://img.youtube.com/vi/9Sp8aIewcIs/0.jpg)](https://www.youtube.com/watch?v=9Sp8aIewcIs)

## Stokes Equation solver
![Description of image](images/Stokes_flow.png)
[![Watch the video](https://img.youtube.com/shorts/8JrjHy0IETQ/0.jpg)](https://www.youtube.com/shorts/8JrjHy0IETQ)


### Visualizaton
[![Watch the video](https://img.youtube.com/watch?v=tvDy6Wnsakc/0.jpg)](https://www.youtube.com/watch?v=9Sp8aIewcIs)
[![Watch the video](https://img.youtube.com/watch?v=kH9ju5QBSCw/0.jpg)](https://www.youtube.com/watch?v=kH9ju5QBSCw)
[![Watch the video](https://img.youtube.com/watch?v=d1kLsCGKszw/0.jpg)](https://www.youtube.com/watch?v=d1kLsCGKszw)




