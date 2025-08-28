# PUC Fluid Simulation Project  

## Overview  
This project implements a **2D Stokes solver** from the ground up, without relying on pre-existing simulation libraries such as FEniCSx. The domain is discretized using triangular meshes generated with *Triangle*, with a circular squirmer placed at the center.  

The project serves two purposes:  
1. **Developing a stable fluid solver** for Stokes flow using finite element methods (FEM).  
2. **Investigating squirmer boundary conditions** to determine how swimming strategies affect mixing and food capture efficiency.  

## Workflow  
The development followed a bottom-up approach, starting with simpler PDEs and building towards the Stokes solver:  
1. **Triangle mesh generation** of a 2D domain with a central squirmer.  
2. **Poisson solver** via FEM for pressure projection.  
3. **Heat equation solver** for time-dependent diffusion.  
4. **Stokes solver** using operator splitting (velocity update → pressure correction → projection).  

## Results  
- A functioning **Poisson and heat equation solver** was implemented successfully.  
- A **2D Stokes solver** was developed and stabilized via mass-lumped FEM, though non-zero divergence and checkerboarding effects limited accuracy.  
- Despite instabilities, the solver enabled **visualization of squirmer-driven flows**, tracer advection, and mixing dynamics.  
- Simulations with different squirmer models (neutral, pusher, puller) showed that **pusher/puller strategies enhance mixing and food capture**, whereas the neutral swimmer is less effective.  

---

## Triangle Mesh  
<img src="images/Mesh.png" alt="Triangle Mesh" width="300" height="300">

## Poisson's Equation Solver  
![Poisson Solver](images/Poisson.png)

## Heat Equation Solver  
![Heat Solver](images/Heat.png)  
### Demo Video  
[![Watch the video](https://img.youtube.com/vi/9Sp8aIewcIs/0.jpg)](https://www.youtube.com/watch?v=9Sp8aIewcIs)

## Stokes Equation Solver  
![Stokes Solver](images/Stokes_flow.png)  
[![Watch the video](https://img.youtube.com/vi/8JrjHy0IETQ/0.jpg)](https://www.youtube.com/shorts/8JrjHy0IETQ)

## Visualization  
Color mixing and tracer-based food capture were simulated under different squirmer boundary conditions:  

- **Neutral swimmer (B1 = -2, B2 = 0):** Inefficient stirring, ~50% food consumed.  
- **Pusher swimmer (B1 = -2, B2 = -5):** Stronger stirring, ~97% food consumed.  
- **Puller swimmer (B1 = -2, B2 = 5):** Slightly more efficient than pusher, ~98% food consumed.  

### Videos  
[![Neutral Swimmer](https://img.youtube.com/vi/tvDy6Wnsakc/0.jpg)](https://www.youtube.com/watch?v=tvDy6Wnsakc)  
[![Pusher Swimmer](https://img.youtube.com/vi/kH9ju5QBSCw/0.jpg)](https://www.youtube.com/watch?v=kH9ju5QBSCw)  
[![Puller Swimmer](https://img.youtube.com/vi/d1kLsCGKszw/0.jpg)](https://www.youtube.com/watch?v=d1kLsCGKszw)  

---

