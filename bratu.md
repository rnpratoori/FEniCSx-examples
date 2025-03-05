# Bratu Equation

This is the implementation for Bratu Equation in 2D using FEniCSx.
The boundary-value problem is described by:
$$
\begin{aligned}
-\Delta u(x) = \lambda e^{u(x)} &\quad x\in \Omega \\
u(x) = 0 &\quad x\in \partial \Omega
\end{aligned}
$$

## Variational formulation

To calculate the variational formulation, we multiply the equation by a test function $v$ and integrate over $\Omega$:
$$\int_\Omega (-\Delta u)v dx = \int_\Omega \lambda e^{u(x)} v dx$$

Using integration by parts to the LHS:
$$\int_\Omega \nabla u \cdot \nabla v dx - \int_{\partial\Omega} \frac{\partial u}{\partial n} v ds
    = \int_\Omega \lambda e^{u(x)} v dx$$
Since the integral term on the boundary is 0, the equation becomes
$$\int_\Omega \nabla u \cdot \nabla v dx = \int_\Omega \lambda e^{u(x)} v dx$$
leading to
$$
F(u,v) := \int_\Omega \nabla u \cdot \nabla v dx - \int_\Omega \lambda e^{u(x)} v dx
$$

## Problem definition

- Domain: $\Omega = [0,1] \times [0,1]$
- Dirichlet BCs: $u = 0$ on $\partial\Omega$
- Forcing fn: $\lambda = 1$