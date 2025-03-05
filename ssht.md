# Steady State Heat Transfer

This is the implementation for steady state heat transfer in 2D using FEniCSx.
The boundary-value problem is described by:
$$
\begin{aligned}
-\Delta u(x) = f(x) &\quad x\in \Omega \\
u(x) = u_D(x) &\quad x\in \partial \Omega
\end{aligned}
$$

## Variational formulation

To calculate the variational formulation, we multiply the equation by a test function $v$ and integrate over $\Omega$:
$$\int_\Omega (-\Delta u)v dx = \int_\Omega f v dx$$

Using integration by parts to the LHS:
$$\int_\Omega \nabla u \cdot \nabla v dx - \int_{\partial\Omega} \frac{\partial u}{\partial n} v ds
    = \int_\Omega f v dx$$
Since the integral term on the boundary is 0, the equation becomes
$$\int_\Omega \nabla u \cdot \nabla v dx = \int_\Omega f v dx$$
leading to
$$
\begin{aligned}
a(u,v) &:= \int_\Omega \nabla u \cdot \nabla v dx\\
L(v) &:= \int_\Omega f v dx
\end{aligned}
$$

## Problem definition

- Domain: $\Omega = [0,2] \times [0,1]$
- Dirichlet BCs: $u = 0$ on $\{(0,y) \cup (2,y)\} \in \partial\Omega$
- Forcing fn: $f = 10 \exp(- ((x - 0.5)^2 + (y - 0.5)^2) / 0.02)$