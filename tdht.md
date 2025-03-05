# Time Dependent Heat Transfer

This is the implementation for time dependent heat transfer in 2D using FEniCSx.
The boundary-value problem is described by:
$$
\begin{aligned}
\frac{\partial u}{\partial t} &= \Delta u(x) + f(x) &\quad x\in \Omega \times (0,T] \\
u(x) &= u_D(x) &\quad x\in \partial \Omega \times (0,T] \\
u(x) &= u_0 &\quad at \quad t = 0
\end{aligned}
$$

## Variational formulation

### Crank-Nicholson time stepper

The sampling of a PDE at time $t_{n+1}$ is given by:
$$\left(\frac{\partial u}{\partial t}\right)^{n+1} = \Delta u^{n+1} + f^{n+1}$$

The time-derivative can be approximated using Crank-Nicholsam time stepper as:
$$
\begin{aligned}
\frac{u^{n+1} - u^n}{\Delta t} = \frac{1}{2}\left(\Delta u^{n+1} + \Delta u^n + f^{n+1} + f^n\right)\\
u^{n+1} - \frac{\Delta t}{2}\Delta u^{n+1} = u^n + \frac{\Delta t}{2}\left(\Delta u^{n} + f^n + f^{n+1}\right)
\end{aligned}
$$

### Weak formulation

To calculate the variational formulation, we multiply the equation by a test function $v$ and integrate over $\Omega$:
$$\int_\Omega \left( u^{n+1} v - \frac{\Delta t}{2} \Delta u^{n+1} v \right) dx = \int_\Omega u^n v dx + \frac{\Delta t}{2} \int_\Omega \left( \Delta u^n v + f^{n+1} v + f^n v \right) dx$$

Using integration by parts to the reduce the higher order terms:
$$\int_\Omega u^{n+1} v dx + \frac{\Delta t}{2}\int_\Omega \nabla u^{n+1} \cdot \nabla v dx - \frac{\Delta t}{2}\int_{\partial\Omega} \frac{\partial u^{n+1}}{\partial n} v ds
    = \int_\Omega u^n v dx - \frac{\Delta t}{2}\int_\Omega \nabla u^n \cdot \nabla v dx + \frac{\Delta t}{2}\int_{\partial\Omega} \frac{\partial u^n}{\partial n} v ds + \frac{\Delta t}{2} \int_\Omega \left( f^{n+1} v + f^n v \right) dx$$
Since the integral term on the boundary is $0$, the equation becomes
$$\int_\Omega u^{n+1} v dx + \frac{\Delta t}{2}\int_\Omega \nabla u^{n+1} \cdot \nabla v dx = \int_\Omega u^n v dx - \frac{\Delta t}{2}\int_\Omega \nabla u^n \cdot \nabla v dx + \frac{\Delta t}{2} \int_\Omega \left( f^{n+1} v + f^n v \right) dx$$
leading to
$$
\begin{aligned}
a(u^{n+1},v) &:= \int_\Omega \left(u^{n+1} v + \frac{\Delta t}{2}\nabla u^{n+1} \cdot \nabla v \right)dx\\
L_{n+1}(v) &:= \int_\Omega \left(u^n v + \frac{\Delta t}{2}\left( f^{n+1} v + f^n v - \nabla u^n \cdot \nabla v \right) \right)dx
\end{aligned}
$$

## Problem definition

- Domain: $\Omega = [0,1] \times [0,1]$
- Dirichlet BCs: $u = 0$ on $\{(0,y) \cup (2,y)\} \in \partial\Omega$
- Forcing fn: $f = 0$