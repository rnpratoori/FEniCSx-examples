# Cahn-Hilliard Equation

This is the implementation for Cahn-Hilliard Equation for two phase separation in 2D using FEniCSx.
The boundary-value problem is described by:
$$\frac{\partial c}{\partial t} - \nabla \cdot M \left( \nabla \left( \frac{\partial f}{\partial c} - \lambda \Delta c \right) \right) = 0 \quad \in \Omega$$

We use split formulation to rephrase the fourth-order equation into two coupled second-order equations:
$$
\begin{aligned}
\frac{\partial c}{\partial t} - \nabla \cdot M \nabla \mu &= 0 \quad \in \Omega\\
\mu - \frac{\partial f}{\partial c} + \lambda \Delta c &= 0 \quad \in \Omega
\end{aligned}
$$

## Variational formulation

### Weak formulation

The variational form of the equations are:
$$
\begin{aligned}
\int_\Omega \frac{\partial c}{\partial t} q dx + \int_\Omega M \nabla \mu \cdot \nabla q dx = 0 \\
\int_\Omega \mu v dx - \int_\Omega \frac{\partial f}{\partial c} v dx - \int_\Omega \lambda \nabla c \cdot \nabla v dx = 0
\end{aligned}
$$

### Crank-Nicholson time stepper

The sampling of the first PDE at time $t_{n+1}$ is given by:
$$
\begin{aligned}
\int_\Omega \frac{c_{n+1} - c_n}{dt} q dx + \frac{1}{2} \int_\Omega M \nabla \left( \mu_{n+1} + \mu_n \right) \cdot \nabla q dx = 0 \\
\int_\Omega \mu_{n+1} v dx - \int_\Omega \frac{d f_{n+1}}{dc} v dx - \int_\Omega \lambda \nabla c_{n+1} \cdot \nabla v dx = 0
\end{aligned}
$$

## Problem definition

- Domain: $\Omega = [0,1] \times [0,1]$
- Local energy: $f = 100 c^2 (1 - c)^2$
- Gradient energy coefficent: $\lambda = 1 \times 10^{-2}$
- Mobility coefficient: $M = 1$
- Dirichlet BCs: $u = 0$ on $\{(0,y) \cup (2,y)\} \in \partial\Omega$