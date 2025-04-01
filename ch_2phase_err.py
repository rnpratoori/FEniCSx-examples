# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, io, log, default_scalar_type, default_real_type
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from pathlib import Path
import matplotlib.pyplot as plt

# Define parameters
epsilon = 0.01  # Interfacial width parameter
kappa = epsilon**(2)  # Gradient coefficient
M = 1.0  # Mobility

# Define function for exact solution
def c_ex(mod):
    return lambda x, t: mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1]) * mod.exp(-t)
# def grad2_c_ex(mod):
#     return lambda x, t: -2 * mod.pi**2 * mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1]) * mod.exp(-t)
def mu_ex(mod):
    return lambda x, t: 2 * mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1]) * mod.exp(-t) * (
        kappa * mod.pi**2 + (1 - mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1]) * mod.exp(-t)) * 
        (1 - 2 * mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1]) * mod.exp(-t)))
# def grad_c_ex2(mod):
#     return lambda x, t: mod.pi**2 * mod.exp(-2 * t) * ((mod.cos(mod.pi * x[0]) * mod.sin(mod.pi * x[1]))**2 + (mod.sin(mod.pi * x[0]) * mod.cos(mod.pi * x[1]))**2)

# Define function for source term
def f_c_numpy(x, t):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-t) - (2 * kappa * np.pi**2 + 2 * (1 - 6 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-t) + 
        6 * (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-t))**2)) * (-2 * np.pi**2 * np.sin(np.pi * x[0]) * 
        np.sin(np.pi * x[1]) * np.exp(-t)) + 2 * (6 * (np.pi**2 * np.exp(-2 * t) * ((np.cos(np.pi * x[0]) * 
        np.sin(np.pi * x[1]))**2 + (np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))**2)) * (2 * np.sin(np.pi * x[0]) * 
        np.sin(np.pi * x[1]) * np.exp(-t)) - 1)
    # return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-t) + 4 * np.pi**2 * np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * (1 + 6 * np.exp(-2 * t) * (np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]))**2 - 6 * np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + kappa * np.pi**2)
        # (np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) + np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])) - 1
# Choose a sample time at which to evaluate the forcing term
t0 = 0.1  # adjust as needed

# Create a grid in the domain [0,1]x[0,1]
nx, ny = 50, 50
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluate f_c_numpy on the grid
Z = np.zeros_like(X)
for i in range(nx):
    for j in range(ny):
        # x is a 2-element list; note that our function expects a list/array of coordinates
        Z[j, i] = f_c_numpy([X[j, i], Y[j, i]], t0)

# Create a contour plot
plt.figure(figsize=(6,5))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title(f"Forcing Term $f_c(x,t)$ at $t={t0}$")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

c_numpy = c_ex(np)
c_ufl = c_ex(ufl)
mu_numpy = mu_ex(np)
mu_ufl = mu_ex(ufl)


# Solve Cahn-Hilliard equation
def solve_ch(N=96, num_steps=20):
    if MPI.COMM_WORLD.rank==0:
        print(f"*****Running simulation for N={N}, num_steps={num_steps}******")
    # Define time parameters
    t = 0.0
    t_n = 0.0 # Previous time
    T = 1.0e-7
    dt = T / num_steps

    # Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    # Create FunctionSpace
    P1 = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
    ME = fem.functionspace(domain, mixed_element([P1, P1]))

    # Define initial condition
    u_n = fem.Function(ME)
    u_n.sub(0).interpolate(lambda x: c_numpy(x, 0.0))
    u_n.sub(1).interpolate(lambda x: mu_numpy(x, 0.0))
    u_n.x.scatter_forward()

    # Define solution variable
    u = fem.Function(ME)

    # Define variational problem
    q, v = ufl.TestFunctions(ME)
    # Split mixed functions
    c, mu = ufl.split(u)
    c_n, mu_n = ufl.split(u_n)

    # Compute df/dc
    c = ufl.variable(c)
    f = c**2 * (1 - c) ** 2
    dfdc = ufl.diff(f, c)

    # Define source function
    f_c = fem.Function(ME.sub(0).collapse()[0])
    
    # if MPI.COMM_WORLD.rank == 0:
    #     log.set_log_level(log.LogLevel.INFO)

    # PETSc options
    opts = PETSc.Options()  # type: ignore

    # Time stepping
    for _ in range(int(num_steps)):
        t += dt

        # Update source term
        f_c.interpolate(lambda x: (f_c_numpy(x, t) + f_c_numpy(x, t_n)) / 2)

        # Update residuals
        F0 = ufl.inner(c - c_n, q) * ufl.dx + (dt/2) * ufl.inner(ufl.grad(mu + mu_n), ufl.grad(q)) * ufl.dx - dt * ufl.inner(f_c, q) * ufl.dx
        F1 = ufl.inner(mu, v) * ufl.dx - ufl.inner(dfdc, v) * ufl.dx + kappa * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
        F = F0 + F1

        # Create Newton Solver
        problem = NonlinearProblem(F, u)
        solver = NewtonSolver(domain.comm, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1.0e-12
        solver.atol = 1.0e-12 # Add absolute tolerance
        solver.max_it = 100    # Add maximum iterations
        ksp = solver.krylov_solver
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
        ksp.setFromOptions()
        r = solver.solve(u)
        if MPI.COMM_WORLD.rank==0:
            print(f"Step {int(t / dt)}: num iterations: {r[0]}")
        # Update previous solution
        u_n.x.array[:] = u.x.array
        t_n = t

    # Interpolate exact solution for error computation
    u_exact = fem.Function(ME)
    u_exact.sub(0).interpolate(lambda x: c_numpy(x, T))
    u_exact.sub(1).interpolate(lambda x: mu_numpy(x, T))
    u_exact.x.scatter_forward()

    return u, u_exact

def compute_errors(u_h, u_ex):
    # Extract components
    c_h, mu_h = u_h.split()
    c_ex, mu_ex = u_ex.split()
    
    # Compute L2 errors for each component
    error_c = fem.form((c_h - c_ex)**2 * ufl.dx)
    error_mu = fem.form((mu_h - mu_ex)**2 * ufl.dx)
    
    comm = u_h.function_space.mesh.comm
    l2_error_c = np.sqrt(comm.allreduce(fem.assemble_scalar(error_c), MPI.SUM))
    l2_error_mu = np.sqrt(comm.allreduce(fem.assemble_scalar(error_mu), MPI.SUM))
    
    return l2_error_c, l2_error_mu

# Error convergence for mesh refinement
N_values = [2**i for i in range(2, 7)]
Errors_c = np.zeros(len(N_values), dtype=default_scalar_type)
Errors_mu = np.zeros(len(N_values), dtype=default_scalar_type)
h_values = np.zeros(len(N_values), dtype=np.float64)

for i, N in enumerate(N_values):
    u_h, c_ex = solve_ch(N=N)
    comm = u_h.function_space.mesh.comm
    Errors_c[i], Errors_mu[i] = compute_errors(u_h, c_ex)
    h_values[i] = 1 / N_values[i]
    if comm.rank == 0:
        print(f"h: {h_values[i]:.2e} Error: {Errors_c[i]:.2e}")
        print(f"h: {h_values[i]:.2e} Error: {Errors_mu[i]:.2e}")
rates_c = np.log(Errors_c[1:] / Errors_c[:-1]) / np.log(h_values[1:] / h_values[:-1])
if comm.rank == 0:
    print(f"Polynomial degree 1, Rates_c {rates_c}")
rates_mu = np.log(Errors_mu[1:] / Errors_mu[:-1]) / np.log(h_values[1:] / h_values[:-1])
if comm.rank == 0:
    print(f"Polynomial degree 1, Rates_mu {rates_mu}")

# Generate convergence plot for mesh size
if MPI.COMM_WORLD.rank == 0:
    plt.figure()
    plt.loglog(h_values, Errors_c, marker='o', linestyle='-', label='L2 Error (c)')
    plt.loglog(h_values, Errors_mu, marker='s', linestyle='-', label='L2 Error (μ)')
    # Add reference lines
    for slope in [2]:
        ref_line_c = Errors_c[0] * (h_values / h_values[0])**slope
        plt.loglog(h_values, ref_line_c, '--', label=f'O(h^{slope})_c')
        ref_line_mu = Errors_mu[0] * (h_values / h_values[0])**slope
        plt.loglog(h_values, ref_line_mu, '--', label=f'O(h^{slope})_mu')
    plt.xlabel('Mesh size (h)')
    plt.ylabel('L2 Error')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Cahn-Hilliard Error Convergence')
    plt.savefig('ch_err.png')

# # Error convergence for time refinement
# num_steps_values = [2**i * 1e4 for i in range(3, 8)]
# t_Errors_c = np.zeros(len(num_steps_values), dtype=default_scalar_type)
# t_Errors_mu = np.zeros(len(num_steps_values), dtype=default_scalar_type)
# dt_values = np.zeros(len(num_steps_values), dtype=np.float64)

# for i, num_steps in enumerate(num_steps_values):
#     u_h, u_ex = solve_ch(num_steps=num_steps)
#     t_Errors_c[i], t_Errors_mu[i] = compute_errors(u_h, u_ex)
#     dt_values[i] = 1 / num_steps_values[i]

# # Generate convergence plot for timestep size
# if MPI.COMM_WORLD.rank == 0:
#     plt.figure()
#     plt.loglog(dt_values, t_Errors_c, marker='o', linestyle='-', label='L2 Error (c)')
#     plt.loglog(dt_values, t_Errors_mu, marker='s', linestyle='-', label='L2 Error (μ)')
#     # Add reference lines
#     for slope in [1, 2]:
#         ref_line = t_Errors_c[0] * (dt_values / dt_values[0])**slope
#         plt.loglog(dt_values, ref_line, '--', label=f'O(dt^{slope})')
#     plt.xlabel('Time step size (dt)')
#     plt.ylabel('L2 Error')
#     plt.legend()
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.title('Cahn-Hilliard Error Convergence (Time)')
#     plt.savefig('ch_err_t.png')
