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
M = 1.0  # Mobility

# Define function for exact solution
def u_ex(mod):
    return lambda x, t: mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1]) * mod.exp(-t)
def q_ex(mod):
    return lambda x, t: M * mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1]) * mod.exp(-t) * (
        (mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1]))**2 * mod.exp(-2 * t) + 2 * mod.pi**2 * epsilon - 1
    )

# Define function for source term
def f_c_numpy(x, t):
    return - np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-t) - M * ((6 * (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))**2 * np.exp(-3 * t) - 2) * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-t) + 2 * np.pi**4 * epsilon * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

u_numpy = u_ex(np)
u_ufl = u_ex(ufl)
q_numpy = q_ex(np)
q_ufl = q_ex(ufl)


# Solve Cahn-Hilliard equation
def solve_ch(N=96, num_steps=2e5):
    t = 0.0
    t_n = 0.0 # Previous time
    T = 1.0e-4
    dt = T / num_steps

    # Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    # Create FunctionSpace
    P1 = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
    ME = fem.functionspace(domain, mixed_element([P1, P1]))

    # Define initial condition
    u_n = fem.Function(ME)
    u_n.sub(0).interpolate(lambda x: u_numpy(x, 0.0))
    u_n.x.scatter_forward()
    q_n = fem.Function(ME)
    q_n.sub(1).interpolate(lambda x: q_numpy(x, 0.0))
    q_n.x.scatter_forward()

    # Define solution variable
    u = fem.Function(ME)

    # Define variational problem
    q, v = ufl.TestFunctions(ME)
    # Split mixed functions
    c, mu = ufl.split(u)
    c_n, mu_n = ufl.split(u_n)

    # Compute df/dc
    c = ufl.variable(c)
    f = 100 * c**2 * (1 - c) ** 2
    dfdc = ufl.diff(f, c)

    # Define source function
    x = ufl.SpatialCoordinate(domain)
    # Define source function
    f_c = fem.Function(ME.sub(0).collapse()[0])
    
    log.set_log_level(log.LogLevel.INFO)

    # Time stepping
    for _ in range(int(num_steps)):
        t += dt

        # Update source term
        f_c.interpolate(lambda x: (f_c_numpy(x, t) + f_c_numpy(x, t_n)) / 2)

        # Update residuals
        F0 = ufl.inner(c, q) * ufl.dx - ufl.inner(c_n, q) * ufl.dx + (dt/2) * ufl.inner(ufl.grad(mu + mu_n), ufl.grad(q)) * ufl.dx - ufl.inner(f_c, v) * ufl.dx
        F1 = ufl.inner(mu, v) * ufl.dx - ufl.inner(dfdc, v) * ufl.dx - epsilon * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
        F = F0 + F1
        problem = NonlinearProblem(F, u)
        # Create Newton Solver
        solver = NewtonSolver(domain.comm, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1.0e-6
        solver.atol = 1.0e-8  # Add absolute tolerance
        solver.max_it = 50    # Add maximum iterations
        ksp = solver.krylov_solver
        opts = PETSc.Options()  # type: ignore
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        sys = PETSc.Sys()
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
        ksp.setFromOptions()
        n, converged = solver.solve(u)
        if not converged:
            print(f"Newton solver failed to converge at t = {t}")
        # Update previous solution
        u_n.x.array[:] = u.x.array
        t_n = t

    # Interpolate exact solution for error computation
    u_exact = fem.Function(ME)
    u_exact.sub(0).interpolate(lambda x: u_numpy(x, T))
    u_exact.sub(1).interpolate(lambda x: q_numpy(x, T))
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
N_values = [2**i for i in range(3, 8)]
Errors_c = np.zeros(len(N_values), dtype=default_scalar_type)
Errors_mu = np.zeros(len(N_values), dtype=default_scalar_type)
h_values = np.zeros(len(N_values), dtype=np.float64)

for i, N in enumerate(N_values):
    u_h, u_ex = solve_ch(N=N)
    Errors_c[i], Errors_mu[i] = compute_errors(u_h, u_ex)
    h_values[i] = 1 / N_values[i]

# Generate convergence plot for mesh size
if MPI.COMM_WORLD.rank == 0:
    plt.figure()
    plt.loglog(h_values, Errors_c, marker='o', linestyle='-', label='L2 Error (c)')
    plt.loglog(h_values, Errors_mu, marker='s', linestyle='-', label='L2 Error (μ)')
    # Add reference lines
    for slope in [1, 2]:
        ref_line = Errors_c[0] * (h_values / h_values[0])**slope
        plt.loglog(h_values, ref_line, '--', label=f'O(h^{slope})')
    plt.xlabel('Mesh size (h)')
    plt.ylabel('L2 Error')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Cahn-Hilliard Error Convergence')
    plt.savefig('ch_err.png')

# Error convergence for time refinement
num_steps_values = [2**i * 1e4 for i in range(3, 8)]
t_Errors_c = np.zeros(len(num_steps_values), dtype=default_scalar_type)
t_Errors_mu = np.zeros(len(num_steps_values), dtype=default_scalar_type)
dt_values = np.zeros(len(num_steps_values), dtype=np.float64)

for i, num_steps in enumerate(num_steps_values):
    u_h, u_ex = solve_ch(num_steps=num_steps)
    t_Errors_c[i], t_Errors_mu[i] = compute_errors(u_h, u_ex)
    dt_values[i] = 1 / num_steps_values[i]

# Generate convergence plot for timestep size
if MPI.COMM_WORLD.rank == 0:
    plt.figure()
    plt.loglog(dt_values, t_Errors_c, marker='o', linestyle='-', label='L2 Error (c)')
    plt.loglog(dt_values, t_Errors_mu, marker='s', linestyle='-', label='L2 Error (μ)')
    # Add reference lines
    for slope in [1, 2]:
        ref_line = t_Errors_c[0] * (dt_values / dt_values[0])**slope
        plt.loglog(dt_values, ref_line, '--', label=f'O(dt^{slope})')
    plt.xlabel('Time step size (dt)')
    plt.ylabel('L2 Error')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Cahn-Hilliard Error Convergence (Time)')
    plt.savefig('ch_err_t.png')
