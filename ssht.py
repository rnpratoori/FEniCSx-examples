import jax
import jax.numpy as np
import os
import matplotlib.pyplot as plt
import logging

# Set logging level to WARNING to suppress both DEBUG and INFO messages
logging.getLogger('jax_fem').setLevel(logging.WARNING)

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

class ssht(Problem):
    def get_tensor_map(self):
        return lambda x: x
    
    def get_mass_map(self):
        def mass_map(u, x):
            val = -np.array([(2 * np.pi**2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])])
            return val
        return mass_map

# Define exact solution for error computation
def exact_solution(point):
    return np.sin(np.pi * point[0]) * np.sin(np.pi * point[1])

def compute_l2_error(mesh, numerical_sol):
    errors = np.array([(numerical_sol[i] - exact_solution(point))**2 for i, point in enumerate(mesh.points)])
    return np.sqrt(np.sum(errors) / len(mesh.points))

# Mesh refinement study
elements = [16, 32, 64, 128, 256, 512]
dh = np.array([1 / i for i in elements])  # Convert to numpy array
errors = []

for Nx in elements:
    Ny = Nx
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 1.
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    
    # Define boundaries
    def left(point): return np.isclose(point[0], 0., atol=1e-5)
    def right(point): return np.isclose(point[0], Lx, atol=1e-5)
    def bottom(point): return np.isclose(point[1], 0., atol=1e-5)
    def top(point): return np.isclose(point[1], Ly, atol=1e-5)
    
    def dirichlet_val(point): return exact_solution(point)
    
    dirichlet_bc_info = [[left, right, bottom, top], [0, 0, 0, 0], [dirichlet_val, dirichlet_val, dirichlet_val, dirichlet_val]]
    problem = ssht(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    
    sol = solver(problem)
    error = compute_l2_error(mesh, sol[0])
    errors.append(error)
    print(f'Mesh {Nx}x{Ny}: L2 Error = {error:.6f}')

# Plot error convergence
plt.figure(figsize=(8, 6))
plt.loglog(dh, errors, label="Computed Error", marker='o', linestyle='-', color='b')
plt.loglog(dh, errors[0] * (dh / dh[0])**(2), '--', label="O(h^2)", color='r')
plt.xlabel('Mesh Size (h)')
plt.ylabel('L2 Error')
plt.title('Error Convergence - Steady state heat')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(os.path.join(os.path.dirname(__file__), 'error_convergence_plot.png'))
plt.show()

# Save error convergence data
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
error_file = os.path.join(data_dir, 'ssht_err.txt')
with open(error_file, 'w') as f:
    for dh, error in zip(elements, errors):
        f.write(f'{dh} {error}\n')

print("Error convergence study completed. Data saved to", error_file)