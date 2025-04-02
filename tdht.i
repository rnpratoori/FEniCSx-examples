[Mesh]
    type = GeneratedMesh
    dim = 2
    nx = 8
    ny = 8
  []
  
  [Variables]
    [u][]
  []
  
  [Kernels]
    [time]
      type = ADTimeDerivative
      variable = u
    []
    [diff]
      type = ADDiffusion
      variable = u
    []
    [force]
      type = BodyForce
      variable = u
      function = force
    []
  []
  
  [Functions]
    [exact]
        type = ParsedFunction
        expression = 'exp(-t)*sin(x*pi)*sin(y*pi)'
      []
    [force]
        type = ParsedFunction
        expression = '-exp(-t)*sin(x*pi)*sin(y*pi) + 2*pi^2*exp(-t)*sin(x*pi)*sin(y*pi)'
    []
  []
  
  [BCs]
    [all]
      type = FunctionDirichletBC
      variable = u
      function = exact
      boundary = 'left right top bottom'
    []
  []
  
  [Postprocessors]
    [error]
      type = ElementL2Error
      function = exact
      variable = u
    []
    [h]
      type = AverageElementSize
    []
  []
  
  [Executioner]
    type = Transient
    dt = 1
    end_time = 3
    solve_type = 'PJFNK'
    # Direct LU solver settings
    petsc_options_iname = '-pc_type -pc_hypre_type'
    petsc_options_value = 'hypre boomeramg'
  []
  
  [Outputs]
    file_base = 'results/tdht_err'
    exodus = true
    csv = true
  []