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
        expression = 'sin(x*pi)*sin(y*pi)'
    []
    [force]
        type = ParsedFunction
        expression = '-0*lambda_*exp(sin(x*pi)*sin(y*pi)) + 2*pi^2*sin(x*pi)*sin(y*pi)'
        symbol_names = 'lambda_'
        symbol_values = '1.0'
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
    type = Steady
    solve_type = 'PJFNK'
    petsc_options_iname = '-pc_type -pc_hypre_type'
    petsc_options_value = 'hypre boomeramg'
  []
  
  [Outputs]
    file_base = 'results/bratu_err'
    exodus = true
    csv = true
  []