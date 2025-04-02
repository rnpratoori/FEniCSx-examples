[Mesh]
    type = GeneratedMesh
    dim = 2
    nx = 8
    ny = 8
  []
  
  [Modules]
    [./PhaseField]
      [./Conserved]
        [./c]
          free_energy = fbulk
          mobility = M
          kappa = kappa_c
          solve_type = REVERSE_SPLIT
        [../]
      [../]
    [../]
  []

  [ICs]
    [./cIC]
      type = RandomIC
      variable = c
      min = -0.1
      max =  0.1
    [../]
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

  [BCs]
    [./Periodic]
      [./all]
        auto_direction = 'x y'
      [../]
    [../]
  []
  
  
  [BCs]
    [all]
      type = FunctionDirichletBC
      variable = u
      function = exact
      boundary = 'left right top bottom'
    []
  []

  [Materials]
    [./mat]
      type = GenericConstantMaterial
      prop_names  = 'M kappa_c'
      prop_values = '1.0 0.5'
    [../]
    [./free_energy]
      type = DerivativeParsedMaterial
      property_name = fbulk
      coupled_variables = c
      constant_names = W
      constant_expressions = 1.0/2^2
      expression = W*(1-c)^2*(1+c)^2
      enable_jit = true
      outputs = exodus
    [../]
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
    solve_type = NEWTON
    scheme = bdf2
  
    petsc_options_iname = '-pc_type -sub_pc_type'
    petsc_options_value = 'asm      lu          '
  
    l_max_its = 30
    l_tol = 1e-4
    nl_max_its = 20
    nl_rel_tol = 1e-9
  
    dt = 2.0
    end_time = 20.0
  []
  
  [Outputs]
    file_base = 'results/tdht_err'
    exodus = true
    csv = true
  []