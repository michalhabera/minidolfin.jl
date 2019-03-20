push!(LOAD_PATH, abspath("../src/"))

using Minidolfin

mesh = Minidolfin.build_unit_square_mesh(50, 50)
@time Minidolfin.compute_connectivity_tdim_d_0!(mesh, 1)
