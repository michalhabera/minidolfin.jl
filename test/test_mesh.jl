push!(LOAD_PATH, abspath("../src/"))

using Minidolfin
using PyCall

ufl = pyimport("ufl")

mesh = Minidolfin.build_unit_square_mesh(1, 1)
Minidolfin.compute_connectivity_tdim_d_0!(mesh, 1)

element = ufl["FiniteElement"]("P", ufl["triangle"], 1)
@time dofmap = Minidolfin.build_dofmap(element, mesh)
