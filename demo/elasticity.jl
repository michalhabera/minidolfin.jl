push!(LOAD_PATH, abspath("../src/"))

using Minidolfin
using PyCall

ufl = pyimport("ufl")

# UFL form
element = ufl.VectorElement("P", ufl.triangle, 1)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)

E = 10.0
nu = 0.25
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


function epsilon(v)
    0.5*(ufl.grad(v) + ufl.grad(v).T)
end


function sigma(v)
    2.0*mu*epsilon(v) + lmbda*ufl.tr(epsilon(v)) * ufl.Identity(v.geometric_dimension())
end


a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

n = 1
mesh = Minidolfin.build_unit_square_mesh(n, n)
Minidolfin.compute_connectivity_tdim_d_0!(mesh, 1)
println("Number of cells=$(Minidolfin.num_entities(mesh, 2))")

dofmap = Minidolfin.build_dofmap(element, mesh)
println("Number of dofs=$(dofmap.dim)")

function a_kernel(A, cell_coords)
    fill!(A, 1.0)
end

function L_kernel(b, cell_coords)
    b[1] = cell_coords[2]
end

@time A, b = Minidolfin.assemble(dofmap, a_kernel, L_kernel)
