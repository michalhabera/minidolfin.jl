push!(LOAD_PATH, abspath("../src/"))

using Minidolfin
using LinearAlgebra
using PyCall
using Profile
using SparseArrays

ufl = pyimport("ufl")

# UFL form
element = ufl.FiniteElement("P", ufl.triangle, 1)

n = 1000
mesh = Minidolfin.build_unit_square_mesh(n, n)
Minidolfin.compute_connectivity_tdim_d_0!(mesh, 1)
println("Number of cells=$(Minidolfin.num_entities(mesh, 2))")

dofmap = Minidolfin.build_dofmap(element, mesh)
println("Number of dofs=$(dofmap.dim)")

function a_kernel!(A::Array{Float64, 2}, cell_coords::Array{Float64, 2})::Nothing
    # Ke=∫Ωe BTe Be dΩ
    x0, y0 = cell_coords[1, 1:2]
    x1, y1 = cell_coords[2, 1:2]
    x2, y2 = cell_coords[3, 1:2]

    # 2x Element area Ae
    Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    B = [y1 - y2 y2 - y0 y0 - y1; x2 - x1 x0 - x2 x1 - x0]
    A = B' * B / (2 * Ae)
    return nothing
end

function L_kernel!(b, cell_coords)
    b[1] = cell_coords[2]
end

@time A, b = Minidolfin.assemble!(dofmap::Minidolfin.DofMap, a_kernel!::Function, L_kernel!::Function)
