push!(LOAD_PATH, abspath("../src/"))

using Minidolfin
using LinearAlgebra
using PyCall
using Profile
using PyPlot

ufl = pyimport("ufl")

# UFL form
element = ufl.FiniteElement("P", ufl.triangle, 1)

n = 100
const mesh = Minidolfin.unit_square_mesh(n, n)
Minidolfin.compute_connectivity_tdim_d_0!(mesh, 1)
println("Number of cells=$(Minidolfin.num_entities(mesh, 2))")

const dofmap = Minidolfin.build_dofmap(element, mesh)
println("Number of dofs=$(dofmap.dim)")


function a_poisson!(A::Array{Float64, 2}, cell_coords::Array{Float64, 2})::Nothing

    x1 = view(cell_coords, 1, 1:2)
    x2 = view(cell_coords, 2, 1:2)
    x3 = view(cell_coords, 3, 1:2)

    J = [x2 - x1 x3 - x1]

    Jinv = inv(J)
    absdetJ = abs(det(J))

    gradPhi = [-1.0 -1.0; 1.0 0.0; 0.0 1.0]

    K = gradPhi * Jinv
    # /2 for quadrature weight
    A[1:3, 1:3] = K * K' * absdetJ / 2.0

    return nothing
end


function L_poisson!(b::Array{Float64, 1}, cell_coords::Array{Float64, 2})

    x1 = view(cell_coords, 1, 1:2)
    x2 = view(cell_coords, 2, 1:2)
    x3 = view(cell_coords, 3, 1:2)

    J = [x2 - x1 x3 - x1]

    Jinv = inv(J)
    absdetJ = abs(det(J))

    f = 1.0
    # /3 for value of basis at (1/3, 1/3)
    # /2 for quadrature weight
    b[1:3] .= absdetJ * f / 3.0 / 2.0

end


# Prepare boundary condition mapping
const bc_dofs = zeros(Bool, dofmap.dim)
const bc_vals = zeros(Float64, dofmap.dim)
bottom = 1:(n+1)
left = (n+1):(n+1):(n+1)^2
right = 1:(n+1):(n+1)^2
top = ((n+1)^2-n):(n+1)^2

for i in vcat(bottom, left, right, top)
    bc_dofs[i] = true
    bc_vals[i] = 1.0
end


@time A, b = Minidolfin.assemble!(dofmap, a_poisson!, L_poisson!, bc_dofs, bc_vals)

sol = A\b

xs = mesh.vertices[:, 1]
ys = mesh.vertices[:, 2]

gca(aspect="equal")
triplot(xs, ys, mesh.topology[(2,0)] .- 1, color="black", linewidth=0.1)
ax = tripcolor(xs, ys, mesh.topology[(2,0)] .- 1, sol, shading="gouraud")
colorbar(ax)
savefig("poisson2d.png")
