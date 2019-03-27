push!(LOAD_PATH, abspath("../src/"))

using Minidolfin
using LinearAlgebra
using PyCall
using Profile
# using PyPlot

ufl = pyimport("ufl")

# UFL form
element = ufl.FiniteElement("P", ufl.quadrilateral, 1)

n = 20

const mesh = Minidolfin.unit_square_mesh(n, n, cell="quadrilateral")
Minidolfin.compute_connectivity_tdim_d_0!(mesh, 1)
println("Number of cells=$(Minidolfin.num_entities(mesh, 2))")

const dofmap = Minidolfin.build_dofmap(element, mesh)
println("Number of dofs=$(dofmap.dim)")


function a_poisson_tri!(A::Array{Float64, 2}, cell_coords::Array{Float64, 2})::Nothing

    x1 = cell_coords[1, 1:2]
    x2 = cell_coords[2, 1:2]
    x3 = cell_coords[3, 1:2]

    J = [x2 - x1 x3 - x1]

    Jinv = inv(J)
    absdetJ = abs(det(J))

    gradPhi = [-1.0 -1.0; 1.0 0.0; 0.0 1.0]

    K = gradPhi * Jinv
    # /2 for quadrature weight
    A[1:3, 1:3] = K * K' * absdetJ / 2.0

    return nothing
end


function L_poisson_tri!(b::Array{Float64, 1}, cell_coords::Array{Float64, 2})

    x1 = cell_coords[1, 1:2]
    x2 = cell_coords[2, 1:2]
    x3 = cell_coords[3, 1:2]

    J = [x2 - x1 x3 - x1]

    Jinv = inv(J)
    absdetJ = abs(det(J))

    f = 1.0
    # /3 for value of basis at (1/3, 1/3)
    # /2 for quadrature weight
    b[1:3] .= absdetJ * f / 3.0 / 2.0

end


function a_poisson_quad!(A::Array{Float64, 2}, cell_coords::Array{Float64, 2})::Nothing

    x1 = cell_coords[1, 1:2]
    x2 = cell_coords[2, 1:2]
    x3 = cell_coords[3, 1:2]
    x4 = cell_coords[4, 1:2]

    println(x1, x2, x3, x4)

    J1 = -x1 / 4 + x2 / 4 - x3 / 4 + x4 / 4
    J2 = -x1 / 4 - x2 / 4 + x3 / 4 + x4 / 4

    J = [J1 J2]

    Jinv = inv(J)
    absdetJ = abs(det(J))

    gradPhi = [- 1/4 -1/4; 1/4 -1/4; -1/4 1/4; 1/4 1/4]

    K = gradPhi * Jinv
    # *2 for quadrature weight
    A[1:4, 1:4] = K * K' * absdetJ * 2

    return nothing
end


function L_poisson_quad!(b::Array{Float64, 1}, cell_coords::Array{Float64, 2})

    x1 = cell_coords[1, 1:2]
    x2 = cell_coords[2, 1:2]
    x3 = cell_coords[3, 1:2]
    x4 = cell_coords[4, 1:2]

    J1 = -x1 / 4 + x2 / 4 - x3 / 4 + x4 / 4
    J2 = -x1 / 4 - x2 / 4 + x3 / 4 + x4 / 4

    J = [J1 J2]

    Jinv = inv(J)
    absdetJ = abs(det(J))

    f = 1.0
    # /4 for value of basis at (0, 0)
    # *2 for quadrature weight
    b[1:4] .= absdetJ * f / 4.0 * 2.0

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


@time A, b = Minidolfin.assemble!(dofmap, a_poisson_quad!, L_poisson_quad!, bc_dofs, bc_vals)

# sol = A\b

# xs = mesh.vertices[:, 1]
# ys = mesh.vertices[:, 2]

# gca(aspect="equal")
# triplot(xs, ys, mesh.topology[(2,0)] .- 1, color="black", linewidth=0.1)
# ax = tripcolor(xs, ys, mesh.topology[(2,0)] .- 1, sol, shading="gouraud")
# colorbar(ax)
# savefig("poisson2d.png")
