using SparseArrays
using Profile

"""
    assemble!(dofmap::DofMap, a_kernel!::Function, L_kernel::Function)

Assemble bilinear and linear kernels, `a_kernel`, `L_kernel`.
Allocates and returns sparse CSC matrix and dense vector.

"""
function assemble!(dofmap::DofMap, a_kernel!::Function, L_kernel::Function)

    mesh = dofmap.mesh
    dofsize = dofmap.dim
    tdim = mesh.reference_cell.get_dimension()::Int64

    dofsize_local = size(dofmap.cell_dofs)[2]
    num_values = dofsize_local ^ 2
    num_cells = num_entities(mesh, tdim)
    cell_vert_conn = mesh.topology[(tdim, 0)]

    vert_per_cell = size(cell_vert_conn)[2]
    gdim = size(mesh.vertices)[2]

    # Vector of non-zero values in the global matrix
    # These values will be scattered according to "sparsity pattern"
    values = Array{Float64, 1}()

    # This index is counting number of nonzero values inserted into `values` vector
    n = 1

    # Storage for coordinates of cell vertices
    cell_coords = zeros(Float64, vert_per_cell, gdim)

    # Storage for insertion index sets
    I = Array{Int64, 1}()
    J = Array{Int64, 1}()

    b = zeros(Float64, dofsize)

    A_local = zeros(Float64, dofsize_local, dofsize_local)
    b_local = zeros(Float64, dofsize_local)

    for cell_id in 1:num_cells

        fill!(A_local, 0.0)
        fill!(b_local, 0.0)

        pack_coordinates!(cell_coords, mesh, cell_vert_conn, vert_per_cell, gdim, cell_id)

        a_kernel!(A_local, cell_coords)
        L_kernel(b_local, cell_coords)

        # Copy out cell dofs to contigous data array
        cell_dofs = dofmap.cell_dofs[cell_id, 1:dofsize_local]

        # Append values to COO format
        append_vals!(cell_dofs, b, b_local, A_local, values, I, J, n, dofsize_local)
    end

    # Return CSC sparse array
    A = sparse(I, J, values)
    return A, b
end


"""Append local element values to global vector and coordinate indices"""
function append_vals!(cell_dofs::Array{Int64, 1}, b::Array{Float64, 1},
    b_local::Array{Float64, 1}, A_local::Array{Float64, 2}, values::Array{Float64, 1},
    I::Array{Int64, 1}, J::Array{Int64, 1}, n::Int64, dofsize_local::Int64)

    for (i, iglobal) in enumerate(cell_dofs[1:dofsize_local])
        b[iglobal] += b_local[i]
        for j in 1:dofsize_local
            val = A_local[i, j]
            if val != 0.0
                push!(values, val)
                push!(I, iglobal)
                push!(J, cell_dofs[j])
            end
            n += 1
        end
    end
end


function pack_coordinates!(cell_coords::Array{Float64, 2}, mesh::Mesh, cell_vert_conn::Array{Int64, 2},
    vert_per_cell::Int64, gdim::Int64, cell_id::Int64)
    for j in 1:gdim
        for i in 1:vert_per_cell
            cell_coords[i, j] = mesh.vertices[cell_vert_conn[cell_id, i], j]
        end
    end
end
