using SparseArrays
using Profile

"""
    assemble!(dofmap, a_kernel!, L_kernel!, bc_dofs, bc_vals)

Assemble bilinear and linear kernels, `a_kernel`, `L_kernel` and
applies boundary contitions in symmetric way.

Allocates and returns sparse CSC matrix and dense vector
of rhs.

"""
function assemble!(dofmap::DofMap, a_kernel!::Function, L_kernel!::Function,
    bc_dofs::Array{Bool, 1}, bc_vals::Array{Float64, 1})

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

        # Pack vertex coordinates of the cell
        cell_coords[1:vert_per_cell, 1:gdim] = mesh.vertices[cell_vert_conn[cell_id, 1:vert_per_cell], 1:gdim]

        a_kernel!(A_local, cell_coords)
        L_kernel!(b_local, cell_coords)

        # Append values to COO format
        for (i, iglobal) in enumerate(dofmap.cell_dofs[cell_id, 1:dofsize_local])

            if bc_dofs[iglobal]
                A_local[i, 1:dofsize_local] .= 0.0
                b_local[1:dofsize_local] -= A_local[1:dofsize_local, i] * bc_vals[iglobal]
                A_local[1:dofsize_local, i] .= 0.0
                A_local[i, i] = 1.0
                b_local[i] = bc_vals[iglobal]
            end

            b[iglobal] += b_local[i]
            for j in 1:dofsize_local
                # Push only nonzero values to avoid later "eliminate zeros" call
                if A_local[i, j] != 0.0
                    push!(values, A_local[i, j])
                    push!(I, iglobal)
                    push!(J, dofmap.cell_dofs[cell_id, j])
                end
            end
        end
    end

    # Return CSC sparse array
    A = sparse(I, J, values)

    return A, b
end
