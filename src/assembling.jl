using SparseArrays

function assemble(dofmap::DofMap, a_kernel, L_kernel, dtype=Float64)

    mesh = dofmap.mesh
    dofsize = dofmap.dim
    tdim = mesh.reference_cell.get_dimension()

    dofsize_local = size(dofmap.cell_dofs)[2]
    num_values = dofsize_local ^ 2
    num_cells = num_entities(mesh, tdim)
    cell_vert_conn = mesh.topology[(tdim, 0)]

    vert_per_cell = size(cell_vert_conn)[2]
    gdim = size(mesh.vertices)[2]

    # Global RHS vector
    b = zeros(dtype, dofsize)

    # Vector of non-zero values in the global matrix
    # These values will be scattered according to "sparsity pattern"
    values = dtype[]

    n = 1
    # Storage for coordinates of cell vertices
    cell_coords = zeros(Float32, vert_per_cell, gdim)

    # Storage for insertion index sets
    I = Int64[]
    J = Int64[]

    A_local = zeros(dtype, dofsize_local, dofsize_local)
    b_local = zeros(dtype, dofsize_local)

    for cell_id in 1:num_cells
        # Zero out local element tensors
        fill!(A_local, zero(dtype))
        fill!(b_local, zero(dtype))

        # Pack coordinates of vertices on this cell
        for vert_id in 1:vert_per_cell
            cell_coords[vert_id, :] = mesh.vertices[cell_vert_conn[cell_id, vert_id], :]
        end

        a_kernel(A_local, cell_coords)
        L_kernel(b_local, cell_coords)

        # Scatter to global positions
        for (i, iglobal) in enumerate(dofmap.cell_dofs[cell_id, :])
            b[iglobal] += b_local[i]
            for j in 1:dofsize_local
                val = A_local[i, j]
                if val != 0.0
                    push!(values, val)
                    push!(I, iglobal)
                    push!(J, dofmap.cell_dofs[cell_id, j])
                end
                n += 1
            end
        end
    end

    # Return CSC sparse array
    sparse(I, J, values), b
end
