mutable struct Mesh
    vertices
    topology
    reference_cell
end


function build_unit_square_mesh(nx, ny)
    x = LinRange(0, 1, nx + 1)
    y = LinRange(0, 1, ny + 1)
    vertices = Iterators.product(x, y)
    cells = zeros(Int64, 2 * nx * ny, 3)

    for iy in 0:(ny-1)
        for ix in 0:(nx-1)
            v0 = iy * (nx + 1) + ix + 1
            v1 = v0 + 1
            v2 = v0 + (nx + 1)
            v3 = v1 + (nx + 1)

            c0 = 2*(iy*nx + ix)
            cells[c0 + 1, :] = [v0, v1, v2]
            cells[c0 + 2, :] = [v1, v2, v3]
        end
    end

    ref_cell = FIAT["reference_element"]["ufc_cell"]("triangle")
    Mesh(vertices, Dict((2, 0) => cells), ref_cell)
end


"""Number of mesh entities of given dimension"""
function num_entities(mesh::Mesh, dim)
    length(mesh.topology[(dim, 0)][:, 1])
end


"""Compute connectivity between (tdim, d) and (d, 0) mesh entities"""
function compute_connectivity_tdim_d_0!(mesh::Mesh, d)
    tdim = mesh.reference_cell["get_dimension"]()
    cell_vertex_conn = mesh.topology[(tdim, 0)]

    # Get local (FIAT) connectivity
    fiat_connectivity = mesh.reference_cell["get_connectivity"]()
    ent_vert_conn_local = fiat_connectivity[(d, 0)]
    cell_ent_conn_local  = fiat_connectivity[(tdim, d)]

    ent_per_cell = length(ent_vert_conn_local)
    vertices_per_ent = length(ent_vert_conn_local[1])
    num_cells = num_entities(mesh, tdim)

    # Allocate space for global connectivities
    ent_vert_conn = zeros(Int64, num_cells * ent_per_cell, vertices_per_ent)
    cell_ent_conn = zeros(Int64, num_cells, ent_per_cell)

    for cell_idx in 1:num_cells
        for ent_idx in 1:(ent_per_cell)
            for vert_idx in 1:(vertices_per_ent)
                # Get local index for vertex on this entity
                # +1 for python --> Julia indexing shift
                vert_for_entity = ent_vert_conn_local[ent_idx][vert_idx] + 1
                # Fill global connectivities
                ent_vert_conn[(cell_idx - 1) * ent_per_cell + ent_idx, vert_idx] = cell_vertex_conn[cell_idx, vert_for_entity]
            end
        end
    end

    ent_vert_conn, cell_ent_conn = unique_inverse(ent_vert_conn)
    cell_ent_conn = reshape(cell_ent_conn, num_cells, ent_per_cell)

    mesh.topology[(tdim, d)] = cell_ent_conn
    mesh.topology[(d, 0)] = ent_vert_conn
end


function unique_inverse(arr)

    inverse = Dict()
    for (i, row) in enumerate(eachrow(arr))
        if haskey(inverse, row)
            # If the row is already in keys then append to the
            # indices under which it is found
            push!(inverse[row], i)
        else
            inverse[row] = [i]
        end
    end

    # Populate arrays
    unique = zeros(Int64, length(inverse), size(arr)[2])
    unique_inverse = zeros(Int64, size(arr)[1])

    for (k, (row, I)) in enumerate(inverse)
        unique[k, :] = row
        for iI in I
            unique_inverse[iI] = k
        end
    end

    (unique, unique_inverse)
end