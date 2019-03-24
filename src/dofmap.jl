struct DofMap
    cell_dofs::Array{Int64, 2}
    dim::Int64
    mesh::Mesh
    element
end


function build_dofmap(element, mesh::Mesh)
    fiat_element = ffc.fiatinterface.create_element(element)
    tdim = mesh.reference_cell.get_dimension()::Int64

    cell_dofs = zeros(Int64, num_entities(mesh, tdim), fiat_element.space_dimension())

    # Explicit conversion
    fiat_entity_dofs = convert(Dict{Int64, Dict{Int64, Array{Int64, 1}}}, fiat_element.entity_dofs())

    offset = 0
    for (dim, local_dofs) in fiat_entity_dofs
        # For each set of entities (and its local dofmaps) of dimension dim
        dofs_per_entity = length(local_dofs[0])

        # Take a handle to connectivity array 
        if tdim == dim
            connectivity = 1:num_entities(mesh, tdim)
        else
            connectivity = mesh.topology[(tdim, dim)]
        end

        for k in 1:dofs_per_entity
            for (entity, entity_dofs) in local_dofs
                cell_dofs[:, entity_dofs[k] + 1] = dofs_per_entity * (connectivity[:, entity + 1] .- 1) .+ (offset + k)
            end
        end
        offset += dofs_per_entity * num_entities(mesh, dim)
    end
    DofMap(cell_dofs, maximum(cell_dofs), mesh, element)
end