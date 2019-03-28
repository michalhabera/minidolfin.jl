function write_vtk(vector::Array{Float64, 1}, name::String, dofmap::DofMap, filename::String, mode::String)

    UFL_TO_VTK = Dict(ufl.quadrilateral => [[1, 2, 4, 3], 9])

    num_cells = num_entities(dofmap.mesh, 2)
    num_vertices = num_entities(dofmap.mesh, 0)
    vert_per_cell = length(dofmap.mesh.topology[(2, 0)][1, :])

    perm = UFL_TO_VTK[dofmap.element.cell()][1]
    vtk_cell_id = UFL_TO_VTK[dofmap.element.cell()][2]
    gdim = length(dofmap.mesh.vertices[1, :])

    open(filename, mode) do f
        write(f, "# vtk DataFile Version 2.0\n")
        write(f, "Minidolfin.jl output\n")
        write(f, "ASCII\n")
        write(f, "DATASET UNSTRUCTURED_GRID\n")

        write(f, "POINTS $num_vertices float\n")
        for vertices in eachrow(dofmap.mesh.vertices)
            write(f, "$(join(vcat(vertices, zeros(3-gdim)), " "))\n")
        end

        write(f, "CELLS $num_cells $((vert_per_cell + 1) * num_cells)\n")
        for vertices_id in eachrow(dofmap.mesh.topology[(2, 0)])
            # -1 for Julia index shift
            write(f, "$vert_per_cell $(join(view(vertices_id .- 1, perm), " "))\n")
        end

        write(f, "CELL_TYPES $num_cells\n")
        for cell_id in 1:num_cells
            write(f, "$vtk_cell_id\n")
        end

        write(f, "POINT_DATA $num_vertices\n")
        write(f, "SCALARS $name double 1\n")
        write(f, "LOOKUP_TABLE default\n")
        for val in vector
            write(f, "$val\n")
        end
    end


end