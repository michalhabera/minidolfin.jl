using PyPlot

function plot(vector::Array{Float64, 1}, dofmap::DofMap, filename::String)
    mesh = dofmap.mesh

    if dofmap.element.cell() == ufl.triangle

        # Prepare handles to vertex coords
        xs = mesh.vertices[:, 1]
        ys = mesh.vertices[:, 2]

        gca(aspect="equal")
        # -1 in topology for Python indexing
        triplot(xs, ys, mesh.topology[(2,0)] .- 1, color="black", linewidth=0.1)
        ax = tripcolor(xs, ys, mesh.topology[(2,0)] .- 1, vector, shading="gouraud")
        colorbar(ax)
        savefig(filename)
    else
        throw("Unsupported dofmap")
    end
end