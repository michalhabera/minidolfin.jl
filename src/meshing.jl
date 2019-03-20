using PyCall
FIAT = pyimport("FIAT")

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
            v0 = iy * (nx + 1) + ix
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
