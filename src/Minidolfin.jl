module Minidolfin

using PyCall

function __init__()
    global FIAT = pyimport("FIAT")
    global ffc = pyimport("ffc")
    global ufl = pyimport("ufl")
end

include("mesh.jl")
include("dofmap.jl")
include("assemble.jl")
include("plot.jl")
include("io.jl")

end # module