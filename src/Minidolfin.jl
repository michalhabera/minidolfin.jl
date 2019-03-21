module Minidolfin

using PyCall

function __init__()
    global FIAT = pyimport("FIAT")
    global ffc = pyimport("ffc")
end

include("meshing.jl")
include("dofmap.jl")
include("assembling.jl")

end # module