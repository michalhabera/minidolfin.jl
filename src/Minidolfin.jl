module Minidolfin

using PyCall

function __init__()
    global FIAT = pyimport("FIAT")
end

include("meshing.jl")

end # module