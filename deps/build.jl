import Pkg
using Pluto

println("# Activating environment for notebook ex1.jl")
Pluto.activate_notebook_environment("../ex1.jl"); 
println("# Instantiating environment for notebook ex1.jl")
Pkg.instantiate();


