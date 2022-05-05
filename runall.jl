# TODO: Compile sysimage

include("adjoints.jl")

adjoint_name = ""
HIDDEN_DIM = 0

for N in 2 .^(2:10), name in keys(adjoints)
    global adjoint_name = name
    global HIDDEN_DIM = N
    println(adjoint_name, HIDDEN_DIM)
    include("bvp_benchmarking.jl")
end
