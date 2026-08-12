
using SpatiotemporalGPs
using Test
using LinearAlgebra
using StaticArrays
using CUDA

@testset "SpatiotemporalGPs.jl" begin end

include("tests/kf.jl")
include("tests/kernels.jl")
include("tests/stgpkf.jl")

if CUDA.functional()
    include("tests/cuda_kron.jl")
    include("tests/cuda_stgpkf.jl")
end

