
using SpatiotemporalGPs
using Test
using LinearAlgebra
using StaticArrays

@testset "SpatiotemporalGPs.jl" begin end

include("tests/kf.jl")
include("tests/kernels.jl")
include("tests/stgpkf.jl")
include("tests/cuda_kron.jl")
include("tests/cuda_stgpkf.jl")

