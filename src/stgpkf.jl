module STGPKF
using LinearAlgebra
using StaticArrays
# this module creates a spatiotemporal GP representation


include("kernels.jl")

export Matern, kernel_matrix

end