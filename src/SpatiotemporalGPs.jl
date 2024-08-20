module SpatiotemporalGPs

# Write your package code here.


include("kf.jl")
include("stgpkf.jl")

using Reexport
@reexport using .KalmanFilter
@reexport using .STGPKF

"""
    greet(n)
greeeeeeetings
"""
function greet(n)
    return "Hello, $n!"
end


end
