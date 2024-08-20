module SpatiotemporalGPs

# Write your package code here.


include("types.jl")
include("utils.jl")
include("kf.jl")

using Reexport
@reexport using .KalmanFilter

# export KalmanFilter.greet

end
