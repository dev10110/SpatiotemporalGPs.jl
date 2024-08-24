module SpatiotemporalGPs

include("kf.jl")
include("stgpkf.jl")

using Reexport
@reexport using .KalmanFilter
@reexport using .STGPKF

end
