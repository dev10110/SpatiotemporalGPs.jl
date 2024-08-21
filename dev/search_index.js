var documenterSearchIndex = {"docs":
[{"location":"api/kf/#Kalman-Filter","page":"Kalman Filters","title":"Kalman Filter","text":"","category":"section"},{"location":"api/kf/","page":"Kalman Filters","title":"Kalman Filters","text":"Modules = [KalmanFilter] ","category":"page"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.KFState","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.KFState","text":"KFState{V, M}\n\nA type for the Kalman Filter State, which is parameterized by the types of the mean estimate and the upper triangular cholesky component of the covariance matrix.\n\n\n\n\n\n","category":"type"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.KFState-Tuple{}","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.KFState","text":"KFState(; μ, Σ, make_symmetric=true)\n\nA constructor for the Kalman Filter State, which is parameterized by the mean estimate and the covariance matrix. If make_symmetric is true, the covariance matrix is made symmetric internally. This is useful for numerical stability.\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#LinearAlgebra.diag-Union{Tuple{LinearAlgebra.Cholesky{T}}, Tuple{T}} where T","page":"Kalman Filters","title":"LinearAlgebra.diag","text":"diag(M::Cholesky)\n\nis a fast method for getting the diagonal of a cholesky matrix.\n\nThis will eventually be included into the Julia standard library.  https://github.com/JuliaLang/julia/pull/53767\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.chol_sqrt-Tuple{Any}","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.chol_sqrt","text":"U = chol_sqrt(A)\n\nreturns an upper-triangular matrix U such that A = U^T U.\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.correct-Union{Tuple{S}, Tuple{S, Any, Any, Any}} where S<:KFState","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.correct","text":"s_{k+1|k+1} = correct(s_{k+1|k}, y_{k+1}, C, V)\n\nUses the system model\n\ny_k+1 = C x_k+1 + v\n\nwhere v sim mathcalN(0 V) to correct the predicted state.\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.kalman_filter-Union{Tuple{S}, Tuple{S, Vararg{Any, 7}}} where S<:KFState","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.kalman_filter","text":"s_{k+1} = kalman_filter(s_k, y_{k+1}, u_k, A, B, C, V, W)\n\nRuns both the prediction and the correction steps. Assumes a system model\n\n  beginalign\n  x_k+1 = A x_k + B u_k + w \n  y_k = C x_k + v\n  endalign\n\nwhere w  mathcalN(0 W), v  mathcalN(0 V).\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.predict-Union{Tuple{S}, Tuple{S, Any, Any}} where S<:KFState","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.predict","text":"s_{k+1|k} = predict(s_{k|k}, A, W)\n\nUses the system model\n\n  x_k+1 = A x_k + w\n\nwhere w  N(0 W) to predict the next state.\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.predict-Union{Tuple{S}, Tuple{S, Vararg{Any, 4}}} where S<:KFState","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.predict","text":"  s_{k+1} = predict(s_k, A, B, u_k, W)\n\nUses the system model\n\n  x_k+1 = A x_k + B u_k  + w\n\nwhere w  mathcalN(0 W) to predict the next state.\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.qrr-Tuple{Any, Any}","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.qrr","text":"R = qrr(A, B)\n\nreturns \n\nR = sqrtA^TA + B^TB\n\nThe result is an UpperTriangular matrix.\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.Σ-Tuple{S} where S<:KFState","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.Σ","text":"Σ(s::S) where {S <: KFState}\n\nGet the covariance matrix of the Kalman Filter State.\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.μ-Tuple{S} where S<:KFState","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.μ","text":"μ(s::S) where {S <: KFState}\n\nGet the mean estimate of the Kalman Filter State.\n\n\n\n\n\n","category":"method"},{"location":"api/kf/#SpatiotemporalGPs.KalmanFilter.σ-Tuple{S} where S<:KFState","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.σ","text":"σ(s::S) where {S <: KFState}\n\nGet a vector of the standard deviation of the Kalman Filter State\n\n\n\n\n\n","category":"method"},{"location":"api/stgpkf/#STGPKP","page":"STGPKF","title":"STGPKP","text":"","category":"section"},{"location":"api/stgpkf/","page":"STGPKF","title":"STGPKF","text":"Modules = [STGPKF]","category":"page"},{"location":"api/stgpkf/#SpatiotemporalGPs.STGPKF.Matern-Tuple{Any, Any, Any}","page":"STGPKF","title":"SpatiotemporalGPs.STGPKF.Matern","text":"Matern(order, σ, l)\n\ncreates a Matern kernel with order order, variance σ, and lengthscale l.  Order must be (1/2, 3/2, or 5/2).\n\n\n\n\n\n","category":"method"},{"location":"api/stgpkf/#SpatiotemporalGPs.STGPKF.kernel_matrix-Union{Tuple{KK}, Tuple{VP}, Tuple{P}, Tuple{KK, VP, VP}} where {P, VP<:AbstractVector{P}, KK<:SpatiotemporalGPs.STGPKF.AbstactKernel}","page":"STGPKF","title":"SpatiotemporalGPs.STGPKF.kernel_matrix","text":"kernel_matrix(kernel, X, Y)\n\nCompute the kernel matrix between two sets of points X and Y using the kernel function kernel. X must be a vector of points Y must be a vector of points\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SpatiotemporalGPs","category":"page"},{"location":"#SpatiotemporalGPs","page":"Home","title":"SpatiotemporalGPs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SpatiotemporalGPs.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SpatiotemporalGPs]","category":"page"},{"location":"#SpatiotemporalGPs.greet-Tuple{Any}","page":"Home","title":"SpatiotemporalGPs.greet","text":"greet(n)\n\ngreeeeeeetings\n\n\n\n\n\n","category":"method"},{"location":"kf/#Kalman-Filtering","page":"Kalman Filters","title":"Kalman Filtering","text":"","category":"section"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"This module provides an efficient and computationally stable method of computing and propagating a Kalman Filter estimate. ","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"The system model is ","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"beginalign\n  x_k+1 = A x_k + B u_k + w\n  y_k = C x_k + v\n  endalign","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"where w sim mathcalN(0 W), v sim mathcalN(0 V).","category":"page"},{"location":"kf/#Initializing","page":"Kalman Filters","title":"Initializing","text":"","category":"section"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"To initialize the KF, ","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"s_0_0 = KFState(μ=μ, Σ=P) # estimated state at time k=0","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"which creates an KFState with mean μ and covariance P. Pass in the full matrix here.","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"To get the mean, covariance, or marginal standard deviations","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"μ(s) # returns the mean\nΣ(s) # returns the full covariance matrix\nσ(s) # returns the sqrt of the diagonal of the covariance matrix","category":"page"},{"location":"kf/#Predicting-and-Correcting","page":"Kalman Filters","title":"Predicting and Correcting","text":"","category":"section"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"Then, you can run a prediction step","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"u_0 = # control input at time k=0\ns_1_0 = predict(s_0_0, A, B, u_0, W)","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"now s_1_0 = s_10  is the kf state at time k=1 conditioned on measurements upto time k=0. ","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"and then correct it use the measurement","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"y_1 = # measurement at time k=1\ns_1_1 = correct(s_1_0, y_1, C, V)","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"now s_1_1 = s_11 is the kf state at time k=1 conditioned on measurements upto time k=1. ","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"You can also do the prediction and correction in the same step:","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"s_1_1 = kalman_filter(s_0_0, y_1, u_0, A, B, C, V, W)","category":"page"},{"location":"kf/#Getters","page":"Kalman Filters","title":"Getters","text":"","category":"section"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"To get the mean, covariance or standard deviations along the diagonal","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"  μ(s) # returns the mean\n  Σ(s) # returns the full covariance matrix\n  σ(s) # returns the sqrt of the diagonal of the covariance matrix","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"These getters are useful since the s.F component of KFStruct is such that Σ = FF^T. By storing only the upper triangular part of the matrix, we have an efficient implementation that is also computationally stable. ","category":"page"},{"location":"kf/#References","page":"Kalman Filters","title":"References","text":"","category":"section"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"This module basically implemented","category":"page"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"@article{tracy2022square,\n  title={A square-root kalman filter using only qr decompositions},\n  author={Tracy, Kevin},\n  journal={arXiv preprint arXiv:2208.06452},\n  year={2022}\n}","category":"page"},{"location":"kf/#Exported-Symbols","page":"Kalman Filters","title":"Exported Symbols","text":"","category":"section"},{"location":"kf/","page":"Kalman Filters","title":"Kalman Filters","text":"Modules = [KalmanFilter]\nPrivate = false","category":"page"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.KFState-Tuple{}-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.KFState","text":"KFState(; μ, Σ, make_symmetric=true)\n\nA constructor for the Kalman Filter State, which is parameterized by the mean estimate and the covariance matrix. If make_symmetric is true, the covariance matrix is made symmetric internally. This is useful for numerical stability.\n\n\n\n\n\n","category":"method"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.KFState-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.KFState","text":"KFState{V, M}\n\nA type for the Kalman Filter State, which is parameterized by the types of the mean estimate and the upper triangular cholesky component of the covariance matrix.\n\n\n\n\n\n","category":"type"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.correct-Union{Tuple{S}, Tuple{S, Any, Any, Any}} where S<:KFState-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.correct","text":"s_{k+1|k+1} = correct(s_{k+1|k}, y_{k+1}, C, V)\n\nUses the system model\n\ny_k+1 = C x_k+1 + v\n\nwhere v sim mathcalN(0 V) to correct the predicted state.\n\n\n\n\n\n","category":"method"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.kalman_filter-Union{Tuple{S}, Tuple{S, Vararg{Any, 7}}} where S<:KFState-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.kalman_filter","text":"s_{k+1} = kalman_filter(s_k, y_{k+1}, u_k, A, B, C, V, W)\n\nRuns both the prediction and the correction steps. Assumes a system model\n\n  beginalign\n  x_k+1 = A x_k + B u_k + w \n  y_k = C x_k + v\n  endalign\n\nwhere w  mathcalN(0 W), v  mathcalN(0 V).\n\n\n\n\n\n","category":"method"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.predict-Union{Tuple{S}, Tuple{S, Any, Any}} where S<:KFState-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.predict","text":"s_{k+1|k} = predict(s_{k|k}, A, W)\n\nUses the system model\n\n  x_k+1 = A x_k + w\n\nwhere w  N(0 W) to predict the next state.\n\n\n\n\n\n","category":"method"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.predict-Union{Tuple{S}, Tuple{S, Vararg{Any, 4}}} where S<:KFState-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.predict","text":"  s_{k+1} = predict(s_k, A, B, u_k, W)\n\nUses the system model\n\n  x_k+1 = A x_k + B u_k  + w\n\nwhere w  mathcalN(0 W) to predict the next state.\n\n\n\n\n\n","category":"method"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.Σ-Tuple{S} where S<:KFState-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.Σ","text":"Σ(s::S) where {S <: KFState}\n\nGet the covariance matrix of the Kalman Filter State.\n\n\n\n\n\n","category":"method"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.μ-Tuple{S} where S<:KFState-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.μ","text":"μ(s::S) where {S <: KFState}\n\nGet the mean estimate of the Kalman Filter State.\n\n\n\n\n\n","category":"method"},{"location":"kf/#SpatiotemporalGPs.KalmanFilter.σ-Tuple{S} where S<:KFState-kf","page":"Kalman Filters","title":"SpatiotemporalGPs.KalmanFilter.σ","text":"σ(s::S) where {S <: KFState}\n\nGet a vector of the standard deviation of the Kalman Filter State\n\n\n\n\n\n","category":"method"}]
}
