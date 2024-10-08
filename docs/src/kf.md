# (Normal) Kalman Filtering

This module provides an efficient and computationally stable method of computing and propagating a Kalman Filter estimate. 
This module assumes a linear discrete time system. It can be a time-varying system. 

The system model is 

```math
\begin{align*}
  x_{k+1} &= A x_k + B u_k + w,\\
  y_k &= C x_k + v
  \end{align*}
```
where ``w \sim \mathcal{N}(0, W)``, ``v \sim \mathcal{N}(0, V)``.


Given ``x_{k} \sim \mathcal{N}(\mu_{k|k}, P_{k|k})``, and the prediction step implements 
```math
  \begin{align*}
  \mu_{k+1|k} &= A \mu_{k|k} + B u_k\\
  P_{k+1|k} &= A P_{k|k} A^T + W
  \end{align*}
```

Then, given a measurement ``y_{k+1}``, the correction step implements
```math
\begin{align*}
\mu_{k+1|k+1} &= \mu_{k+1|k} + K ( y_{k+1} - C \mu_{k+1|k})\\
P_{k+1|k+1} &= (I - K C) P_{k+1|k}\\
K &= P_{k+1|k} C^T (C P_{k+1|k} C^T + V)^{-1}
\end{align*}
```

## Initializing
To initialize the KF, 

```julia
s_0_0 = KFState(μ=μ, Σ=P) # estimated state at time k=0
```
which creates an `KFState` with mean ``μ`` and covariance ``P``. Pass in the full matrix here.

To get the mean, covariance, or marginal standard deviations
```julia
μ(s) # returns the mean
Σ(s) # returns the full covariance matrix
σ(s) # returns the sqrt of the diagonal of the covariance matrix
```

## Predicting and Correcting
You can run a prediction step
```julia
u_0 = # control input at time k=0
s_1_0 = predict(s_0_0, A, B, u_0, W)
```
now `s_1_0` = ``s_{1|0}``  is the kf state at time ``k=1`` conditioned on measurements upto time ``k=0``. 


and then correct it use the measurement
```julia
y_1 = # measurement at time k=1
s_1_1 = correct(s_1_0, y_1, C, V)
```
now `s_1_1` = ``s_{1|1}`` is the kf state at time ``k=1`` conditioned on measurements upto time ``k=1``. 

You can also do the prediction and correction in the same step:
```julia
s_1_1 = kalman_filter(s_0_0, y_1, u_0, A, B, C, V, W)
```
## Extracting State and Covariances
To get the mean, covariance or standard deviations along the diagonal
```julia
  μ(s) # returns the mean
  Σ(s) # returns the full covariance matrix
  σ(s) # returns the sqrt of the diagonal of the covariance matrix
```

These getters are useful since the `s.F` component of `KFStruct` is such that ``Σ = FF^T``. By storing only the upper triangular part of the matrix, we have an efficient implementation that is also computationally stable. 


## References
This module basically implemented
```bibtex
@article{tracy2022square,
  title={A square-root kalman filter using only qr decompositions},
  author={Tracy, Kevin},
  journal={arXiv preprint arXiv:2208.06452},
  year={2022}
}
```


## Exported Symbols
```@autodocs; canonical=false
Modules = [KalmanFilter]
Private = false
```