# Quickstart


## Import 

```julia
using StaticArrays, LinearAlgebra, Plots
using SpatiotemporalGPs
```

## Define the Problem Domain

```julia
# setup the spatial and temporal kernels
σt = 2.0   # m/s
σs = 1.0   # m/s
lt = 3*60.0  # minutes
ls = 3.0   # km

kt = Matern(3/2, σt, lt)
ks = Matern(1/2, σs, ls)

# determine the temporal step size
Δt = 5.0 # minutes
Δx = 0.25 # km

# create the spatial domain
xs = 0:Δx:7.0
ys = 0:Δx:10.0

grid_pts = vec([@SVector[x, y] for x in xs, y in ys]);
```

## Initialize 

```julia
problem = STGPKFProblem(grid_pts, ks, kt, Δt)
state_initial = stgpkf_initialize(problem) 
```

## Assimilate a single point measurement

```julia

# for a single measurement
measured_pt = @SVector [ ... ] 
measured_value = ...
measured_σ = ...

state_corrected = stgpkf_correct(problem, state_initial, measured_pt, measured_y, measured_σ)
```


## Assimilate a set of measurements

```julia

# define all the measurements
measured_pts = [ ... ] # should be a vector of positions that were measured
measured_ys = [ ... ] # should be a vector of measured values
measured_Σ = [ ... ]  # should be a pos def matrix


state_corrected = stgpkf_correct(problem, state_initial, measured_pts, measured_ys, measured_Σ)
```

## Predict the next timestemp

```julia

state_predicted = stgpkf_predict(problem, state_corrected)
```

## Generate Plots

```julia
plot(problem, state; plot_type=:estimate, kwargs...)
```
where `plot_type` can be `:estimate, :std, :clarity, :percentile`. If it is `percentile` use the `percentile=0.95` kwarg to specify which percentile to visualize. Currently only supports visualization of 2D domains. 

## Extract Data

To get the estimate of the spatiotemporal field
```julia
get_estimate(problem, state)
```
or you can also call
```julia
get_estimate_covariance(problem, state)
get_estimate_std(problem, state)
get_estimate_clarity(problem, state)
```


See `examples/synthetic_data.ipynb` for a full demo. 