# Spatiotemporal Gaussian Process Kalman Filtering

## Theory
STGPKF is a modelling technique to estimate the state of an environment.

In particular, the goal is to estimate a spatiotemporal field 

```math
f : \mathcal{R} \times \mathcal{D} \to \mathcal{R}
```
i.e., $f(t, p)$ is the value of the field at time ``t`` and position ``p \in \mathcal{D} \subset \mathcal{R}^d``. 

The assumption is that the field is a realization of a Gaussian Process:

```math
f \sim \operatorname{GP}(m, k)
```
with mean function ``m(t, p) = 0`` and kernel function 
```math
k(t, t', p, p') = k_t(t, t') k_s(p, p')
```
Notice, we explicitly assume the kernel is separable in space and time. 


```@autodocs; canonical=false
Modules = [STGPKF]
Private = false
```
