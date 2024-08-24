# SpatiotemporalGPs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dev10110.github.io/SpatiotemporalGPs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dev10110.github.io/SpatiotemporalGPs.jl/dev/)
[![Build Status](https://github.com/dev10110/SpatiotemporalGPs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dev10110/SpatiotemporalGPs.jl/actions/workflows/CI.yml?query=branch%3Amain)


![](spatiotemporal.gif)

## Testing

`cd` into this directory, and run `julia`. From inside the REPL do
```
] test
```

## Making Docs
To make docs locally, `cd` into this directory and run `julia`. From inside the REPL do 
```
using Revise
] activate docs
] resolve
include("docs/make.jl")
```
then from a new terminal, run a `LiveServer` to see the constructed docs:
```
julia -e 'using LiveServer; serve(dir="docs/build")'
```


