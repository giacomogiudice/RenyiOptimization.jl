# RenyiOptimization.jl

[![CI][ci-img]][ci-url]
[![codecov][codecov-img]][codecov-url]

[ci-img]: https://github.com/giacomogiudice/RenyiOptimization.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/giacomogiudice/RenyiOptimization.jl/actions?query=workflow%3ACI

[codecov-img]: https://codecov.io/gh/giacomogiudice/RenyiOptimization.jl/branch/master/graph/badge.svg?token=fQlukhogec
[codecov-url]: https://codecov.io/gh/giacomogiudice/RenyiOptimization.jl/

## Introduction
This package optimizes the _Rényi free energy_

<!-- See hack in https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b  -->
![formula](https://render.githubusercontent.com/render/math?math=%5CLarge%20F(%5Crho)%20%3D%20%5Cmathrm%7BTr%7D(H%5Crho)%20%2B%20%5Cfrac%7B1%7D%7B%5Cbeta%7D%5Clog(%5Crho%5E2))

over a matrix-product-state purification of the density matrix.
It leverages the [TensorKit.jl](https://github.com/Jutho/TensorKit.jl/) package for all tensor manipulations, and [OptimKit.jl](https://github.com/jutho/OptimKit.jl) for nonlinear optimization on Riemannian manifolds.

## Installation

Currently this package is not registered.
Enter `]` in the REPL to access the package manager and type
```
pkg> add https://github.com/giacomogiudice/RenyiOptimization.jl/
```
to add the package to the current environment.

## Scripts

To run the optimization on the so-called XY model, you can use the scripts from the `scripts` directory.
They can be run directly from the command line, for example as `$ julia --project=@. routines/beta.jl [--options]`.
Note that you may need to install some additional packages.

## Compatibility

The package is compatible with Julia 1.4 and above.

