# Expectation Programming Experiments

## Installation instructions

To run this code download [Julia 1.3](https://julialang.org/downloads/oldreleases/#v131_dec_30_2019) 
and follow the installation instructions.

Then open the terminal and navigate to the folder which contains 
this README and start Julia. We recommend that you allow Julia to use multiple 
threads because Annealed Importance Sampling benefits from parallelization.
This can be done as follows:
```bash
$ export JULIA_NUM_THREADS=16
$ julia
```

This should open the Julia REPL. Once you are in the Julia REPL you can use the 
Julia package manager to install all the dependcies:
```bash
julia> ]
julia> activate . # This activates the environment
julia> instantiate # This installs all the dependencies
```
See the official [documentation](https://julialang.github.io/Pkg.jl/v1/) for 
Julia's package manager for more details.

## Run Posterior Predictive Example

In the Julia REPL run
```bash
julia> include("scripts/posterior_predictive_estimate.jl")
```
This saves the experimental results to 
`data/posterior_predictive_estimate`. It also already creates all 
the plots from the paper for this example. For us getting 1,000 samples took 
around 10 seconds.

## Run Banana Example

In the Julia REPL run 
```bash
julia> include("scripts/bananas_estimate.jl")
```
This saves the experimental results to `data/bananas_estimate`. For us getting 
1,000 samples took around 20 seconds.

The ground truth was computed using `scripts/bananas_ground_truth.jl`.

## Run SIR Example

In the Julia REPL run
```bash
julia> include("scripts/sir_estimate.jl")
```
This saves the experimental results to `data/inferences`. Not all samples are 
created in one run. To get 100 samples took around 10 minutes on our machine 
(using 32 threads). We expect that the sampling can be made orders of magnitude 
faster by a more efficient implementation (e.g. writing a vectorised implementation).
To collect more samples run the script multiple 
times and make sure to change the random seed between different runs. The 
script `scripts/combine_results.jl` allows you to combine the results of multiple 
runs into one. See the bottom of the script for instructions on how to use it. 
Finally, `scripts/post_hoc_analysis.jl` creates all the necessary plots.

The script to generate the observed data can be found in `scripts/generate_data.jl`
and `scripts/is_ground_truth.jl` is used to generate the ground truth.

## Run Radon Example 

The setup for the Radon example is similar to the SIR example. In the Julia REPL
run
```bash
julia> include("scripts/radon_estimate.jl")
```
This saves the experimental results to `data/radon_estimate`. Getting 100 samples 
took around 72 minutes on our machine (using 32 threads). Again you can 
use `scripts/combine_results.jl` to combine the results from multiple runs.
Finally, `scripts/radon_combined_post_hoc_plots.jl` produces the plots.

The script `scripts/create_small_radon.jl` was used to remove counties from the 
dataset.

## Run MCMC Baselines

The MCMC baselines are run separately with the scripts 
`scripts/posterior_predictive_mh_baseline.jl`,
`scripts/sir_nuts_baseline.jl`, and `scripts/radon_nuts_baseline.jl`.
Then `scripts/combine_nuts_baseline.jl` and `script/analyse_nuts_baseline.jl` are
used to create estimates of the target expectation.