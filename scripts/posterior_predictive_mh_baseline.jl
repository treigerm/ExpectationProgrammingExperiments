using DrWatson
@quickactivate "EPExperiments"

using EPExperiments
using Random
using Logging
using LoggingExtras
using JLD2
using FileIO
using CSV
using DataFrames
using LinearAlgebra: I

const LOG_DIR_NAME = "posterior_predictive_mh_baseline"
const LOG_FILENAME = "out.log"

@with_kw struct PostPredMHBaseline
    experiment_name::String = "test"
    num_samples::Int = 100
    num_chains::Int = 4
    seed::Int = 1234
end

function post_pred_mh_baseline(experiment_config)
    @unpack seed, num_samples, num_chains = experiment_config

    result_folder = mkpath(joinpath(
        datadir(LOG_DIR_NAME),
        savename(experiment_config)
    ))

    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

    Random.seed!(seed)

    post_pred_turing = post_pred(POST_PRED_Y_OBSERVED).gamma2

    @time begin
    nuts_samples = sample(
        post_pred_turing, 
        Turing.MH(Matrix{Float64}(I, POST_PRED_DIMENSION, POST_PRED_DIMENSION)), 
        MCMCThreads(),
        num_samples,
        num_chains;
        discard_adapt=false
    )
    end

    @tagsave(joinpath(result_folder, "results.jld2"), Dict(
        "nuts_samples" => nuts_samples,
        "experiment_config" => experiment_config
    ))

    with_logger(logger) do
        @info "MCMC samples:" nuts_samples
    end
end

post_pred_mh_baseline(PostPredMHBaseline(
    experiment_name="mh_baseline",
    num_samples=10000,
    num_chains=28,
    seed=1390
))