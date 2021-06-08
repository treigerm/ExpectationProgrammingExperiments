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

const LOG_DIR_NAME = "sir_nuts_baseline"
const LOG_FILENAME = "out.log"

@with_kw struct SIRNUTSBaseline
    experiment_name::String = "test"
    num_samples::Int = 100
    num_chains::Int = 4
    nuts_acceptance_ratio::Float64 = 0.65
    seed::Int = 1234
    data_file::String = "neg_binomial_data.jld2"
    cost_fn_name::Symbol = :cost_fn_sir
end

function sir_nuts_baseline(experiment_config)
    @unpack seed, num_samples, nuts_acceptance_ratio, num_chains = experiment_config
    @unpack cost_fn_name, data_file = experiment_config

    result_folder = mkpath(joinpath(
        datadir(LOG_DIR_NAME),
        savename(experiment_config)
    ))

    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

    Random.seed!(seed)

    input_data = load(datadir("sims", data_file))
    @unpack tmax, total_population, i_0_true, β_true, γ = input_data["params"]
    X, Y = input_data["X"], input_data["Y"]

    sir_exp = bayes_sir(
        Y, 
        total_population, 
        γ, 
        COST_FUNCTIONS[cost_fn_name]
    )
    sir_turing = sir_exp.gamma2

    @time begin
    nuts_samples = sample(
        sir_turing, 
        Turing.NUTS(nuts_acceptance_ratio), 
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

sir_nuts_baseline(SIRNUTSBaseline(
    experiment_name="nuts_baseline",
    num_samples=10000,
    num_chains=16,
    seed=1240
))