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

const LOG_DIR_NAME = "radon_nuts_baseline"
const LOG_FILENAME = "out.log"

@with_kw struct RadonNUTSBaseline
    experiment_name::String = "test"
    num_samples::Int = 100
    num_chains::Int = 4
    nuts_acceptance_ratio::Float64 = 0.65
    seed::Int = 1234
    data_file::String = "radon.csv"
    radon_model::Expectation = radon_unbounded
    cost_fn_name::Symbol = :radon_hierarchical
end

function radon_nuts_baseline(experiment_config)
    @unpack seed, num_samples, nuts_acceptance_ratio, num_chains = experiment_config
    @unpack radon_model, cost_fn_name, data_file = experiment_config

    result_folder = mkpath(joinpath(
        datadir(LOG_DIR_NAME),
        savename(experiment_config)
    ))

    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

    Random.seed!(seed)

    radon_df = DataFrame(CSV.File(datadir("exp_raw", data_file)))
    log_radon, county_idx, floor, num_counties = preprocess_radon_df(
        radon_df, data_file
    )

    radon_exp = radon_model(
        log_radon, 
        county_idx, 
        floor, 
        num_counties; 
        cost_fn=RADON_COST_FNS[cost_fn_name]
    )
    radon_turing = radon_exp.gamma2

    @time begin
    nuts_samples = sample(
        radon_turing, 
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

radon_nuts_baseline(RadonNUTSBaseline(
    experiment_name="collect_baseline_samples",
    num_samples=10000,
    num_chains=16,
    radon_model=radon_hierarchical,
    seed=200
))
