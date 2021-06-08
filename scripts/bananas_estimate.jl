using DrWatson
@quickactivate "EPExperiment"

using EPExperiments

using Random
using Logging
using LoggingExtras
using JLD2
using FileIO
using CSV
using DataFrames
using LinearAlgebra: I
using StatsFuns: logsumexp
using StatsPlots

const LOG_DIR_NAME = "bananas_estimate"
const LOG_FILENAME = "out.log"

@with_kw struct BananasEstimateConfig{T<:AnISConfig}
    experiment_name::String = "test"
    num_replications::Int = 1
    num_samples::Int = 10
    num_betas::Int = 101
    anis_params::T = AnISMHConfig(
        transition_kernel_cfg=MHFunction(
            kernel_fn=kernel_fn
        )
    )
    seed::Int = 1234
end

function banana_reprocess_results(experiment_name)
    result_folder = datadir(LOG_DIR_NAME, experiment_name)
    results = load(joinpath(result_folder, "results.jld2"))

    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

    banana_process_results(
        results["exp_estimate"],
        results["tabi_results"],
        results["experiment_config"],
        result_folder,
        logger
    )
end

function banana_process_results(
    exp_estimate, 
    tabi_results, 
    experiment_config,
    result_folder,
    logger
)
    Z1p_chns = tabi_results[:Z1_positive_info]
    Z1n_chns = tabi_results[:Z1_negative_info]
    Z2_chns = tabi_results[:Z2_info]

    gt_file = load(datadir("bananas_ground_truth", "ground_truth.jld2"))
    ground_truth = gt_file["expectation_estimate"]
    @show ground_truth

    ep, _ = convergence_plot(
        "banana",
        Z1p_chns,
        Z2_chns,
        banana_f,
        ground_truth;
        Z1_negative_chn=Z1n_chns,
        ylims=(10^(-6.5),5),
        yticks=[1, 0.01, 0.0001, 0.000001],
        num_xlogspaced=100,
        legend=:bottomleft
    )
    savefig(ep, joinpath(result_folder, "error_plot.pdf"))

    for (name, chn) in pairs(tabi_results)
        with_logger(logger) do
            @info "$name"
            lws = Array(chn[:log_weight])
            for chain_ix in 1:size(lws, 2)
                ess = compute_ess(lws[:,chain_ix])
                @info "Chain $chain_ix: $ess"
            end
        end
    end
end

function bananas_estimate(experiment_config)
    @unpack seed, num_samples, num_replications = experiment_config
    @unpack num_betas, anis_params = experiment_config

    result_folder = mkpath(joinpath(
        datadir(LOG_DIR_NAME),
        savename(experiment_config)
    ))
    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

    Random.seed!(seed)

    bananas_exp = banana()

    # Make AnIS algorithm
    betas = (geomspace(1, 1001, num_betas) .- 1) ./ 1000
    anis_alg = get_anis_alg(anis_params, betas)
    tabi = TABI(
        TuringAlgorithm(anis_alg, num_samples),
        TuringAlgorithm(anis_alg, num_samples),
        TuringAlgorithm(anis_alg, num_samples)
    )
    
    # Replicate results N times
    Z1p_chns = Chains[]
    Z1n_chns = Chains[]
    Z2_chns = Chains[]
    exp_ests = []
    for n in 1:num_replications
        println("Run $n:")
        @time begin
        exp_estimate, tabi_results = estimate_expectation(
            bananas_exp,
            tabi;
            store_intermediate_samples=true
        )
        end
        push!(Z1p_chns, tabi_results[:Z1_positive_info])
        push!(Z1n_chns, tabi_results[:Z1_negative_info])
        push!(Z2_chns, tabi_results[:Z2_info])
        push!(exp_ests, exp_estimate)
    end

    Z1p_chns = chainscat(Z1p_chns...)
    Z1n_chns = chainscat(Z1n_chns...)
    Z2_chns = chainscat(Z2_chns...)
    tabi_results = (
        Z1_positive_info = Z1p_chns,
        Z1_negative_info = Z1n_chns,
        Z2_info = Z2_chns
    )
    exp_estimate = mean(exp_ests)

    # Save results and experiment config
    @tagsave(joinpath(result_folder, "results.jld2"), Dict(
        "exp_estimate" => exp_estimate,
        "tabi_results" => tabi_results,
        "experiment_config" => experiment_config
    ))

    banana_process_results(
        exp_estimate, 
        tabi_results,
        experiment_config,
        result_folder,
        logger
    )
end

kernel_fn(prior_sample::T, i) where {T<:Real} = Normal(0, 1)
kernel_fn(prior_sample::NamedTuple, i) = map(x -> kernel_fn(x, i), prior_sample)

bananas_estimate(BananasEstimateConfig(
    experiment_name="banana",
    num_samples=1000,
    num_betas=201,
    num_replications=5,
    anis_params=AnISMHConfig(
        transition_kernel_cfg=MHFunction(
            kernel_fn=kernel_fn
        )
    )
))