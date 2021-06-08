using DrWatson
@quickactivate "EPExperiments"

using EPExperiments
using Logging
using LoggingExtras
using JLD2
using FileIO
using Random
using AdvancedHMC
using CSV
using DataFrames
using StatsPlots
using ArgParse

import StatsFuns: logsumexp

const LOG_FILENAME = "out.log"

@with_kw struct RadonEstimateConfig{T<:AnISConfig}
    experiment_name::String = "test"
    num_samples::Int = 10
    num_betas::Int = 101
    anis_params::T = AnISHMCConfig{Turing.ReverseDiffAD{false}}(
        proposal=AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(0.05), 10),
        num_steps=10,
        rejection_type=SimpleRejection()
    )
    seed::Int = 1234
    data_file::String = "radon.csv"
    radon_model::Expectation = radon_hierarchical
    cost_fn_name::Symbol = :radon_target_f
end

# Some old experimental results didn't include the cost_fn_name 
# as a parameter in the config. 
function old2new_config(old_config, cost_fn_name)
    return RadonEstimateConfig(
        experiment_name=old_config.experiment_name, 
        num_samples=old_config.num_samples, 
        anis_params=old_config.anis_params, 
        num_betas=old_config.num_betas, 
        data_file=old_config.data_file, 
        radon_model=old_config.radon_model, 
        seed=old_config.seed, 
        cost_fn_name=cost_fn_name
    )
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--seed", "-s"
            help = "Random seed for the program."
            arg_type = Int
            required = true
    end

    return parse_args(s)
end

function load_radon_results(
    experiment_dirname
)
    result_folder = datadir("radon_estimate", experiment_dirname)
    results = load(joinpath(result_folder, "results.jld2"))

    cost_estimate = results["cost_estimate"]
    radon_tabi = results["radon_tabi"]
    experiment_config = results["experiment_config"]
    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

    return cost_estimate, radon_tabi, experiment_config, result_folder, logger
end

function reprocess_radon_results(
    experiment_dirname;
    old_experiment_config=false,
    cost_fn_name=nothing
)
    cost_estimate, radon_tabi, experiment_config, result_folder, logger = load_radon_results(
        experiment_dirname
    )

    if old_experiment_config
        @assert !isnothing(cost_fn_name)
        experiment_config = old2new_config(experiment_config, cost_fn_name)
    end

    process_radon_results(
        cost_estimate,
        radon_tabi,
        experiment_config,
        result_folder,
        logger
    )
end

function process_radon_results(
    cost_estimate, 
    radon_tabi, 
    experiment_config,
    result_folder,
    logger
)
    @unpack num_betas = experiment_config
    betas = (geomspace(1, 1001, num_betas) .- 1) ./ 1000

    @unpack cost_fn_name = experiment_config
    cost_fn = RADON_COST_FNS[cost_fn_name]

    @show size(radon_tabi[:Z2_info].info[:intermediate_log_weights])
    @show size(betas)
    savefig(plot_intermediate_weights(
        radon_tabi[:Z2_info].info[:intermediate_log_weights], betas),
        joinpath(result_folder, "Z2_intermediate_weights.png")
    )
    savefig(plot_intermediate_weights(
        radon_tabi[:Z1_positive_info].info[:intermediate_log_weights], betas),
        joinpath(result_folder, "Z1_intermediate_weights.png")
    )
    savefig(plot_intermediate_ess(
        radon_tabi[:Z2_info].info[:intermediate_log_weights], betas),
        joinpath(result_folder, "Z2_intermediate_ess.png")
    )
    savefig(plot_intermediate_ess(
        radon_tabi[:Z1_positive_info].info[:intermediate_log_weights], betas),
        joinpath(result_folder, "Z1_intermediate_ess.png")
    )
    params_to_plot = [:mu_alpha, :mu_beta]
    savefig(
        plot(radon_tabi[:Z2_info][:,params_to_plot,:]), 
        joinpath(result_folder, "Z2_samples.png")
    )
    savefig(
        plot(radon_tabi[:Z1_positive_info][:,params_to_plot,:]), 
        joinpath(result_folder, "Z1_samples.png")
    )

    alpha_names = namesingroup(radon_tabi[:Z1_positive_info], :alpha)
    savefig(
        plot(radon_tabi[:Z1_positive_info][:,alpha_names,:], colordim=:parameter),
        joinpath(result_folder, "Z1_counties.png")
    )
    savefig(
        plot(radon_tabi[:Z1_positive_info][:,[alpha_names[10]],:]),
        joinpath(result_folder, "Z1_county10.png")
    )
    savefig(
        plot(radon_tabi[:Z2_info][:,alpha_names,:], colordim=:parameter),
        joinpath(result_folder, "Z2_counties.png")
    )
    savefig(
        plot(radon_tabi[:Z2_info][:,[alpha_names[10]],:]),
        joinpath(result_folder, "Z2_county10.png")
    )

    Z2_chn = radon_tabi[:Z2_info]
    alpha = Array(Z2_chn[namesingroup(Z2_chn, :alpha)])
    beta = Array(Z2_chn[namesingroup(Z2_chn, :beta)])

    eps = 1
    radon_losses = map(1:size(alpha, 1)) do (i)
        cost_fn(
            alpha[i,:], 
            beta[i,:], 
            POST_PRED_COUNTY_IDX,
            POST_PRED_FLOOR
        )
    end

    Z2_lws = Array(Z2_chn[:log_weight])
    normalisation = exp(logsumexp(Z2_lws))
    anis_estimate = sum(exp.(Z2_lws) .* radon_losses) / normalisation

    Z1_lws = Array(radon_tabi[:Z1_positive_info][:log_weight])
    Z1_lws_retargeted = Z1_lws .- log.(radon_losses)
    Z2_lws_retargeted = Z2_lws .+ log.(radon_losses)

    ESS_retargeted = (
        Z1_positive_info = compute_ess(Z1_lws_retargeted),
        Z2_info = compute_ess(Z2_lws_retargeted)
    )

    # NOTE: Usually we have to divide each estimate by the number of samples used 
    # but because those "normalisations" cancel in our downstream calculations we 
    # avoid them.
    Z1_ws = exp.(Z1_lws)
    Z2_ws = exp.(Z2_lws)
    Z1s_unnormalised = cumsum(Z1_ws, dims=1)
    Z2s_unnormalised = cumsum(Z2_ws, dims=1)
    tabi_ests = Z1s_unnormalised ./ Z2s_unnormalised
    anis_ests = cumsum(vec(Z2_ws) .* radon_losses) ./ Z2s_unnormalised

    num_samples = size(tabi_ests, 1)
    ixs = 1:num_samples
    tabi_ixs = 2 * ixs
    convergence_plot = plot(tabi_ixs, tabi_ests[ixs], label="TAAnIS")
    plot!(
        convergence_plot,
        ixs, 
        anis_ests[ixs],
        label="AnIS"
    )
    savefig(convergence_plot, joinpath(result_folder, "convergence_plot.png"))

    ess_plot = make_ess_plot(Z1_lws, Z2_lws, Z2_lws_retargeted)
    savefig(ess_plot, joinpath(result_folder, "ess_plot.png"))

    with_logger(logger) do
        @info "Expectation estimate TABI: $(cost_estimate)"
        @info "Expectation estimate AnIS: $(anis_estimate)"

        for (k, est_results) in pairs(radon_tabi)
            if isnothing(est_results)
                continue
            end
            msg = string([
                "$(string(k)):\n", 
                "ESS: $(est_results.info[:ess])\n",
                "ESS retargeted: $(ESS_retargeted[k])\n",
                "Log evidence: $(est_results.logevidence)\n",
                "Log weights: $(get(est_results, :log_weight)[:log_weight])\n"
            ]...)

            @info "$(msg)" 
        end
    end
end

function radon_estimate(experiment_config)
    @unpack seed, anis_params, num_betas, num_samples = experiment_config
    @unpack radon_model, cost_fn_name, data_file = experiment_config
    Random.seed!(seed)

    result_folder = mkpath(joinpath(
        datadir("radon_estimate"),
        savename(experiment_config)
    ))

    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

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

    # Default:
    betas = (geomspace(1, 1001, num_betas) .- 1) ./ 1000
    # 200_400 ratio:
    # low_beta = 3 * 1e-3
    # low_betas = ((geomspace(1, 1001, 200) .- 1) ./ 1000) .* low_beta
    # high_betas = ((geomspace(1, 1001, 202) .- 1) ./ 1000) .* (1 - low_beta) .+ low_beta
    # betas = vcat(low_betas, high_betas[2:end])
    anis_alg = get_anis_alg(anis_params, betas)
    tabi = TABI(
        TuringAlgorithm(anis_alg, num_samples),
        TuringAlgorithm(anis_alg, 0),
        TuringAlgorithm(anis_alg, num_samples)
    )

    @time begin
    cost_estimate, radon_tabi = estimate_expectation(
        radon_exp, 
        tabi; 
        store_intermediate_samples=true,
        progress=true
    )
    end

    @tagsave(joinpath(result_folder, "results.jld2"), Dict(
        "cost_estimate" => cost_estimate,
        "radon_tabi" => radon_tabi,
        "experiment_config" => experiment_config
    ))

    process_radon_results(
        cost_estimate, 
        radon_tabi, 
        experiment_config,
        result_folder,
        logger
    )
end

radon_estimate(RadonEstimateConfig(
    experiment_name="eps_super_simple_model",
    num_samples=1000,
    anis_params=AnISHMCConfig{Turing.ReverseDiffAD{false}}(
        proposal=AdvancedHMC.NUTS{MultinomialTS,GeneralisedNoUTurn}(
            AdvancedHMC.Leapfrog(0.044)
        ),
        num_steps=1,
        rejection_type=SimpleRejection()
    ),
    num_betas=201,
    data_file="small_radon_num_counties=20.csv",
    radon_model=radon_hierarchical,
    seed=600,
    cost_fn_name=:radon_target_f
))
