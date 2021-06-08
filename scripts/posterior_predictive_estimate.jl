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

const LOG_DIR_NAME = "posterior_predictive_estimate"
const LOG_FILENAME = "out.log"

@with_kw struct PostPredEstimateConfig{T<:AnISConfig}
    experiment_name::String = "test"
    num_replications::Int = 1
    num_samples::Int = 10
    num_betas::Int = 101
    anis_params::T = AnISHMCConfig{Turing.ForwardDiffAD{10}}(
        proposal=AdvancedHMC.NUTS{MultinomialTS,GeneralisedNoUTurn}(
            AdvancedHMC.Leapfrog(0.5)
        ),
        num_steps=1,
        rejection_type=SimpleRejection()
    )
    seed::Int = 1234
end

function reprocess_pp_results(experiment_name; mcmc_baseline_name=nothing)
    result_folder = datadir(LOG_DIR_NAME, experiment_name)
    results = load(joinpath(result_folder, "results.jld2"))

    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

    process_post_pred_results(
        results["exp_estimate"],
        results["tabi_results"],
        results["experiment_config"],
        result_folder,
        logger;
        mcmc_baseline_name=mcmc_baseline_name
    )
end

function process_post_pred_results(
    exp_estimate, 
    tabi_results, 
    experiment_config,
    result_folder,
    logger;
    mcmc_baseline_name=nothing
)
    Z2_chn = tabi_results[:Z2_info]
    num_chains = size(Z2_chn, 3)
    num_samples = size(Z2_chn, 1)
    Z2_xs = Z2_chn[namesingroup(Z2_chn,:x)].value.data
    Z2_fxs = mapslices(Z2_xs; dims=[2]) do (x)
        pdf(MvNormal(-POST_PRED_Y_OBSERVED, 0.5*I), x)
    end
    Z2_fxs = reshape(Z2_fxs, (num_samples, num_chains))

    Z1_chn = tabi_results[:Z1_positive_info]
    Z1_xs = Z1_chn[namesingroup(Z1_chn,:x)].value.data
    Z1_fxs = mapslices(Z1_xs; dims=[2]) do (x)
        pdf(MvNormal(-POST_PRED_Y_OBSERVED, 0.5*I), x)
    end
    Z1_fxs = reshape(Z1_fxs, (num_samples, num_chains))

    Z2_lws = Array(Z2_chn[:log_weight])
    normalisation = exp(logsumexp(Z2_lws))
    anis_estimate = sum(exp.(Z2_lws) .* Z2_fxs) / normalisation

    Z1_lws = Array(Z1_chn[:log_weight])
    Z1_lws_retargeted = Z1_lws .- log.(Z1_fxs)
    Z2_lws_retargeted = Z2_lws .+ log.(Z2_fxs)

    ESS_retargeted = (
        Z1_positive_info = compute_ess(Z1_lws_retargeted),
        Z2_info = compute_ess(Z2_lws_retargeted)
    )

    num_xlogspaced = 100
    ep, _ = convergence_plot(
        "post_pred",
        Z1_chn, 
        Z2_chn, 
        x -> pdf(MvNormal(-POST_PRED_Y_OBSERVED, 0.5*I), x),
        POST_PRED_GROUND_TRUTH;
        num_xlogspaced=num_xlogspaced
    )
    savefig(ep, joinpath(result_folder, "error_plot.png"))

    ess_plot = make_ess_plot(
        Z1_lws, 
        Z2_lws, 
        Z2_lws_retargeted;
        take_min=false,
        legend=:topleft,
        num_xlogspaced=num_xlogspaced)
    savefig(ess_plot, joinpath(result_folder, "ess_plot.pdf"))

    if !isnothing(mcmc_baseline_name)
        # RSE plot
        mcmc_results = load(datadir(
            "posterior_predictive_mh_baseline",
            mcmc_baseline_name,
            "ess.jld2"
        ))
        mcmc_estimates = mcmc_results["exp_estimates"]
        fake_ess = mcmc_results["fake_ess"]

        mcmc_rse = relative_squared_error.(
            POST_PRED_GROUND_TRUTH, mcmc_estimates
        )

        mds, qs = get_median_and_quantiles(mcmc_rse)

        num_samples = size(Z2_lws, 1)
        if !isnothing(num_xlogspaced)
            log_ixs = unique(round.(Int, geomspace(1, num_samples, num_xlogspaced)))
            @show 450*log_ixs
            mds = mds[450*log_ixs]
            qs = qs[:,450*log_ixs]
            plot_ixs = log_ixs
        else
            ixs = Array(0:4500:size(mcmc_rse, 1))
            ixs[1] = 90
            mds = mds[ixs]
            qs = qs[:,ixs]

            plot_ixs = Array(0:50:10_000)
            plot_ixs[1] = 1
        end
        plot!(
            ep,
            plot_ixs,
            mds,
            ribbon=[(mds.-qs[1,:]), (qs[2,:]-mds)],
            label="MCMC"
        )
        savefig(ep, joinpath(result_folder, "error_plot_mcmc.pdf"))
    end

    with_logger(logger) do
        @info "Expectation estimate TABI: $(exp_estimate)"
        @info "Expectation estimate AnIS: $(anis_estimate)"

        for (k, est_results) in pairs(tabi_results)
            if isnothing(est_results)
                continue
            end
            msg = string([
                "$(string(k)):\n", 
                "ESS: $(est_results.info[:ess])\n",
                "ESS retargeted: $(ESS_retargeted[k])\n",
                "Log evidence: $(est_results.logevidence)\n",
            ]...)

            @info "$(msg)" 
        end
    end
end

function post_pred_estimate(experiment_config)
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

    post_pred_exp = post_pred(POST_PRED_Y_OBSERVED)

    # Make AnIS algorithm
    betas = Array(range(0, 1; length=num_betas))
    anis_alg = get_anis_alg(anis_params, betas)
    tabi = TABI(
        TuringAlgorithm(anis_alg, num_samples),
        TuringAlgorithm(anis_alg, 0),
        TuringAlgorithm(anis_alg, num_samples)
    )
    
    # Replicate results N times
    Z1_chns = Chains[]
    Z2_chns = Chains[]
    exp_ests = []
    for n in 1:num_replications
        println("Run $n:")
        @time begin
        exp_estimate, tabi_results = estimate_expectation(
            post_pred_exp,
            tabi;
            store_intermediate_samples=true
        )
        end
        push!(Z1_chns, tabi_results[:Z1_positive_info])
        push!(Z2_chns, tabi_results[:Z2_info])
        push!(exp_ests, exp_estimate)
    end

    Z1_chns = chainscat(Z1_chns...)
    Z2_chns = chainscat(Z2_chns...)
    tabi_results = (
        Z1_positive_info = Z1_chns,
        Z2_info = Z2_chns
    )
    exp_estimate = mean(exp_ests)

    # Save results and experiment config
    @tagsave(joinpath(result_folder, "results.jld2"), Dict(
        "exp_estimate" => exp_estimate,
        "tabi_results" => tabi_results,
        "experiment_config" => experiment_config
    ))

    # Process results
    process_post_pred_results(
        exp_estimate, tabi_results, experiment_config, result_folder, logger
    )

    @show POST_PRED_Y_OBSERVED
    @show pdf(
        MvNormal(POST_PRED_Y_OBSERVED / 2, I), -POST_PRED_Y_OBSERVED
    )
end

# This sets the transition kernel for AnIS.
function kernel_fn(prior_sample::AbstractArray, i)
    return MvNormal(size(prior_sample, 1), 0.5) 
end
kernel_fn(prior_sample::NamedTuple, i) = map(x -> kernel_fn(x, i), prior_sample)

post_pred_estimate(PostPredEstimateConfig(
    experiment_name="collect_samples",
    num_samples=1_000,
    num_betas=101,
    num_replications=5,
    anis_params=AnISMHConfig(
        transition_kernel_cfg=MHFunction(
            kernel_fn=kernel_fn
        )
    )
))