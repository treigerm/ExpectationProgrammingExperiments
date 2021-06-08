using DrWatson
@quickactivate "EPExperiment"

using EPExperiments

using FileIO
using JLD2
using StatsPlots
using AnnealedIS
using StatsFuns: logsumexp

function AnnealedIS.effective_sample_size(log_weights)
    denominator = logsumexp(2 * log_weights)
    numerator = 2 * logsumexp(log_weights)
    return exp(numerator - denominator)
end

function post_hoc_analysis(
    data_filename,
    is_ground_truth_name, 
    taanis_name;
    mcmc_baseline_name=nothing,
    make_joint_plot=false
)
    is_ground_truth_dir = datadir("is_ground_truth", is_ground_truth_name)
    taanis_dir = datadir("inferences", taanis_name)
    input_data = load(datadir("sims", data_filename))

    true_gamma = input_data["params"].γ

    is_results = load(joinpath(is_ground_truth_dir, "results.jld2"))
    taanis_results = load(joinpath(taanis_dir, "results.jld2"))

    # experiment_config = taanis_results["experiment_config"]
    # @unpack cost_fn_name = experiment_config
    cost_fn_name = :cost_fn_sir
    @show cost_fn_name
    cost_fn = COST_FUNCTIONS[cost_fn_name]

    true_expectation_value = is_results["expectation_estimate"]

    num_xlogspaced = 100
    ep, cp = convergence_plot(
        "sir",
        taanis_results["ode_tabi"][:Z1_positive_info],
        taanis_results["ode_tabi"][:Z2_info],
        (i, b) -> cost_fn(b, true_gamma),
        true_expectation_value;
        thinning=1,
        num_xlogspaced=num_xlogspaced
    )
    savefig(ep, joinpath(taanis_dir, "test_error_plot_log.png"))
    #savefig(cp, joinpath(taanis_dir, "convergence_plot.png"))
    # ep, cp = convergence_plot(
    #     "sir",
    #     taanis_results["ode_tabi"][:Z1_positive_info],
    #     taanis_results["ode_tabi"][:Z2_info],
    #     (i, b) -> cost_fn(b, true_gamma),
    #     true_expectation_value;
    #     thinning=1,
    #     num_xlogspaced=num_xlogspaced,
    #     mean_and_std=false
    # )
    # savefig(ep, joinpath(taanis_dir, "error_plot_means.png"))

    Z1_lws = Array(taanis_results["ode_tabi"][:Z1_positive_info][:log_weight])
    Z1_betas = Array(taanis_results["ode_tabi"][:Z1_positive_info][:β])

    Z2_lws = Array(taanis_results["ode_tabi"][:Z2_info][:log_weight])
    Z2_betas = Array(taanis_results["ode_tabi"][:Z2_info][:β])

    # Z1_ess = effective_sample_size(Z1_lws)
    # Z2_ess = effective_sample_size(Z2_lws)

    Z2_lws_retargeted = Z2_lws .+ log.(cost_fn.(Z2_betas, true_gamma))
    Z1_lws_retargeted = Z1_lws .- log.(cost_fn.(Z1_betas, true_gamma))

    # Z2_ess_retargeted = effective_sample_size(Z2_lws_retargeted)
    # Z1_ess_retargeted = effective_sample_size(Z1_lws_retargeted)

    ess_plot = make_ess_plot(
        Z1_lws, 
        Z2_lws, 
        Z2_lws_retargeted; 
        yaxisscale=:log10,
        take_min=false,
        legend=false,
        num_xlogspaced=num_xlogspaced
    )
    savefig(ess_plot, joinpath(taanis_dir, "sir_ess_not_min.pdf"))

    if !isnothing(mcmc_baseline_name)
        mcmc_results = load(datadir(
            "sir_nuts_baseline", 
            mcmc_baseline_name, 
            "ess.jld2"
        ))
        mcmc_estimates = mcmc_results["exp_estimates"]
        num_intermediates = mcmc_results["num_intermediates"]
        fake_ess = mcmc_results["fake_ess"]
        @show mcmc_estimates[end]
        @show true_expectation_value
        @show relative_squared_error(
            true_expectation_value, mcmc_estimates[end]
        )
        mcmc_rse = relative_squared_error.(
            true_expectation_value, mcmc_estimates
        )

        @show size(mcmc_estimates)
        @show mcmc_rse[end,:]
        @show mcmc_estimates[end,:]

        mds, qs = get_median_and_quantiles(mcmc_rse)

        thinning = 100
        num_samples = size(Z2_lws, 1)
        # TODO: Remove magic numbers
        # ixs = Array(0:4500:size(mcmc_rse, 1))
        # ixs[1] = 90
        # @show size(mcmc_rse[ixs])
        # plot_ixs = Array(0:50:num_samples)
        # plot_ixs[1] = 1
        # @show num_samples
        # @show size(plot_ixs)
        if !isnothing(num_xlogspaced)
            log_ixs = unique(round.(Int, geomspace(1, num_samples, num_xlogspaced)))
            plot_ixs = log_ixs
            @show 90*log_ixs
            mds = mds[90*log_ixs]
            qs = qs[:,90*log_ixs]
        else
            ixs = Array(0:4500:size(mcmc_rse, 1))
            ixs[1] = 90
            @show size(mcmc_rse[ixs])
            plot_ixs = Array(0:50:num_samples)
            plot_ixs[1] = 1
            mds = mds[ixs]
            qs = qs[ixs]
        end
        plot!(
            ep,
            plot_ixs,
            mds,
            ribbon=[(mds.-qs[1,:]), (qs[2,:]-mds)],
            label="MCMC",
            legend=:bottomleft
        )
        savefig(ep, joinpath(taanis_dir, "sir_error_plot_mcmc_log.pdf"))

        # plot!(
        #     ess_plot,
        #     1:num_samples,
        #     fake_ess,
        #     label="MCMC"
        # )
        # savefig(ess_plot, joinpath(taanis_dir, "sir_ess_test_mcmc.png"))
    end

    if make_joint_plot
        @unpack i_0_true, β_true = input_data["params"]
        joint_plot = plot_sir_samples(
            taanis_results["ode_tabi"][:Z2_info],
            color=palette(:default)[2]
        )
        savefig(joint_plot, joinpath(taanis_dir, "anis_Z2_joint_samples.png"))

        joint_plot = plot_sir_samples(
            taanis_results["ode_tabi"][:Z1_positive_info],
            color=palette(:default)[1]
        )
        savefig(joint_plot, joinpath(taanis_dir, "Z1_joint_samples.png"))
    end

    return
    num_samples = length(Z2_lws)
    # TODO: Is the logevidence field in the Chains object still correct if we merged several chains?
    Z2_weight_normalisation = taanis_results["ode_tabi"][:Z2_info].logevidence + log(num_samples)
    Z1_weight_normalisation = taanis_results["ode_tabi"][:Z1_positive_info].logevidence + log(num_samples)
    savefig(
        plot_is_samples(Z2_betas, Z2_lws .- Z2_weight_normalisation), 
        joinpath(taanis_dir, "Z2_betas.png")
    )
    savefig(
        plot_is_samples(Z1_betas, Z1_lws .- Z1_weight_normalisation), 
        joinpath(taanis_dir, "Z1_betas.png")
    )
end

post_hoc_analysis(
    "neg_binomial_data.jld2",
    GROUND_TRUTH_FOLDER, # TODO: Give name of folder with ground truth.
    RESULTS_FOLDER; # TODO: Give name of folder which contains (TA)AnIS results.
    # mcmc_baseline_name="combined_baselines_5replications",
    make_joint_plot=true
)