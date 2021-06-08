using DrWatson
@quickactivate "EPExperiments"

using EPExperiments
using Statistics
using StatsPlots
using Printf

function get_mean_std(xs)
    return mean(xs), std(xs)
end

function radon_combined_post_hoc_plots(
    experiment_dirname;
    mcmc_baseline_name=nothing
)
    # Load Z1 and Z2 chains
    result_folder = datadir("radon_estimate", experiment_dirname)
    results = load(joinpath(result_folder, "results.jld2"))

    experiment_config = results["experiment_config"]
    @unpack cost_fn_name = experiment_config
    cost_fn = RADON_COST_FNS[cost_fn_name]

    Z1_chn = results["radon_tabi"][:Z1_positive_info]
    Z2_chn = results["radon_tabi"][:Z2_info]

    # Evaluate target function
    # size: (num_samples, num_params, num_chains)
    # If we just use Array(chn) then the different chains get appended and 
    # we "lose" the third dimension. Using chn.value.data preserves the three 
    # dimensional shape.
    alpha = Z2_chn[:,namesingroup(Z2_chn, :alpha),:].value.data
    beta = Z2_chn[:,namesingroup(Z2_chn, :beta),:].value.data
    Z2_radon_losses = Array{Float64}(undef, size(alpha, 1), size(alpha, 3))
    fill!(Z2_radon_losses, -Inf)
    for sample_ix in 1:size(Z2_radon_losses, 1)
        for chain_ix in 1:size(Z2_radon_losses, 2)
            Z2_radon_losses[sample_ix,chain_ix] = cost_fn(
                alpha[sample_ix,:,chain_ix],
                beta[sample_ix,:,chain_ix],
                POST_PRED_COUNTY_IDX,
                POST_PRED_FLOOR
            )
        end
    end

    @show sum(Z2_radon_losses .== -Inf)

    # Load log weights and compute retargeted weights
    Z2_lws = Array(Z2_chn[:log_weight])
    normalisation = exp.(logsumexp(Z2_lws; dims=1))
    anis_estimate = sum(exp.(Z2_lws) .* Z2_radon_losses; dims=1) ./ normalisation

    Z1_lws = Array(Z1_chn[:log_weight])
    Z1_vals = exp.(logsumexp(Z1_lws; dims=1))
    Z2_lws_retargeted = Z2_lws .+ log.(Z2_radon_losses)

    taanis_estimate = Z1_vals ./ normalisation
    @show taanis_estimate
    @show anis_estimate
    
    # Make ESS plot
    num_xlogspaced = 100
    # ess_plot = make_ess_plot(
    #     Z1_lws, 
    #     Z2_lws, 
    #     Z2_lws_retargeted; 
    #     yaxisscale=:log10,
    #     num_xlogspaced=num_xlogspaced,
    #     legend=:topleft
    # )
    # savefig(ess_plot, joinpath(result_folder, "ess_plot_log_scale.pdf"))
    ess_plot = make_ess_plot(
        Z1_lws, 
        Z2_lws, 
        Z2_lws_retargeted; 
        yaxisscale=:log10,
        num_xlogspaced=num_xlogspaced,
        legend=false,
        take_min=false
    )
    savefig(ess_plot, joinpath(result_folder, "radon_ess_not_min.pdf"))

    if !isnothing(mcmc_baseline_name)
        mcmc_results = load(datadir(
            "radon_nuts_baseline", 
            mcmc_baseline_name, 
            "ess.jld2"
        ))
        mcmc_estimates = mcmc_results["exp_estimates"]
        num_intermediates = mcmc_results["num_intermediates"]
        fake_ess = mcmc_results["fake_ess"]

        thinning = 200
        num_samples = size(fake_ess, 1) * thinning
        @show size(fake_ess)
        @show size(mcmc_estimates)
        @show mcmc_estimates[end,:]
        taanis_m, taanis_std = get_mean_std(taanis_estimate)
        anis_m, anis_std = get_mean_std(anis_estimate)
        mcmc_m, mcmc_std = get_mean_std(mcmc_estimates[end,:])
        p = scatter([1], [taanis_m], yerr=[taanis_std], label="TAAnIS", fontfamily="serif-roman")
        scatter!(p, [2], [anis_m], yerr=[anis_std], label="AnIS")
        scatter!(p, [3], [mcmc_m], yerr=[mcmc_std], label="MCMC")
        savefig(p, joinpath(result_folder, "final_estimates.png"))
        @printf("TAAnIS: %.2e \$\\pm\$ %.2e\n", taanis_m, taanis_std)
        @printf("AnIS: %.2e \$\\pm\$ %.2e\n", anis_m, anis_std)
        @printf("MCMC: %.2e \$\\pm\$ %.2e\n", mcmc_m, mcmc_std)
        return
        ixs = Array(1:thinning:num_samples) ./ num_intermediates
        ixs[1] = 1
        plot!(
            ess_plot,
            ixs,
            fake_ess,
            label="MCMC"
        )
        savefig(ess_plot, joinpath(result_folder, "ess_plot_mcmc.png"))
    end
end

radon_combined_post_hoc_plots(
    "combined_logistic_all_below_eps_super_simple_model_10_replications";
    # mcmc_baseline_name="combined_baselines_10replications"
)