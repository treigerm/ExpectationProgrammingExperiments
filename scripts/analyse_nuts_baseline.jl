using DrWatson
@quickactivate "EPExperiment"

using EPExperiments

using FileIO
using JLD2
using LaTeXStrings
using StatsPlots
using StatsFuns: logsumexp

function get_sir_losses(chn, cost_fn)
    betas = chn[:β].value.data
    return cost_fn.(betas, )
end

function analyse_nuts_baseline(
    subfolder, 
    experiment_name; 
    thinning=1,
    burn_in=1001,
    num_replications=1
)
    # Load chain and experiment config
    experiment_folder = datadir(subfolder, experiment_name)
    results = load(joinpath(experiment_folder, "results.jld2"))
    experiment_config = results["experiment_config"]
    chn = results["nuts_samples"]

    # Get cost function
    if subfolder == "sir_nuts_baseline"
        # Indicates the number of intermediate distributions in AnIS. This is 
        # important for adjusting the x-axis in the ESS plot.
        num_intermediates = 100

        @unpack cost_fn_name = experiment_config
        cost_fn = COST_FUNCTIONS[cost_fn_name]

        @unpack data_file = experiment_config
        input_data = load(datadir("sims", data_file))
        true_gamma = input_data["params"].γ
        @unpack i_0_true, β_true = input_data["params"]

        betas = Array(chn[:β])[burn_in:end,:]
        losses = cost_fn.(betas, true_gamma)

        # Plot the samples from the joint
        i_0s = Array(chn[:i₀])[burn_in:end,:]
        p = scatter(
            Array(chn[:β])[1:(burn_in-1),:], 
            Array(chn[:i₀])[1:(burn_in-1),:], 
            xlabel=L"\beta", 
            ylabel=L"i_0",
            label="Burn-in samples", 
            alpha=0.5, 
            color=:black,
            markersize=2.5
        )
        scatter!(
            p,
            betas, 
            i_0s, 
            xlabel=L"\beta", 
            ylabel=L"i_0",
            label="HMC samples", 
            alpha=0.5, 
            legend=false,
            xlims=[0,2.3],
            xticks=[0.0, 0.5, 1.0, 1.5, 2.0],
            ylims=[0,60],
            color=palette(:default)[3],
            fontfamily="serif-roman", 
            framestyle=:semi,
            grid=false,
            thickness_scaling=1.5,
            markersize=2.5
        )
        savefig(p, joinpath(experiment_folder, "sir_joint_dist_mcmc_with_burn_in.png"))

        # Plot the maximum weight
        # veclosses = vec(losses)
        # ixs = Array(0:4500:size(veclosses, 1))
        # ixs[1] = 90
        # cummax_weight = Array{Float64,1}(undef, length(ixs))
        # for (i, ix) in enumerate(ixs)
        #     cummax_weight[i] = maximum(veclosses[1:ix])
        # end

        # plot_ixs = Array(0:50:10_000)
        # plot_ixs[1] = 1
        # p = plot(
        #     plot_ixs, 
        #     cummax_weight, 
        #     xscale=:log10,
        #     xlabel="Number of Samples",
        #     ylabel="Maximum log weight",
        #     legend=false
        # )
        # savefig(p, joinpath(experiment_folder, "max_weight.png"))
    elseif subfolder == "radon_nuts_baseline"
        num_intermediates = 200
        @unpack cost_fn_name = experiment_config
        @show cost_fn_name
        cost_fn = RADON_COST_FNS[cost_fn_name]

        # Compute weights by applying cost function to all samples
        # If we just use Array(chn) then the different chains get appended and 
        # we "lose" the third dimension. Using chn.value.data preserves the three 
        # dimensional shape.
        alpha = chn[burn_in:end,namesingroup(chn, :alpha),1:2000].value.data
        beta = chn[burn_in:end,namesingroup(chn, :beta),1:2000].value.data
        losses = Array{Float64}(undef, size(alpha, 1), size(alpha, 3))
        fill!(losses, -Inf)
        for sample_ix in 1:size(losses, 1)
            for chain_ix in 1:size(losses, 2)
                losses[sample_ix,chain_ix] = cost_fn(
                    alpha[sample_ix,:,chain_ix],
                    beta[sample_ix,:,chain_ix],
                    POST_PRED_COUNTY_IDX,
                    POST_PRED_FLOOR
                )
            end
        end
    elseif subfolder == "posterior_predictive_mh_baseline"
        num_intermediates = 100

        cost_fn(x) = pdf(MvNormal(-POST_PRED_Y_OBSERVED, 0.5*I), x)

        savefig(
            plot(chn), joinpath(experiment_folder, "traces.png")
        )

        
        xs = chn[burn_in:end,namesingroup(chn, :x),:].value.data
        println("Cost function evaluation time:")
        @time begin
        losses = Array{Float64}(undef, size(xs, 1), size(xs, 3))
        for sample_ix in 1:size(losses, 1)
            for chain_ix in 1:size(losses, 2)
                losses[sample_ix,chain_ix] = cost_fn(
                    xs[sample_ix,:,chain_ix]
                )
            end
        end
        end
    end

    # ESS based on Turing implementation
    # fxs = losses
    # @show size(fxs)
    # ess_vals = Array{Float64}(undef, num_replications)
    # chains_per_replication = Int(size(fxs, 2) / num_replications)
    # start_ix = 1
    # for i in 1:num_replications
    #     @show start_ix
    #     fxs_replication = fxs[:,start_ix:(start_ix+chains_per_replication-1)]
    #     fxs_chn = Chains(
    #         reshape(
    #             fxs_replication, 
    #             size(fxs_replication, 1), 
    #             1, 
    #             size(fxs_replication, 2)
    #         ), 
    #         ["fx"]
    #     )
    #     ess_vals[i] = ess(fxs_chn).nt[:ess][1]
    #     start_ix = start_ix + chains_per_replication
    # end
    # @show ess_vals
    # return ess_vals

    # Compute ESS based on those weights
    # num_samples = prod(size(losses))
    fxs = reshape(losses, :, num_replications)
    # println("ESS time:")
    # @time begin
    # #fake_ess = ess_batch(fake_log_weights; thinning=thinning)
    # fake_ess = Array{Float64,2}(undef, 100, 10)
    # end

    # Compute the estimates of the actual expecation
    @show size(fxs)
    println("Expectation estimation time:")
    @time begin
    exp_estimates = cumsum(fxs, dims=1) ./ Array(1:size(fxs, 1))
    end

    @show size(fake_ess)
    @show size(exp_estimates)
    @tagsave(joinpath(experiment_folder, "ess.jld2"), Dict(
        "fake_ess" => fake_ess,
        "num_intermediates" => num_intermediates,
        "exp_estimates" => exp_estimates
    ))

    return
    ixs = Array(1:thinning:num_samples) ./ num_intermediates
    ixs[1] = 1
    ess_plot = plot(
        ixs,
        fake_ess,
        xscale=:log10,
        yscale=:identity,
        label="MCMC",
        legend=true,
        xlabel="Number of Samples",
        ylabel="ESS"
    )
    savefig(ess_plot, joinpath(experiment_folder, "ess_plot_burn_in=$(burn_in).png"))
end

# analyse_nuts_baseline(
#     "sir_nuts_baseline",
#     "combined_baselines_5replications"; 
#     thinning=90,
#     num_replications=5
# )

# analyse_nuts_baseline(
#     "posterior_predictive_mh_baseline",
#     "new_model_combined_baselines_5mh_10replications"; 
#     thinning=90,
#     num_replications=10
# )

analyse_nuts_baseline(
    "radon_nuts_baseline",
    "combined_baselines_10replications"; 
    thinning=90,
    num_replications=10
)