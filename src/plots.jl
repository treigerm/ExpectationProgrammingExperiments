using StatsPlots

function plot_predictive(obstimes, X_true, Y_obs, chain)
    Xp = []
    for i in 1:10
        pred = predict(Y_obs, chain)
        push!(Xp, pred[2:end,6])
    end

    p = plot(obstimes, Xp; legend=false, color=:red, alpha=0.8)
    plot!(p, obstimes, X_true, color=:black, lw=3)
    scatter!(p, obstimes, Y_obs)
    return p
end

function plot_intermediate_samples(
    intermediate_samples, 
    param_name, 
    true_param,
    betas
)
    inter_samples = permutedims(hcat(intermediate_samples...), [2,1])

    num_samples = size(inter_samples, 1)
    p = plot(title="Intermediate samples for $(string(param_name))", legend=false)
    plot!(p, [true_param], color=:red, seriestype="vline")
    for i in 1:size(inter_samples, 2)
        ix = i > 10 ? (i-9)*10 : i
        beta = betas[ix]
		scatter!(
			p, 
			map(x->x[param_name], inter_samples[:,i]), 
			ones(num_samples)*beta, 
			alpha=0.2,
            color=:black,
            xlims=[-0.1,0.1],
            ylims=[0,1.0]
		)
    end
    return p
end

function plot_intermediate_weights(intermediate_weights, betas; ixs=nothing)
    inter_weights = permutedims(hcat(intermediate_weights...), [2,1])

    num_samples = size(inter_weights, 1)
    p = plot(
        title="Intermediate weights", 
        xlabel="betas",
        ylabel="log weight",
        legend=false
    )

    if isnothing(ixs) 
        ixs = 1:size(inter_weights, 2)
    end
    @show size(inter_weights)
	for i in ixs
        ix = i > 10 ? (i-9)*10 : i
        beta = betas[ix]
		scatter!(
			p,
			ones(num_samples)*beta, 
			inter_weights[:,i], 
			alpha=0.2,
            color=:black
		)
    end
    return p
end

function plot_intermediate_acceptance(intermediate_acceptance, betas; ixs=nothing)
    inter_acceptance = permutedims(hcat(intermediate_acceptance...), [2,1])
    num_samples = size(inter_acceptance, 1)

    inter_means = mean(inter_acceptance, dims=1)[1,:]

    beta_ix = [i > 10 ? (i-9)*10 : i for i in 1:size(inter_acceptance, 2)]
    p = plot(
        beta_ix,
        inter_means,
        fillalpha=0.4,
        title="Intermediate Acceptance Rate", 
        xlabel="betas",
        ylabel="acceptance rate",
        legend=false,
        color=:red,
        lw=2
    )

    if isnothing(ixs) 
        ixs = 1:size(inter_acceptance, 2)
    end
	for i in ixs
        ix = i > 10 ? (i-9)*10 : i
		scatter!(
			p,
			ones(num_samples)*ix, 
			inter_acceptance[:,i], 
			alpha=0.2,
            color=:black
		)
    end
    return p
end

function plot_intermediate_ess(intermediate_weights, betas; ixs=nothing)
    inter_weights = permutedims(hcat(intermediate_weights...), [2,1])

    ess = exp.(
        2 * logsumexp(inter_weights; dims=1) - logsumexp(2 * inter_weights; dims=1)
    )[1,:]

    if isnothing(ixs)
        ixs = 1:size(inter_weights, 2)
    end
    dist_ixs = [i > 10 ? (i-9)*10 : i for i in ixs]
    betas = [betas[i] for i in dist_ixs]
    ess = ess[ixs]
    return plot(
        dist_ixs,
        ess,
        title="ESS for intermediate distributions",
        xlabel="beta index",
        ylabel="ESS",
        lw=2,
        color=:black,
        legend=false
    )
end

function plot_joint_dist(chain, beta_true, i_0_true)
    betas = get(chain, :β)[:β]
    i_0s = get(chain, :i₀)[:i₀]
    p = scatter(
        betas, i_0s, xlabel="β", ylabel="i₀", label="AnIS samples", color=:black
    )
    scatter!(
        p, [beta_true], [i_0_true], label="True Params", color=:red, marker=:x
    )
    return p
end

function plot_sir_samples(chain; color=:blue)
    betas = get(chain, :β)[:β]
    i_0s = get(chain, :i₀)[:i₀]
    p = scatter(
        betas, 
        i_0s, 
        xlabel=L"\beta", 
        ylabel=L"i_0", 
        alpha=0.5,
        legend=false,
        xlims=[0,2.3],
        xticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        ylims=[0,60],
        fontfamily="serif-roman", 
        framestyle=:semi,
        grid=false,
        color=color,
        thickness_scaling=1.5,
        markersize=2.5
    )
    return p
end

function plot_is_samples(samples, log_weights)
    return scatter(
        samples,
        log_weights, 
        markersize=3, 
        markeralpha=0.4, 
        legend=false,
        xlabel="beta",
        ylabel="log weight",
        xlims=[0,2],
        ylims=[-40,0]
    )
end

function get_median_and_quantiles(errors)
    # errors: num_samples x num_chains
    qs = [quantile(errors[i,:], [0.25, 0.75]) for i in 1:size(errors,1)]
    qs = hcat(qs...)
    medians = median(errors, dims=2)[:,1]
    # medians: num_samples
    # qs: 2 x num_samples
    return medians, qs
end

function convergence_plot(
    problem_type,
    Z1_chn, 
    Z2_chn, 
    f, 
    ground_truth;
    thinning=1,
    num_xlogspaced=nothing,
    Z1_negative_chn=nothing,
    mean_and_std=false
)
    # Shape: num_samples x num_chains
    Z1_ws = exp.(Array(Z1_chn[:log_weight]))
    Z2_ws = exp.(Array(Z2_chn[:log_weight]))

    num_samples = size(Z1_ws, 1)
    #resampled_ixs = StatsBase.sample(1:num_samples, pweights(Z2_ws), num_samples)
    #Z2_ws = Z2_ws[resampled_ixs]

    # NOTE: Usually we have to divide each estimate by the number of samples used 
    # but because those "normalisations" cancel in our downstream calculations we 
    # avoid them.
    Z1s_unnormalised = cumsum(Z1_ws, dims=1)
    Z2s_unnormalised = cumsum(Z2_ws, dims=1)

    if isnothing(Z1_negative_chn)
        tabi_ests = Z1s_unnormalised ./ Z2s_unnormalised
    else
        Z1_negative_ws = exp.(Array(Z1_negative_chn[:log_weight]))
        Z1_negatives_unnormalised = cumsum(Z1_negative_ws, dims=1)
        tabi_ests = (Z1s_unnormalised .- Z1_negatives_unnormalised) ./ Z2s_unnormalised
    end

    if problem_type == "sir"
        i₀ = Array(Z2_chn[:i₀])
        β = Array(Z2_chn[:β])
        anis_ests = Array{Float64,2}(undef, num_samples, size(Z2_ws, 2))
        for ix in CartesianIndices(Z2_ws)
            anis_ests[ix] = Z2_ws[ix] * f(i₀[ix], β[ix])
        end
        #i₀ = Array(Z2_chn[:i₀])[resampled_ixs]
        #β = Array(Z2_chn[:β])[resampled_ixs]
        #anis_ests = cumsum(map(zip(ones(num_samples), i₀, β)) do (w, i, b)
        #    w * f(i, b)
        #end)
        #anis_ests = anis_ests ./ (1:num_samples)
    elseif problem_type == "post_pred"
        xs = Z2_chn[:,namesingroup(Z2_chn, :x),:].value.data
        anis_ests = Array{Float64,2}(undef, num_samples, size(Z2_ws, 2))
        for ix in CartesianIndices(Z2_ws)
            sample_ix, chain_ix = ix[1], ix[2]
            anis_ests[ix] = Z2_ws[ix] * f(xs[sample_ix,:,chain_ix])
        end
    elseif problem_type == "seventh_moment"
        xs = Array(Z2_chn[:x])
        anis_ests = Array{Float64,2}(undef, num_samples, size(Z2_ws, 2))
        for ix in CartesianIndices(Z2_ws)
            sample_ix, chain_ix = ix[1], ix[2]
            anis_ests[ix] = Z2_ws[ix] * f(xs[sample_ix,chain_ix])
        end
    else
        error("Unknown problem type")
    end
    anis_ests = cumsum(anis_ests, dims=1)
    anis_ests = anis_ests ./ Z2s_unnormalised

    tabi_ests = tabi_ests[1:thinning:end,:]
    anis_ests = anis_ests[1:thinning:end,:]

    @show tabi_ests[end,:]
    @show anis_ests[end,:]
    tabi_rserror = relative_squared_error.(ground_truth, tabi_ests)
    anis_rserror = relative_squared_error.(ground_truth, anis_ests)
    @show tabi_rserror[end,:]
    @show anis_rserror[end,:]

    #ixs = 50:num_samples
    ixs = 1:thinning:num_samples
    taanis_ixs = if isnothing(Z1_negative_chn)
        2 * ixs # TAAnIS uses twice the number of samples
    else
        3 * ixs
    end

    data = [
        ("TAAnIS", taanis_ixs, tabi_rserror),
        ("AnIS", ixs, anis_rserror)
    ]
    error_plot = plot(
        yscale=:log10,
        xscale=:log10,
        xlabel="Number of Samples",
        ylabel="RSE",
        legend=false,
        thickness_scaling=1.5,
        fontfamily="serif-roman", 
        framestyle=:semi, 
        grid=false,
        xlims=(10,10^4),
        # ylims=(10^(-6.5),1),
        ylims=(10^(-8),1),
        xticks=[10, 100, 1_000, 10_000],
        yticks=[1, 0.01, 0.0001, 0.000001]
    )
    for (label, indexes, rserrors) in data
        if mean_and_std
            mds = vec(mean(rserrors, dims=2))
            qs = vec(std(rserrors, dims=2))
        else
            mds, qs = get_median_and_quantiles(rserrors)
        end
        means = vec(mean(rserrors, dims=2))
        mds, qs = get_median_and_quantiles(rserrors)

        if !isnothing(num_xlogspaced)
            log_ixs = unique(round.(Int, geomspace(1, num_samples, num_xlogspaced)))
            mds = mds[log_ixs]
            # qs = if mean_and_std
            #     qs[log_ixs]
            # else
            #     qs[:,log_ixs]
            # end
            qs = qs[:,log_ixs]
            means = means[log_ixs]
            indexes = Array(indexes)[log_ixs]
        end

        # Error bars differ when we use quantiles instead of standard deviation.
        # ribbon = if mean_and_std
        #     qs
        # else
        #     [(mds.-qs[1,:]), (qs[2,:]-mds)]
        # end
        ribbon = [(mds.-qs[1,:]), (qs[2,:]-mds)]
        plot!(
            error_plot,
            indexes, 
            mds,#mds[ixs],
            ribbon=ribbon,
            label=label
        )
    end
    return error_plot, nothing

    ground_truth_plot = plot(
        taanis_ixs,
        tabi_ests[ixs],
        label="TAAnIS"
    )
    plot!(
        ground_truth_plot,
        ixs,
        anis_ests[ixs],
        label="AnIS"
    )
    plot!(
        ground_truth_plot,
        [1, 2*num_samples],
        repeat([ground_truth], 2),
        label="GT"
    )
    return error_plot, ground_truth_plot
end

function ess_batch(log_weights; thinning=1)
    # Input: (num_samples, num_chains)
    # Output: (num_samples/thinning, num_chains)

    num_samples = size(log_weights, 1)
    ess = Array{Float64}(undef, Int(num_samples/thinning), size(log_weights,2))

    ess_ix = 1
    for ix in 1:thinning:num_samples
        denominator = logsumexp(2 * log_weights[1:ix,:]; dims=1)
        numerator = 2 * logsumexp(log_weights[1:ix,:]; dims=1)
        ess[ess_ix,:] = exp.(numerator .- denominator)
        ess_ix += 1
    end
    return ess
end

function make_ess_plot(
    Z1_lws, 
    Z2_lws, 
    anis_lws_retargeted; 
    thinning=1, 
    anis_lws=nothing,
    yaxisscale=:log10,
    take_min=true,
    num_xlogspaced=nothing,
    legend=false
)
    num_samples = size(Z1_lws, 1)

    Z1_ess_cumsum = ess_batch(Z1_lws; thinning=thinning)
    Z2_ess_cumsum = ess_batch(Z2_lws; thinning=thinning)

    if isnothing(anis_lws)
        anis_lws = Z2_lws
    end
    anis_ess_cumsum = ess_batch(anis_lws; thinning=thinning)
    anis_ess_retargeted_cumsum = ess_batch(anis_lws_retargeted; thinning=thinning)

    ixs = Array(1:thinning:num_samples)
    # taanis_mds, taanis_qs = get_median_and_quantiles(taanis_min_ess)

    # ess_plot = plot(
    #     ixs*2,
    #     taanis_mds,
    #     ribbon=[(taanis_mds.-taanis_qs[1,:]), (taanis_qs[2,:].-taanis_mds)],
    #     xscale=:log10,
    #     yscale=yaxisscale,
    #     label="TAAnIS",
    #     legend=false,
    #     xlabel="Number of Samples",
    #     ylabel="ESS",
    #     xlims=(10,10^4),
    #     thickness_scaling=1.7
    # )
    # anis_ixs = Array(1:thinning:num_samples)
    # anis_mds, anis_qs = get_median_and_quantiles(anis_min_ess)
    # plot!(
    #     ess_plot,
    #     anis_ixs,
    #     anis_mds,
    #     ribbon=[(anis_mds.-anis_qs[1,:]), (anis_qs[2,:].-anis_mds)],
    #     label="AnIS"
    # )
    
    taanis_color = palette(:default)[1]
    anis_color = palette(:default)[2]
    ess_quantities = if take_min
        taanis_min_ess = min.(Z1_ess_cumsum, Z2_ess_cumsum)
        anis_min_ess = min.(anis_ess_retargeted_cumsum, anis_ess_cumsum)
        [
            ("TAAnIS", ixs*2, get_median_and_quantiles(taanis_min_ess), taanis_color, :solid),
            ("AnIS", ixs, get_median_and_quantiles(anis_min_ess), anis_color, :solid)
        ]
    else 
        [
            ("Z1 (TAAnIS)", ixs*2, get_median_and_quantiles(Z1_ess_cumsum), taanis_color, :dash),
            ("Z2 (TAAnIS)", ixs*2, get_median_and_quantiles(Z2_ess_cumsum), taanis_color, :solid),
            ("AnIS", ixs, get_median_and_quantiles(anis_ess_cumsum), anis_color, :solid),
            ("AnIS retargeted", ixs, get_median_and_quantiles(anis_ess_retargeted_cumsum), anis_color, :dash)
        ]
    end

    name1, indices1, (medians1, quantiles1), cl1, ls1 = ess_quantities[1]
    if !isnothing(num_xlogspaced)
        log_ixs = unique(round.(Int, geomspace(1, num_samples, num_xlogspaced)))
        indices1 = indices1[log_ixs]
        medians1 = medians1[log_ixs]
        quantiles1 = quantiles1[:,log_ixs]
    end
    ess_plot = plot(
        indices1,
        medians1,
        ribbon=[(medians1.-quantiles1[1,:]), (quantiles1[2,:].-medians1)],
        xscale=:log10,
        yscale=yaxisscale,
        label=name1,
        color=cl1,
        linestyle=ls1,
        legend=legend,
        xlabel="Number of Samples",
        ylabel="ESS",
        xlims=(10,10^4),
        thickness_scaling=1.5,
        fontfamily="serif-roman", 
        framestyle=:semi, 
        grid=false
    )
    for (name, indices, (medians, quantiles), cl, ls) in ess_quantities[2:end]
        if !isnothing(num_xlogspaced)
            log_ixs = unique(round.(Int, geomspace(1, num_samples, num_xlogspaced)))
            indices = indices[log_ixs]
            medians = medians[log_ixs]
            quantiles = quantiles[:,log_ixs]
        end
        plot!(
            ess_plot,
            indices,
            medians,
            ribbon=[(medians.-quantiles[1,:]), (quantiles[2,:].-medians)],
            color=cl,
            linestyle=ls,
            label=name
        )
    end
    return ess_plot
end