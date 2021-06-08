module EPExperiments

using Reexport
@reexport using DifferentialEquations
using DiffEqSensitivity
@reexport using EPT
using StatsFuns: logsumexp, logistic
using DataFrames
@reexport using Parameters
using Logging
using LoggingExtras
using LaTeXStrings
import LinearAlgebra

export sir_ode!,
    base_reproduction_rate,
    predict,
    neg_bin_params2,
    bayes_sir,
    geomspace,
    SIRParameters,
    plot_predictive,
    process_results,
    COST_FUNCTIONS,
    plot_is_samples,
    MHTransitionConfig,
    MHFixedTuple,
    MHThreeLevel,
    MHFunction,
    AnISConfig,
    ISConfig,
    AnISHMCConfig,
    AnISHMCGranularConfig,
    AnISMHConfig,
    SIRExperimentConfig,
    convergence_plot,
    ISGroundTruthConfig,
    get_anis_alg,
    radon_hierarchical,
    preprocess_radon_df,
    POST_PRED_COUNTY_IDX,
    POST_PRED_FLOOR,
    RADON_COST_FNS,
    radon_target_f,
    plot_intermediate_ess,
    plot_intermediate_weights,
    plot_intermediate_acceptance,
    get_median_and_quantiles,
    check_diagnostics,
    ess_batch,
    make_ess_plot,
    relative_squared_error,
    plot_joint_dist,
    plot_sir_samples,
    post_pred,
    POST_PRED_Y_OBSERVED,
    POST_PRED_GROUND_TRUTH,
    POST_PRED_DIMENSION,
    compute_ess,
    banana,
    banana_f

# Constants for the loss function for radon problem.
const POST_PRED_COUNTY_IDX = 10
const POST_PRED_FLOOR = 0

include("configs.jl")
include("cost_functions.jl")
include("check_diagnostics.jl")
include("post_pred_model.jl")

include("radon_model.jl")

function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,γ) = p
    N = S+I+R
    infection = β*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
    nothing
end

function base_reproduction_rate(β, γ)
    return β / γ
end

function predict(y,chain)
    # Length of data
    l = length(y)
    # Length of chain
    m = length(chain)
    # Choose random
    idx = sample(1:m)
    i₀ = Array(chain[:i₀])[idx]
    β = Array(chain[:β])[idx]
    I = i₀*10.0
    u0=[1000.0-I,I,0.0,0.0]
    p=[β,0.25]
    tspan = (0.0,float(l))
    prob = ODEProblem(sir_ode!,
            u0,
            tspan,
            p)
    sol = solve(prob,
                Tsit5(),
                saveat = 1.0)
    out = Array(sol)
    sol_X = [0.0; out[4,2:end] - out[4,1:(end-1)]]
    return hcat(sol.t, out', sol_X)
end

neg_bin_r(mean, var) = mean^2 / (var - mean)
neg_bin_p(r, mean) = r / (r + mean)

function neg_bin_params(mean, var)
    r = neg_bin_r(mean, var)
    return r, neg_bin_p(r, mean)
end

neg_bin_params2(mean, phi) = neg_bin_params(mean, mean + mean^2 / phi)

@expectation function bayes_sir(y, total_population, γ, cost_fn)
    # Calculate number of timepoints
    l = length(y)
    i₀ ~ truncated(Normal(10, 10), 0, total_population/10)
    β ~ truncated(Normal(2, 1.5), 0, Inf)

    I = i₀ * 10
    u0 = [total_population-I, I, 0.0, 0.0]
    p = [β, γ]

    tspan = (0.0, float(l))
    prob = ODEProblem(sir_ode!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = 1.0)
    sol_C = Array(sol)[4,:] # Cumulative cases
    sol_X = sol_C[2:end] - sol_C[1:(end-1)]
    
    l = length(y)
    if any(sol_X .< 0) || length(sol_X) != l 
        # Check if we have negative cumulative cases
        @warn "Negative cumulative cases" β i₀
        Turing.acclogp!(_varinfo, -Inf)
        return l
    end
    
    phi = 0.5
    variance = sol_X .+ sol_X.^2 ./ phi
    rs = neg_bin_r.(sol_X, variance)
    ps = neg_bin_p.(rs, sol_X)
    if !all(rs .> 0) || !all(0 .< ps .<= 1)
        Turing.acclogp!(_varinfo, -Inf)
        return l
    end
    
    y ~ arraydist(NegativeBinomial.(rs, ps))
    #y ~ arraydist([Poisson(x) for x in sol_X])
    return cost_fn(β, γ)
end

function geomspace(start, stop, length)
    logstart = log10(start)
    logstop = log10(stop)
    points = 10 .^ range(logstart, logstop; length=length)
    points[1] = start
    points[end] = stop
    return points
end

function relative_squared_error(mu_true, mu_hat)
    return (mu_true - mu_hat)^2 / mu_true^2
end

function compute_ess(log_weights)
    denominator = logsumexp(2 * log_weights)
    numerator = 2 * logsumexp(log_weights)
    return exp(numerator - denominator)
end

include("plots.jl")

function est_exp(chn, f)
    log_weights = Array(chn[:log_weight])
    i₀ = Array(chn[:i₀])
    β = Array(chn[:β])

    normalisation = exp(logsumexp(log_weights))
    weighted_sum = sum(zip(log_weights, i₀, β)) do (lw, i, b)
        exp(lw) * f(i, b)
    end

    return weighted_sum / normalisation
end

function process_results(
    result_folder,
    logger; 
    ode_tabi, 
    R0_estimate,
    tabi,
    X,
    Y,
    obstimes,
    i_0_true,
    β_true,
    cost_fn
)
    betas = tabi.Z2_alg.inference_algorithm.betas

    savefig(plot_intermediate_weights(
        ode_tabi[:Z2_info].info[:intermediate_log_weights], betas),
        joinpath(result_folder, "intermediate_weights.png")
    )
    savefig(plot_intermediate_weights(
        ode_tabi[:Z2_info].info[:intermediate_log_weights], betas; ixs=1:14),
        joinpath(result_folder, "intermediate_weights_first40.png")
    )
    savefig(plot_intermediate_ess(
        ode_tabi[:Z2_info].info[:intermediate_log_weights], betas),
        joinpath(result_folder, "intermediate_ess.png")
    )
    savefig(plot_intermediate_ess(
        ode_tabi[:Z2_info].info[:intermediate_log_weights], betas; ixs=1:14),
        joinpath(result_folder, "intermediate_ess_first40.png")
    )
    savefig(
        plot_predictive(obstimes, X, Y, ode_tabi[:Z2_info]), 
        joinpath(result_folder, "anis_post_pred.png")
    )
    savefig(
        plot_joint_dist(ode_tabi[:Z2_info], β_true, i_0_true),
        joinpath(result_folder, "joint_samples.png")
    )

    anis_estimate = est_exp(ode_tabi[:Z2_info], cost_fn)

    with_logger(logger) do
        @info "Expectation estimate TABI: $(R0_estimate)"
        @info "Expectation estimate AnIS: $(anis_estimate)"

        @info "Estimation algorithm: $(tabi)"

        for (k, est_results) in pairs(ode_tabi)
            if isnothing(est_results)
                continue
            end
            msg = string([
                "$(string(k)):\n", 
                "ESS: $(est_results.info[:ess])\n",
                "Log evidence: $(est_results.logevidence)\n",
                "Log weights: $(get(est_results, :log_weight)[:log_weight])\n"
            ]...)

            @info "$(msg)" 
        end
    end
end

end