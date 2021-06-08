using DrWatson
@quickactivate "EPExperiments"

using EPExperiments
using AnnealedIS

using Distributions
using Random
using StatsFuns: logsumexp
using LinearAlgebra: dot
using Logging
using LoggingExtras
using StatsPlots

const LOG_FILENAME = "out.log"

function is_sampling(logjoint, qi, qb, num_samples)
    qi_samples = rand(qi, num_samples)
    qb_samples = rand(qb, num_samples)

    i_b_samples = map(t -> (i₀ = t[1], β = t[2]), zip(qi_samples, qb_samples))
    is_weights = logjoint.(i_b_samples) - (
        logpdf.(qi, qi_samples) .+ logpdf.(qb, qb_samples)
    )
    normalised_is_weights = is_weights .- logsumexp(is_weights)
    
    return normalised_is_weights, qi_samples, qb_samples
end

@with_kw struct ISGroundTruthConfig
    num_samples::Int = Int(1e6)
    qi::Distributions.Distribution = truncated(Normal(10, 10), 0, 1_000)
    qb::Distributions.Distribution = truncated(Normal(1.5, 1.5), 0, Inf)
    cost_fn_name::Symbol = :cost_fn_default
    data_filename::String = "neg_binomial_data.jld2"
    seed::Int = 1234
end

DrWatson.allignore(::ISGroundTruthConfig) = (:data_filename,)

function calculate_ground_truth(is_config)
    @unpack num_samples, qi, qb, cost_fn_name, data_filename, seed = is_config
    cost_fn = COST_FUNCTIONS[cost_fn_name]

    Random.seed!(seed)

    result_folder = mkpath(joinpath(
        datadir("is_ground_truth"),
        savename(is_config)
    ))
    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )
    
    input_data = load(datadir("sims", is_config.data_filename))
    @unpack tmax, total_population, i_0_true, β_true, γ = input_data["params"]
    Y = input_data["Y"]

    tm = bayes_sir(Y, total_population, γ, cost_fn).gamma2
    vi = Turing.VarInfo(tm)
    joint_dens = AnnealedIS.make_log_joint_density(tm, vi)

    @time begin
    log_weights, qi_samples, qb_samples = is_sampling(
        joint_dens, qi, qb, num_samples
    )
    end

    costs = cost_fn.(qb_samples, γ)

    ess = 1 / exp(logsumexp(2 * log_weights))
    target_weights = log_weights .+ log.(costs)
    ess_target = exp(2 * logsumexp(target_weights) - logsumexp(2 * target_weights))

    exp_estimate = dot(exp.(log_weights), costs)
    with_logger(logger) do 
        @info "Proposals:" qi qb
        @info "Number of samples: $num_samples"
        @info "ESS p(x|y): $ess"
        @info "ESS p(x|y)f(x): $ess_target"
        @info "Expectation estimate: $exp_estimate"
    end

    @tagsave(joinpath(result_folder, "results.jld2"), Dict(
        "log_weights" => log_weights,
        "qi_samples" => qi_samples,
        "qb_samples" => qb_samples,
        "is_config" => is_config,
        "expectation_estimate" => exp_estimate
    ))

    savefig(
        plot_is_samples(qb_samples, log_weights),
        joinpath(result_folder, "beta_samples.png")
    )
end

calculate_ground_truth(ISGroundTruthConfig(
    num_samples=Int(1e8),
    cost_fn_name=:cost_fn_shift20
))