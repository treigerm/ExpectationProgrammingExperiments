using AnnealedIS
using AdvancedHMC
using Distributions
using DrWatson
using Dates

@with_kw struct SIRParameters
    tmax::Float32 = 15.0
    total_population::Int = 10_000
    i_0_true::Int = 10
    β_true::Float32 = 0.25
    γ::Float32 = 0.25
    neg_bin_dispersion_param::Float32 = 10
    seed::Int = 1234
end

function transition_kernel(prior_sample::T, i) where {T<:Real}
    if (i > 100) && (i <= 700)
        return Normal(0, 0.01)
    elseif (i > 700) 
        return Normal(0, 0.001)
    else
        return Normal(0, 0.1)
    end
end

function transition_kernel(prior_sample::NamedTuple, i)
    return map(x -> transition_kernel(x, i), prior_sample)
end

abstract type MHTransitionConfig end

# TODO: Relax type constraint so that we can have other distributions than Normal.
@with_kw struct MHFixedTuple <: MHTransitionConfig
    proposal::NamedTuple{(:i₀, :β),Tuple{Normal{Float64},Normal{Float64}}} = (
        i₀ = Normal(0, 1), β = Normal(0, 0.1)
    )
end

function get_transition_kernel(transition_kernel_params::MHFixedTuple)
    function kernel(prior_sample, i)
        return transition_kernel_params.proposal
    end
    return kernel
end

@with_kw struct MHFunction <: MHTransitionConfig
    kernel_fn::Function
end

function get_transition_kernel(transition_kernel_params::MHFunction)
    return transition_kernel_params.kernel_fn
end

struct MHThreeLevel <: MHTransitionConfig end

function get_transition_kernel(transition_kernel_params::MHThreeLevel)
    return transition_kernel
end

abstract type AnISConfig end

@with_kw struct AnISMHConfig{T<:MHTransitionConfig,U<:RejectionSampler} <: AnISConfig
    transition_kernel_cfg::T
    rejection_type::U = SimpleRejection()
end

function get_anis_alg(
    anis_params::AnISMHConfig, betas
)
    return AnIS(
        betas, 
        get_transition_kernel(anis_params.transition_kernel_cfg), 
        anis_params.rejection_type
    )
end

struct ISConfig <: AnISConfig end

get_anis_alg(anis_params::ISConfig, betas) = IS()

@with_kw struct AnISHMCConfig{AD,T<:AdvancedHMC.AbstractProposal,U<:RejectionSampler} <: AnISConfig
    proposal::T 
    num_steps::Int = 10
    rejection_type::U = SimpleRejection()
end

AnISHMCConfig(args...; kwargs...) = AnISHMCConfig{Turing.Core.ForwardDiffAD{1}}(args...; kwargs...)
function AnISHMCConfig{AD}(;
    proposal::T,
    num_steps::Int=10,
    rejection_type::U=SimpleRejection()
) where {AD,T<:AdvancedHMC.AbstractProposal,U<:RejectionSampler}
    return AnISHMCConfig{AD,T,U}(proposal, num_steps, rejection_type)
end

function get_anis_alg(
    anis_params::AnISHMCConfig{AD,T,U}, 
    betas
) where {AD,T<:AdvancedHMC.AbstractProposal,U<:RejectionSampler}
    anis_alg = AnISHMC{AD}(
        betas,
        anis_params.proposal,
        anis_params.num_steps,
        anis_params.rejection_type
    )
end

@with_kw struct AnISHMCGranularConfig{
    AD,
    T<:AdvancedHMC.AbstractProposal,
    M<:AdvancedHMC.AbstractMetric,
    U<:RejectionSampler
} <: AnISConfig
    proposals::Array{T}
    metrics::Array{M}
    num_steps::Int = 10
    rejection_type::U = SimpleRejection()
end

AnISHMCGranularConfig(args...; kwargs...) = AnISHMCGranularConfig{Turing.Core.ForwardDiffAD{1}}(args...; kwargs...)
function AnISHMCGranularConfig{AD}(;
    proposals::Array{T,1},
    metrics::Array{M,1}=repeat(
        [AdvancedHMC.UnitEuclideanMetric(1)], length(proposals)
    ),
    num_steps::Int=10,
    rejection_type::U=SimpleRejection()
) where {
    AD,
    T<:AdvancedHMC.AbstractProposal,
    M<:AdvancedHMC.AbstractMetric,
    U<:RejectionSampler
}
    return AnISHMCGranularConfig{AD,T,M,U}(
        proposals, metrics , num_steps, rejection_type)
end

function get_anis_alg(
    anis_params::AnISHMCGranularConfig{AD,T,M,U}, 
    betas
) where {
    AD,
    T<:AdvancedHMC.AbstractProposal,
    M<:AdvancedHMC.AbstractMetric,
    U<:RejectionSampler
}
    num_betas = length(betas)
    num_metrics = length(anis_params.metrics)
    num_proposals = length(anis_params.proposals)

    @assert num_metrics == num_proposals
    @assert num_metrics == (num_betas - 1)

    anis_alg = AnISHMC{AD}(
        betas,
        anis_params.proposals,
        anis_params.metrics,
        anis_params.num_steps,
        anis_params.rejection_type
    )
end

@with_kw struct SIRExperimentConfig{T<:AnISConfig}
    experiment_name::String = "test"
    num_samples::Int = 10
    num_betas::Int = 101
    anis_params::T = AnISMHConfig(MHFixedTuple(), SimpleRejection())
    seed::Int = 1234
    data_filename::String = "neg_binomial_data.jld2"
    cost_fn_name::Symbol = :cost_fn_default
end

function DrWatson.default_prefix(e::SIRExperimentConfig) 
    datestring = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    return "$(datestring)_$(e.experiment_name)_$(typeof(e.anis_params))" 
end

DrWatson.allignore(::SIRExperimentConfig) = (:data_filename, :experiment_name)

@with_kw struct ISGroundTruthConfig
    num_samples::Int = Int(1e6)
    qi::Distributions.Distribution = truncated(Normal(10, 10), 0, 1_000)
    qb::Distributions.Distribution = truncated(Normal(1.5, 1.5), 0, Inf)
    cost_fn_name::Symbol = :cost_fn_default
    data_filename::String = "neg_binomial_data.jld2"
    seed::Int = 1234
end