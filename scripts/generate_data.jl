using DrWatson
@quickactivate "EPExperiments"

using EPExperiments
using Random
using JLD2
using FileIO
using Parameters
using Distributions

@with_kw struct SIRParameters
    tmax::Float32 = 15.0
    total_population::Int = 10_000
    i_0_true::Int = 10
    β_true::Float32 = 0.25
    γ::Float32 = 0.25
    neg_bin_dispersion_param::Float32 = 10
    seed::Int = 1234
end

function generate_data(params::SIRParameters)
    @unpack tmax, total_population, i_0_true, β_true, γ, neg_bin_dispersion_param, seed = params

    Random.seed!(seed)
    tspan = (0.0,tmax)
    obstimes = 1.0:1.0:tmax

    I0 = i_0_true * 10
	S0 = total_population - I0
	u0 = [S0, I0, 0.0, 0.0] # S,I.R,C

    # Fixed parameters.
    p = [β_true, γ]

    prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
    sol_ode = solve(prob_ode, Tsit5(), saveat = 1.0)

    C = Array(sol_ode)[4,:] # Cumulative cases
    X = C[2:end] - C[1:(end-1)]
    #Y = rand.(Poisson.(X))
    Y = rand.([NegativeBinomial(neg_bin_params2(m, 10)...) for m in X])
    @tagsave(datadir("sims", "neg_binomial_data.jld2"), Dict(
        "params" => params,
        "Y" => Y,
        "X" => X
    ))
end

generate_data(SIRParameters())