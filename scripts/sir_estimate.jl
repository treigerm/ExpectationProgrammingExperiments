using DrWatson
@quickactivate "EPExperiments"

using EPExperiments
using Random
using Parameters
using Distributions
using AnnealedIS
using StatsPlots: savefig
using StatsFuns: logsumexp, logistic
using StatsBase
using Logging
using LoggingExtras
using AdvancedMH
using AdvancedHMC
using Dates
using JLD2
using FileIO

const LOG_FILENAME = "out.log"

function estimate_sir(
    experiment_config::SIRExperimentConfig
)
    @unpack num_samples, num_betas, anis_params, seed, cost_fn_name = experiment_config
    Random.seed!(seed)

    cost_fn = COST_FUNCTIONS[cost_fn_name]
    
    input_data = load(datadir("sims", experiment_config.data_filename))

    @unpack tmax, total_population, i_0_true, β_true, γ = input_data["params"]
    X, Y = input_data["X"], input_data["Y"]

    result_folder = mkpath(joinpath(
        datadir("inferences"),
        savename(experiment_config)
    ))
    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, LOG_FILENAME))
    )

    # Prior predictive simulation
    ode_prior = sample(bayes_sir(Y, total_population, γ, cost_fn).gamma2, Prior(), 1000)
    obstimes = 1.0:1.0:tmax
    savefig(
        plot_predictive(obstimes, X, Y, ode_prior), 
        joinpath(result_folder, "prior_pred.png")
    )

    # We first create a geometric spacing between 1 and 1001 because directly 
    # doing it between 0 and 1 gives numerical problems.
    betas = (geomspace(1, 1001, num_betas) .- 1) ./ 1000

    anis_alg = get_anis_alg(anis_params, betas)
    tabi = TABI(
        TuringAlgorithm(anis_alg, num_samples),
        TuringAlgorithm(anis_alg, 0),
        TuringAlgorithm(anis_alg, num_samples)
    )

    @time begin
    R0_estimate, ode_tabi = estimate_expectation(
        bayes_sir(Y, total_population, γ, cost_fn), tabi; store_intermediate_samples=true
    )
    end

    # Save experimental data
    @tagsave(joinpath(result_folder, "results.jld2"), Dict(
        "ode_tabi" => ode_tabi,
        "R0_estimate" => R0_estimate,
        "experiment_config" => experiment_config
    ))

    process_results(
        result_folder,
        logger,
        ode_tabi=ode_tabi,
        R0_estimate=R0_estimate,
        tabi=tabi,
        X=X,
        Y=Y,
        obstimes=obstimes,
        i_0_true=i_0_true,
        β_true=β_true,
        cost_fn=((i, b) -> cost_fn(b, γ))
    )
end

estimate_sir(SIRExperimentConfig(
    experiment_name="trial_run",
    num_samples=1000,
    anis_params=AnISHMCConfig{Turing.ForwardDiffAD{2}}(
        proposal=AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(0.05), 10)
    ),
    cost_fn_name=:cost_fn_sir,
    seed=1298
))
