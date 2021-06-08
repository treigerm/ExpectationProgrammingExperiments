# NOTE: Need to make sure to load the following scripts before running this script:
# include("scripts/posterior_predictive_estimate.jl")
# include("scripts/post_hoc_plots.jl")
# include("scripts/radon_combined_post_hoc_plots.jl")

# Plots for posterior predictive model
function make_paper_plots()
    posterior_predictive_estimate_name = # Fill out
    posterior_predictive_baseline_name = # Fill out
    reprocess_pp_results(
        posterior_predictive_estimate_name;
        mcmc_baseline_name=posterior_predictive_baseline_name
    )

    # Plots for SIR model
    sir_estimate_name = # Fill out
    sir_baseline_name = # Fill out
    sir_data_file = # Fill out
    sir_ground_truth_calculation = # Fill out
    post_hoc_analysis(
        sir_data_file, 
        sir_ground_truth_calculation,
        sir_estimate_name;
        mcmc_baseline_name=sir_baseline_name,
        make_joint_plot=false
    )

    # Plots for Radon model
    radon_estimate_name = # Fill out
    radon_combined_post_hoc_plots(radon_estimate_name)
end

make_paper_plots()