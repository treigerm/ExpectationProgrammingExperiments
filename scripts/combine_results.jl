using DrWatson
@quickactivate "EPExperiment"

using EPExperiments

using FileIO
using JLD2

function split(x, n)
    result = Vector{Vector{eltype(x)}}()
    start = firstindex(x)
    for len in n
      push!(result, x[start:(start + len - 1)])
      start += len
    end
    result
  end

function combine_results(
    subfolder,
    tabi_result_key,
    experiment_names, 
    out_name; 
    n_replications=1
)
    num_experiments = length(experiment_names)

    # Load all experiments
    Z1_chns = Chains[]
    Z2_chns = Chains[]
    experiment_config = nothing
    for en in experiment_names 
        results = load(joinpath(datadir(subfolder, en), "results.jld2"))
        experiment_config = results["experiment_config"]
        push!(Z1_chns, results[tabi_result_key][:Z1_positive_info])
        push!(Z2_chns, results[tabi_result_key][:Z2_info])
    end

    experiments_per_replication = Int(num_experiments / n_replications)

    ex_idx = Array(1:num_experiments)
    ex_idx = split(
        ex_idx, repeat([experiments_per_replication], n_replications)
    )

    Z1_chns2 = Chains[]
    Z2_chns2 = Chains[]
    for idx in ex_idx
        push!(Z1_chns2, vcat(Z1_chns[idx]...))
        push!(Z2_chns2, vcat(Z2_chns[idx]...))
    end

    # Combine MCMCChains
    Z1_chn = chainscat(Z1_chns2...)
    Z2_chn = chainscat(Z2_chns2...)

    out_folder = mkpath(datadir(subfolder, out_name))
    @tagsave(joinpath(out_folder, "results.jld2"), Dict(
        tabi_result_key => (
            Z1_positive_info = Z1_chn, 
            Z1_negative_info = nothing,
            Z2_info = Z2_chn
        ),
        "experiment_config" => experiment_config
    ))

    open(joinpath(out_folder, "experiments.txt"), "w") do f
        for en in experiment_names
            write(f, "$(en)\n")
        end
    end
end

# NOTE: This array needs to be populated with the folder names of the experiments
#       that you want to combine.
const FOLDERS_WITH_RESULTS = []
    
# Results are saved in a new folder called combined_results
combine_results(
    "sir_estimate",
    "ode_tabi",
    FOLDERS_WITH_RESULTS, 
    "combined_results",
    n_replications=10
)