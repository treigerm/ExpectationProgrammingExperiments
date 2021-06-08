using DrWatson
@quickactivate "EPExperiment"

using EPExperiments

using FileIO
using JLD2

const CHN_KEY = "nuts_samples"

function combine_nuts_baseline(subfolder, experiment_names, out_dir_name)
    chns_list = Chains[]
    experiment_config = nothing
    for en in experiment_names
        results = load(joinpath(datadir(subfolder, en), "results.jld2"))
        experiment_config = results["experiment_config"]
        push!(chns_list, results[CHN_KEY])
    end

    combined_chn = chainscat(chns_list...)

    @show combined_chn
    out_folder = mkpath(datadir(subfolder, out_dir_name))
    @tagsave(joinpath(out_folder, "results.jld2"), Dict(
        "nuts_samples" => combined_chn,
        "experiment_config" => experiment_config
    ))
end

# TODO: Adjust those arguments to your experiments.
combine_nuts_baseline(
    "radon_nuts_baseline", 
    radon_ens_10_replications, 
    "combined_baselines_10replications"
)