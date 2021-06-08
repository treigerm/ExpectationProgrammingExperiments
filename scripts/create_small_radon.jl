# Create a version of the radon dataset with some counties removed.

using DrWatson
@quickactivate "EPExperiments"

using Random
using CSV
using DataFrames

const NEW_CSV_NAME = "small_radon_num_counties=20.csv"
const NEW_NUM_COUNTIES = 20 # Number of counties in the new dataset

function create_small_radon()
    Random.seed!(1234)

    radon_df = DataFrame(CSV.File(datadir("exp_raw", "radon.csv")))

    num_counties = length(unique(radon_df[:,:county]))
    ixs = randperm(num_counties)[1:NEW_NUM_COUNTIES]
    new_counties = unique(radon_df[:,:county])[ixs]

    small_radon_df = filter(row -> in(row.county, new_counties), radon_df)
    CSV.write(datadir("exp_raw", NEW_CSV_NAME), small_radon_df)
end

create_small_radon()