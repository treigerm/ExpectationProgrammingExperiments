@expectation function radon_hierarchical(
    log_radon, 
    county_idx, 
    floor, 
    num_counties;
    log_sigma_alpha=-2.1,
    log_sigma_beta=-1.5,
    cost_fn=radon_target_f
)
    mu_alpha ~ Normal(0, 10)
    sigma_alpha = exp(log_sigma_alpha)
    
    mu_beta ~ Normal(0, 10)
    sigma_beta = exp(log_sigma_beta)
    
    alpha ~ filldist(Normal(mu_alpha, sigma_alpha), num_counties)
    beta ~ filldist(Normal(mu_beta, sigma_beta), num_counties)
    
    log_eps ~ transformed(truncated(Cauchy(0, 5), 0, Inf))
    eps = exp(log_eps)

    radon_est = alpha[county_idx] .+ beta[county_idx] .* floor
    log_radon ~ MvNormal(radon_est, eps*LinearAlgebra.I)

    # NOTE: Here the county idx and floor are supposed to be fixed.
    return cost_fn(
        alpha, beta, POST_PRED_COUNTY_IDX, POST_PRED_FLOOR)
end

function radon_target_f(alpha, beta, county_idx, floor)
    radon_level(a, b) = exp(a + b * floor)
    # Note we do not have a negative sign to "flip" the logistic function
    step_fun(a, b) = 1 / (1 + exp(5 * (radon_level(a, b) - 4)))
    vals = step_fun.(alpha, beta)
    return max(prod(vals), 1e-200)
end

const RADON_COST_FNS = (
    radon_target_f = radon_target_f,
)

function preprocess_radon_df(radon_df, data_file)
    num_counties = length(unique(radon_df[:,:county]))
    if data_file != "radon.csv"
        # Have N<84 counties but county_code is still between 0 and 84.
        # Need to map all the old county codes to new ones between 1 and N
        old_county_idx = unique(radon_df[:,:county_code])
        oldix2newix = Dict(old_county_idx[i] => i for i in 1:num_counties)
        county_idx = map(ix -> oldix2newix[ix], radon_df[:,:county_code])
    else
        county_idx = radon_df[:,:county_code] .+ 1
    end
    log_radon = radon_df[:,:log_radon]
    floor = radon_df[:,:floor]
    return log_radon, county_idx, floor, num_counties
end