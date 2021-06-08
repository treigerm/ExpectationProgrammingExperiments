function cost_fn_sir(β, γ; k=1) 
    return 1_000_000_000_000 * logistic(k*(10*base_reproduction_rate(β, γ) - 30))
end

const COST_FUNCTIONS = (
    cost_fn_sir = cost_fn_sir,
)
