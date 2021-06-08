@expectation function banana()
    x1 ~ Normal(0, 4)
    x2 ~ Normal(0, 4)
    Turing.acclogp!(_varinfo, banana_density(x1, x2))
    return banana_f(x1, x2)
end

function banana_f(x1, x2)
    cond = 1 / (1 + exp(50 * (x2 + 5)))
    return cond * (x1 - 2)^3
end

function banana_density(x1, x2)
    return -0.5 * (0.03 * x1^2 + (x2/2 + 0.03 * (x1^2 - 100))^2 )
end