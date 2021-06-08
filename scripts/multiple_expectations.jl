using DrWatson
@quickactivate "EPExperiment"

using EPExperiments
using AnnealedIS

function multiple_expectations()
    @expectation function expt_prog(y)
        x ~ Normal(0, 1)
        y ~ Normal(x, 1)
        return x, x^2, x^3
    end

    y_observed = 3
    expt_prog1, expr_prog2, expt_prog3 = expt_prog
    expct1 = expt_prog1(y_observed)

    expct1_estimate, diagnostics = estimate_expectation(
        expct1, TABI(TuringAlgorithm(AnIS(), 100))
    )
end

multiple_expectations()