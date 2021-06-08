using DrWatson
@quickactivate "EPExperiment"

using EPExperiments
using Quadrature

const LOG_DIR_NAME = "bananas_ground_truth"

function bananas_ground_truth()
    result_folder = datadir(LOG_DIR_NAME)
    exp_banana = banana()
    turing_banana = exp_banana.gamma2

    logjoint = AnnealedIS.make_log_joint_density(turing_banana)
    gamma2(x, p) = exp(logjoint((x1 = x[1], x2 = x[2])))
    prob1 = QuadratureProblem(gamma2, [-50, -50], [50, 50])
    Z2 = solve(prob1, HCubatureJL(), abstol=1e-20)
    @show Z2.u

    gamma1(x, p) = exp(logjoint((x1 = x[1], x2 = x[2]))) * banana_f(x[1], x[2])
    prob2 = QuadratureProblem(gamma1, [-30, -40], [30, 10])
    Z1 = solve(prob2, HCubatureJL(), reltol=1e-10, abstol=1e-30)
    @show Z1.u

    @show Z1.u / Z2.u

    @tagsave(joinpath(result_folder, "ground_truth.jld2"), Dict(
        "expectation_estimate" => Z1.u / Z2.u
    ))
end

bananas_ground_truth()