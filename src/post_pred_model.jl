using LinearAlgebra: I

const POST_PRED_YVAL = -3.5
const POST_PRED_DIMENSION = 10
const POST_PRED_Y_OBSERVED = POST_PRED_YVAL * ones(POST_PRED_DIMENSION) / sqrt(POST_PRED_DIMENSION)
const POST_PRED_GROUND_TRUTH = pdf(
    MvNormal(POST_PRED_Y_OBSERVED / 2, I), -POST_PRED_Y_OBSERVED
)

@expectation function post_pred(y)
    x ~ MvNormal(zeros(length(y)), I) 
    y ~ MvNormal(x, I)
    return pdf(MvNormal(-y, 0.5*I), x)
end