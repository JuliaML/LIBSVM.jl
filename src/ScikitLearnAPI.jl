
function predict(model::AbstractSVM, X::AbstractArray)
    (p,d) = svmpredict(model.fit, X)
    return p
end

function transform(model::OneClassSVM, X::AbstractArray)
    (p,d) = svmpredict(model.fit, X)
    return p
end

SVC(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
    cost::Float64 = 1.0, degree::Int32 = Int32(3),
    coef0::Float64 = 0.0, tolerance::Float64 = .001,
    shrinking::Bool = true,
    probability::Bool = false) = SVC(kernel, gamma, cost,
    degree, coef0, tolerance, shrinking, probability, nothing)

NuSVC(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
        nu::Float64 = 0.5,
        degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001,
        shrinking::Bool = true) = NuSVC(kernel, gamma, nu,
        degree, coef0, tolerance, shrinking, nothing)

OneClassSVM(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
                nu::Float64 = 0.1, degree::Int32 = Int32(3),
                coef0::Float64 = 0.0, tolerance::Float64 = .001,
                shrinking::Bool = true) = OneClassSVM(kernel, gamma, nu,
                degree, coef0, tolerance, shrinking, nothing)

function fit!(model::SVC, X::AbstractMatrix, y::Vector)
    model.gamma == :auto && (model.gamma = 1.0/size(X, 1))
    model.fit = svmtrain(X, y, svmtype = :CSVC, kernel = model.kernel,
            gamma = model.gamma,
            cost = model.cost, coef0 = model.coef0,
            degree = model.degree, tolerance = model.tolerance,
            shrinking = model.shrinking, probability = model.probability)
    return(model)
end

function fit!(model::NuSVC, X::AbstractMatrix, y::Vector)
    model.gamma == :auto && (model.gamma = 1.0/size(X, 1))
    model.fit = svmtrain(X, y,
        svmtype = :nuSVC, kernel = model.kernel,
        gamma = model.gamma,
        nu = model.nu, coef0 = model.coef0,
        degree = model.degree, tolerance = model.tolerance,
        shrinking = model.shrinking)
    return(model)
end

function fit!(model::OneClassSVM, X::AbstractMatrix)
    model.gamma == :auto && (model.gamma = 1.0/size(X, 1))
    model.fit = svmtrain(X,
        svmtype = :oneclassSVM, kernel = model.kernel,
        gamma = model.gamma,
        nu = model.nu, coef0 = model.coef0,
        degree = model.degree, tolerance = model.tolerance,
        shrinking = model.shrinking)
    return(model)
end
