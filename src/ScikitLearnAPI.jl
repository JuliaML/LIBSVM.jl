import ScikitLearnBase: predict, predict_proba,
                        fit!, get_classes, transform, @declare_hyperparameters

function predict(model::Union{AbstractSVC, AbstractSVR} , X::AbstractArray)
    (p,d) = svmpredict(model.fit, X)
    return p
end

function predict(model::LinearSVC, X::AbstractArray)
    (p,d) = LIBLINEAR.linear_predict(model.fit, X)
    return p
end

function transform(model::OneClassSVM, X::AbstractArray)
    (p,d) = svmpredict(model.fit, X)
    return p
end

SVC(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
    weights = nothing, cost::Float64 = 1.0, degree::Int32 = Int32(3),
    coef0::Float64 = 0.0, tolerance::Float64 = .001,
    shrinking::Bool = true, probability::Bool = false,
    verbose::Bool = false) = SVC(kernel, gamma,
    weights, cost, degree, coef0, tolerance, shrinking,
    probability, verbose, nothing)

NuSVC(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
        weights = nothing, nu::Float64 = 0.5, cost::Float64 = 1.0,
        degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001, shrinking::Bool = true,
        verbose::Bool = false,) = NuSVC(kernel, gamma, weights, nu, cost,
            degree, coef0, tolerance, shrinking, verbose, nothing)

OneClassSVM(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
        nu::Float64 = 0.1, cost::Float64 = 1.0, degree::Int32 = Int32(3),
        coef0::Float64 = 0.0, tolerance::Float64 = .001,
        shrinking::Bool = true,
        verbose::Bool = false,) = OneClassSVM(kernel, gamma, nu, cost,
        degree, coef0, tolerance, shrinking, verbose, nothing)

NuSVR(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
        nu::Float64 = 0.5, cost::Float64 = 1.0, degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001, shrinking::Bool = true,
        verbose::Bool = false,) = NuSVR(kernel, gamma, nu, cost,
                    degree, coef0, tolerance, shrinking, verbose, nothing)

EpsilonSVR(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
        epsilon::Float64 = 0.1, cost::Float64 = 1.0,
        degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001, shrinking::Bool = true,
        verbose::Bool = false,) = EpsilonSVR(kernel, gamma, epsilon, cost,
                        degree, coef0, tolerance, shrinking, verbose, nothing)

LinearSVC(;solver = Linearsolver.L2R_L2LOSS_SVC_DUAL,
          weights::Union{Dict, Void} = nothing, tolerance::Float64=Inf,
          cost::Float64 = 1.0, p::Float64 = 0.1, bias::Float64 = -1.0,
          verbose::Bool = false) = LinearSVC(solver, weights, tolerance,
          cost, p, bias, verbose, nothing)

const SVMTYPES = Dict{Type, Symbol}(
            SVC => :CSVC,
            NuSVC => :nuSVC,
            OneClassSVM => :oneclassSVM,
            EpsilonSVR => :epsilonSVR,
            NuSVR => :nuSVR)

function fit!(model::Union{AbstractSVC,AbstractSVR}, X::AbstractMatrix, y::Vector=[])
    #Build arguments for calling svmtrain
    model.gamma == :auto && (model.gamma = 1.0/size(X, 1))
    kwargs = Tuple{Symbol, Any}[]
    push!(kwargs, (:svmtype, SVMTYPES[typeof(model)]))
    for fn in fieldnames(model)
        if fn != :fit
            push!(kwargs, (fn, getfield(model, fn)))
        end
    end

    model.fit = svmtrain(X, y; kwargs...)
    return(model)
end

function fit!(model::LinearSVC, X::AbstractMatrix, y::Vector)
    model.fit = LIBLINEAR.linear_train(y, X, solver_type =
        Linearsolver.SOLVERS[model.solver], weights = model.weights,
        C = model.cost, bias = model.bias, p = model.p, eps = model.tolerance,
        verbose = model.verbose
        )
    return(model)
end
