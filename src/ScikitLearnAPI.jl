import ScikitLearnBase: predict, predict_proba,
                        fit!, get_classes, transform,
                        @declare_hyperparameters, get_params,
                        set_params!

function predict(model::Union{AbstractSVC, AbstractSVR} , X::AbstractArray)
    (p,d) = svmpredict(model.fit, X')
    return p
end

function predict(model::LinearSVC, X::AbstractArray)
    (p,d) = LIBLINEAR.linear_predict(model.fit, X')
    return p
end

function transform(model::OneClassSVM, X::AbstractArray)
    (p,d) = svmpredict(model.fit, X')
    return p
end

SVC(;kernel::Kernel.KERNEL = Kernel.RadialBasis, gamma::Union{Float64,Symbol} = :auto,
    weights = nothing, cost::Float64 = 1.0, degree::Int32 = Int32(3),
    coef0::Float64 = 0.0, tolerance::Float64 = .001,
    shrinking::Bool = true, probability::Bool = false,
    verbose::Bool = false) = SVC(kernel, gamma,
    weights, cost, degree, coef0, tolerance, shrinking,
    probability, verbose, nothing)
@declare_hyperparameters(SVC, [:kernel, :gamma, :weights, :cost, :degree, :coef0, :tolerance])

NuSVC(;kernel::Kernel.KERNEL = Kernel.RadialBasis, gamma::Union{Float64,Symbol} = :auto,
        weights = nothing, nu::Float64 = 0.5, cost::Float64 = 1.0,
        degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001, shrinking::Bool = true,
        verbose::Bool = false,) = NuSVC(kernel, gamma, weights, nu, cost,
            degree, coef0, tolerance, shrinking, verbose, nothing)
@declare_hyperparameters(NuSVC, [:kernel, :gamma, :weights, :nu, :cost, :degree, :coef0, :tolerance])

OneClassSVM(;kernel::Kernel.KERNEL = Kernel.RadialBasis, gamma::Union{Float64,Symbol} = :auto,
        nu::Float64 = 0.1, cost::Float64 = 1.0, degree::Int32 = Int32(3),
        coef0::Float64 = 0.0, tolerance::Float64 = .001,
        shrinking::Bool = true,
        verbose::Bool = false,) = OneClassSVM(kernel, gamma, nu, cost,
        degree, coef0, tolerance, shrinking, verbose, nothing)
@declare_hyperparameters(OneClassSVM, [:kernel, :gamma, :nu, :cost, :degree, :coef0, :tolerance])

NuSVR(;kernel::Kernel.KERNEL = Kernel.RadialBasis, gamma::Union{Float64,Symbol} = :auto,
        nu::Float64 = 0.5, cost::Float64 = 1.0, degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001, shrinking::Bool = true,
        verbose::Bool = false,) = NuSVR(kernel, gamma, nu, cost,
                    degree, coef0, tolerance, shrinking, verbose, nothing)
@declare_hyperparameters(NuSVR, [:kernel, :gamma, :nu, :cost, :degree, :coef0, :tolerance])

EpsilonSVR(;kernel::Kernel.KERNEL = Kernel.RadialBasis, gamma::Union{Float64,Symbol} = :auto,
        epsilon::Float64 = 0.1, cost::Float64 = 1.0,
        degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001, shrinking::Bool = true,
        verbose::Bool = false,) = EpsilonSVR(kernel, gamma, epsilon, cost,
                        degree, coef0, tolerance, shrinking, verbose, nothing)
@declare_hyperparameters(EpsilonSVR, [:kernel, :gamma, :epsilon, :cost, :degree, :coef0, :tolerance])

LinearSVC(;solver = Linearsolver.L2R_L2LOSS_SVC_DUAL,
          weights::Union{Dict, Compat.Nothing} = nothing, tolerance::Float64=Inf,
          cost::Float64 = 1.0, p::Float64 = 0.1, bias::Float64 = -1.0,
          verbose::Bool = false) = LinearSVC(solver, weights, tolerance,
          cost, p, bias, verbose, nothing)
@declare_hyperparameters(LinearSVC, [:solver, :weights, :tolerance, :cost, :p, :bias])

function fit!(model::Union{AbstractSVC,AbstractSVR}, X::AbstractMatrix, y::Vector=[])
    #Build arguments for calling svmtrain
    model.gamma == :auto && (model.gamma = 1.0/size(X', 1))
    kwargs = Tuple{Symbol, Any}[]
    push!(kwargs, (:svmtype, typeof(model)))
    for fn in fieldnames(typeof(model))
        if fn != :fit
            push!(kwargs, (fn, getfield(model, fn)))
        end
    end

    model.fit = svmtrain(X', y; kwargs...)
    return(model)
end

function set_params(model::Union{AbstractSVC,AbstractSVR, LinearSVC}; kwargs...)
    for arg in kwargs
        setfield!(model, arg...)
    end
    return model
end

function get_params(model::Union{AbstractSVC,AbstractSVR, LinearSVC})
    params = Dict{Symbol, Any}()
    for fn in fieldnames(model)
        if fn != :fit
            params[fn] = getfield(model, fn)
        end
    end
    return params
end

function fit!(model::LinearSVC, X::AbstractMatrix, y::Vector)
    model.fit = LIBLINEAR.linear_train(y, X', solver_type = Int32(model.solver),
    weights = model.weights, C = model.cost, bias = model.bias,
    p = model.p, eps = model.tolerance, verbose = model.verbose)
    return(model)
end
