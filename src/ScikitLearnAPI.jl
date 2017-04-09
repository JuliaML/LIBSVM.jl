
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
        nu::Float64 = 0.5, cost::Float64 = 1.0,
        degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001,
        shrinking::Bool = true) = NuSVC(kernel, gamma, nu, cost,
        degree, coef0, tolerance, shrinking, nothing)

OneClassSVM(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
        nu::Float64 = 0.1, cost::Float64 = 1.0, degree::Int32 = Int32(3),
        coef0::Float64 = 0.0, tolerance::Float64 = .001,
        shrinking::Bool = true) = OneClassSVM(kernel, gamma, nu, cost,
        degree, coef0, tolerance, shrinking, nothing)

NuSVR(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
        nu::Float64 = 0.5, cost::Float64 = 1.0, degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001,
        shrinking::Bool = true) = NuSVR(kernel, gamma, nu, cost,
                        degree, coef0, tolerance, shrinking, nothing)

EpsilonSVR(;kernel::Symbol = :RBF, gamma::Union{Float64,Symbol} = :auto,
        epsilon::Float64 = 0.1, cost::Float64 = 1.0,
        degree::Int32 = Int32(3), coef0::Float64 = 0.,
        tolerance::Float64 = .001,
        shrinking::Bool = true) = EpsilonSVR(kernel, gamma, epsilon, cost,
                            degree, coef0, tolerance, shrinking, nothing)


const SVMTYPES = Dict{Type, Symbol}(
            SVC => :CSVC,
            NuSVC => :nuSVC,
            OneClassSVM => :oneclassSVM,
            EpsilonSVR => :epsilonSVR,
            NuSVR => :nuSVR)

function fit!(model::AbstractSVM, X::AbstractMatrix, y::Vector=[])
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
