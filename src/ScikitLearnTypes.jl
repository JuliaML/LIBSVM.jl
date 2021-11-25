
import ScikitLearnBase: BaseClassifier, BaseRegressor

abstract type AbstractSVC<:BaseClassifier end
abstract type AbstractSVR<:BaseRegressor end

mutable struct SVC<:AbstractSVC
    kernel
    gamma::Union{Float64,Symbol}
    weights::Union{Dict, Cvoid}
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    probability::Bool
    verbose::Bool

    fit::Union{SVM, Cvoid}
end

mutable struct NuSVC<:AbstractSVC
    kernel
    gamma::Union{Float64,Symbol}
    weights::Union{Dict, Cvoid}
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    verbose::Bool

    fit::Union{SVM, Cvoid}
end

mutable struct OneClassSVM<:AbstractSVC
    kernel
    gamma::Union{Float64,Symbol}
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    verbose::Bool

    fit::Union{SVM, Cvoid}
end

mutable struct NuSVR<:AbstractSVR
    kernel
    gamma::Union{Float64,Symbol}
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    verbose::Bool

    fit::Union{SVM, Cvoid}
end

mutable struct EpsilonSVR<:AbstractSVR
    kernel
    gamma::Union{Float64,Symbol}
    epsilon::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    verbose::Bool

    fit::Union{SVM, Cvoid}
end

"""
Linear SVM using LIBLINEAR
"""
mutable struct LinearSVC<:BaseClassifier
    solver::Linearsolver.LINEARSOLVER
    weights::Union{Dict, Cvoid}
    tolerance::Float64
    cost::Float64
    p::Float64
    bias::Float64
    verbose::Bool

    fit::Union{LIBLINEAR.LinearModel, Cvoid}
end

# Map types to Int for Libsvm C api
# https://github.com/cjlin1/libsvm/blob/557d85749aaf0ca83fd229af0f00e4f4cb7be85c/svm.h#L25
const SVMTYPES = Dict{Type, Int32}(
    SVC         => 0,
    NuSVC       => 1,
    OneClassSVM => 2,
    EpsilonSVR  => 3,
    NuSVR       => 4)
