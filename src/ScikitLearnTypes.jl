using Compat
import ScikitLearnBase: BaseClassifier, BaseRegressor

abstract type AbstractSVC<:BaseClassifier end
abstract type AbstractSVR<:BaseRegressor end

mutable struct SVC<:AbstractSVC
    kernel::Kernel.KERNEL
    gamma::Union{Float64,Symbol}
    weights::Union{Dict, Compat.Nothing}
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    probability::Bool
    verbose::Bool

    fit::Union{SVM, Compat.Nothing}
end

mutable struct NuSVC<:AbstractSVC
    kernel::Kernel.KERNEL
    gamma::Union{Float64,Symbol}
    weights::Union{Dict, Compat.Nothing}
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    verbose::Bool

    fit::Union{SVM, Compat.Nothing}
end

mutable struct OneClassSVM<:AbstractSVC
    kernel::Kernel.KERNEL
    gamma::Union{Float64,Symbol}
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    verbose::Bool

    fit::Union{SVM, Compat.Nothing}
end

mutable struct NuSVR<:AbstractSVR
    kernel::Kernel.KERNEL
    gamma::Union{Float64,Symbol}
    nu::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    verbose::Bool

    fit::Union{SVM, Compat.Nothing}
end

mutable struct EpsilonSVR<:AbstractSVR
    kernel::Kernel.KERNEL
    gamma::Union{Float64,Symbol}
    epsilon::Float64
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    verbose::Bool

    fit::Union{SVM, Compat.Nothing}
end

"""
Linear SVM using LIBLINEAR
"""
mutable struct LinearSVC<:BaseClassifier
    solver::Linearsolver.LINEARSOLVER
    weights::Union{Dict, Compat.Nothing}
    tolerance::Float64
    cost::Float64
    p::Float64
    bias::Float64
    verbose::Bool

    fit::Union{LIBLINEAR.LinearModel, Compat.Nothing}
end

#Map types to Int for Libsvm C api
const SVMTYPES = Dict{Type, Int32}(
            SVC => 0,
            NuSVC => 1,
            OneClassSVM => 2,
            EpsilonSVR => 3,
            NuSVR => 4)
