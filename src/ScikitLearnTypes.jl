abstract AbstractSVM
abstract AbstractSVC<:AbstractSVM

type SVC<:AbstractSVC
    kernel::Symbol
    gamma::Union{Float64,Symbol}
    cost::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    probability::Bool

    fit::Union{SVM, Void}
end

type NuSVC<:AbstractSVC
    kernel::Symbol
    gamma::Union{Float64,Symbol}
    nu::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool

    fit::Union{SVM, Void}
end

type OneClassSVM
    kernel::Symbol
    gamma::Union{Float64,Symbol}
    nu::Float64
    degree::Int32
    coef0::Float64
    tolerance::Float64
    shrinking::Bool

    fit::Union{SVM, Void}
end
