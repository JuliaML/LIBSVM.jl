module LIBSVM

import LIBLINEAR

using LinearAlgebra
using libsvm_jll

export svmtrain, svmpredict, fit!, predict, transform,
       SVC, NuSVC, OneClassSVM, NuSVR, EpsilonSVR, LinearSVC,
       Linearsolver, Kernel

include("LibSVMtypes.jl")
include("constants.jl")
include("libcalls.jl")
include("nodes.jl")

struct SupportVectors{T<:AbstractVector,U<:AbstractMatrix}
    l::Int32
    nSV::Vector{Int32}
    y::T
    X::U
    indices::Vector{Int32}
    SVnodes::Vector{SVMNode}
end

function SupportVectors(smc::SVMModel, y, X)
    sv_indices = Array{Int32}(undef, smc.l)
    unsafe_copyto!(pointer(sv_indices), smc.sv_indices, smc.l)
    nodes = [unsafe_load(unsafe_load(smc.SV, i)) for i in 1:smc.l]

    if smc.nSV != C_NULL
        nSV = Array{Int32}(undef, smc.nr_class)
        unsafe_copyto!(pointer(nSV), smc.nSV, smc.nr_class)
    else
        nSV = Array{Int32}(undef, 0)
    end

    yi = smc.param.svm_type == 2 ? Float64[] : y[sv_indices]

    SupportVectors(smc.l, nSV, yi , X[:,sv_indices], sv_indices, nodes)
end

struct SVM{T, K}
    SVMtype::Type
    kernel::K
    weights::Union{Dict{T,Float64},Cvoid}
    nfeatures::Int
    nclasses::Int32
    labels::Vector{T}
    libsvmlabel::Vector{Int32}
    libsvmweight::Vector{Float64}
    libsvmweightlabel::Vector{Int32}
    SVs::SupportVectors
    coef0::Float64
    coefs::Array{Float64,2}
    probA::Vector{Float64}
    probB::Vector{Float64}

    rho::Vector{Float64}
    degree::Int32
    gamma::Float64
    cache_size::Float64
    tolerance::Float64
    cost::Float64
    nu::Float64
    epsilon::Float64
    shrinking::Bool
    probability::Bool
end

function SVM(smc::SVMModel, y, X, weights, labels, svmtype, kernel)
    svs = SupportVectors(smc, y, X)
    coefs = zeros(smc.l, smc.nr_class-1)
    for k in 1:(smc.nr_class-1)
        unsafe_copyto!(pointer(coefs, (k-1)*smc.l +1 ), unsafe_load(smc.sv_coef, k), smc.l)
    end
    k = smc.nr_class
    rs = Int(k*(k-1)/2)
    rho = Vector{Float64}(undef, rs)
    unsafe_copyto!(pointer(rho), smc.rho, rs)

    if smc.label == C_NULL
        libsvmlabel = Vector{Int32}(undef, 0)
    else
        libsvmlabel = Vector{Int32}(undef, k)
        unsafe_copyto!(pointer(libsvmlabel), smc.label, k)
    end

    if smc.probA == C_NULL
        probA = Float64[]
        probB = Float64[]
    else
        probA = Vector{Float64}(undef, rs)
        probB = Vector{Float64}(undef, rs)
        unsafe_copyto!(pointer(probA), smc.probA, rs)
        unsafe_copyto!(pointer(probB), smc.probB, rs)
    end

    # Weights
    nw = smc.param.nr_weight
    libsvmweight = Array{Float64}(undef, nw)
    libsvmweight_label = Array{Int32}(undef, nw)

    if nw > 0
        unsafe_copyto!(pointer(libsvmweight), smc.param.weight, nw)
        unsafe_copyto!(pointer(libsvmweight_label), smc.param.weight_label, nw)
    end

    if kernel == Kernel.Precomputed || isa(kernel, Function)
        # in the precomputed case, each data point we want to predict has
        # l gram matrix entries (≈ features), where l is the number of training vectors.
        nfeatures = size(X, 2)
    else
        nfeatures = size(X,1)
    end

    SVM(svmtype, kernel, weights, nfeatures,
        smc.nr_class, labels, libsvmlabel, libsvmweight, libsvmweight_label,
        svs, smc.param.coef0, coefs, probA, probB,
        rho, smc.param.degree,
        smc.param.gamma, smc.param.cache_size, smc.param.eps,
        smc.param.C, smc.param.nu, smc.param.p, Bool(smc.param.shrinking),
        Bool(smc.param.probability))
end

#Keep data for SVMModel to prevent GC
struct SVMData
    coefs::Vector{Ptr{Float64}}
    nodes::Array{SVMNode}
    nodeptrs::Array{Ptr{SVMNode}}
end

"""Convert SVM model to libsvm struct for prediction"""
function svmmodel(mod::SVM)
    svm_type = Int32(SVMTYPES[mod.SVMtype])
    if isa(mod.kernel, Function)
       kernel = Int32(Kernel.Precomputed) 
    else
        kernel = Int32(mod.kernel)
    end

    param = SVMParameter(svm_type, kernel, mod.degree, mod.gamma,
                        mod.coef0, mod.cache_size, mod.tolerance, mod.cost,
                        length(mod.libsvmweight), pointer(mod.libsvmweightlabel), pointer(mod.libsvmweight),
                        mod.nu, mod.epsilon, Int32(mod.shrinking), Int32(mod.probability))

    n,m = size(mod.coefs)
    sv_coef = Vector{Ptr{Float64}}(undef, m)
    for i in 1:m
        sv_coef[i] = pointer(mod.coefs, (i-1)*n+1)
    end

    if isa(mod.kernel, Function) || mod.kernel == Kernel.Precomputed
        # This is necessary to construct SupportVectors correctly
        nodes, ptrs = LIBSVM.instances2nodes(mod.SVs.indices')
    else
        nodes, ptrs = LIBSVM.instances2nodes(mod.SVs.X)
    end
    data = SVMData(sv_coef, nodes, ptrs)

    cmod = SVMModel(param, mod.nclasses, mod.SVs.l, pointer(data.nodeptrs), pointer(data.coefs),
                pointer(mod.rho), pointer(mod.probA), pointer(mod.probB), pointer(mod.SVs.indices),
                pointer(mod.libsvmlabel),
                pointer(mod.SVs.nSV), Int32(1))

    return cmod, data
end

const libsvm_version = Ref{Cint}(0)
const noprint_ptr = Ref{Ptr{Nothing}}(C_NULL)

function __init__()
    libsvm_version[] = unsafe_load(cglobal((:libsvm_version, libsvm), Cint))
    noprint_ptr[] = @cfunction(noprint, Cvoid, (Ptr{UInt8},))
    libsvm_set_verbose(false)
end


function grp2idx(::Type{S}, labels::AbstractVector,
    label_dict::Dict{T, Int32}, reverse_labels::Vector{T}) where {T, S <: Real}

    idx = Array{S}(undef, length(labels))
    nextkey = length(reverse_labels) + 1
    for i = 1:length(labels)
        key = labels[i]
        if (idx[i] = get(label_dict, key, nextkey)) == nextkey
            label_dict[key] = nextkey
            push!(reverse_labels, key)
            nextkey += 1
        end
    end
    idx
end

function indices_and_weights(labels::AbstractVector{T},
        instances::AbstractMatrix{U},
        weights::Union{Dict{T, Float64}, Cvoid}=nothing) where {T, U<:Real}
    label_dict = Dict{T, Int32}()
    reverse_labels = Array{T}(undef, 0)
    idx = grp2idx(Float64, labels, label_dict, reverse_labels)

    # Construct SVMParameter
    if weights == nothing || length(weights) == 0
        weight_labels = Int32[]
        weights = Float64[]
    else
        weight_labels = grp2idx(Int32, collect(keys(weights)), label_dict,
            reverse_labels)
        weights = collect(values(weights))
    end

    (idx, reverse_labels, weights, weight_labels)
end

function set_num_threads(nt::Integer)
    if nt == 0
        nt = parse(Int64, get(ENV, "OMP_NUM_THREADS", "1"))
    end
    if nt < 0
        nt = libsvm_get_max_threads()
    end
    libsvm_set_num_threads(nt)
end

function check_train_input(X, y, kernel)
    if kernel == Kernel.Precomputed && !issymmetric(X)
        throw(ArgumentError("The input matrix must be symmetric"))
    end

    if size(y, 1) != size(X, 2)
        throw(DimensionMismatch("Size of second dimension of training instance
                                matrix ($(size(X, 2))) does not match length of
                                labels ($(size(y, 1)))"))
    end
end

function data2gram(kernel_function::Function, X)
    n1 = axes(X, 2)
    gram = Array{Float64}(undef, n1.stop, n1.stop)
    for i in n1
        for j in 1:i
            if i==j
                gram[i,i] = kernel_function(X[:,i], X[:,i])
            else
                interm = kernel_function(X[:, i], X[:, j])
                gram[i,j] = interm
                gram[j,i] = interm
            end
        end
    end
    return gram
end


function data2gram(kernel::Function, T, SVs::SupportVectors, nfeatures::Int)
    n2 = axes(T, 2) # number of datapoints to predict
    gram = zeros(nfeatures, size(T, 2))

    for (i, idx) in enumerate(SVs.indices)
        for j in n2
            x = SVs.X[:, i]
            gram[idx,j] = kernel(x, T[:,j])
        end
    end
    return gram
end


"""
    svmtrain(
        X::AbstractMatrix{U}, y::AbstractVector{T} = [];
        svmtype::Type = SVC,
        kernel::Kernel.KERNEL = Kernel.RadialBasis,
        degree::Integer = 3,
        gamma::Float64 = 1.0/size(X, 1),
        coef0::Float64 = 0.0,
        cost::Float64=1.0,
        nu::Float64 = 0.5,
        epsilon::Float64 = 0.1,
        tolerance::Float64 = 0.001,
        shrinking::Bool = true,
        probability::Bool = false,
        weights::Union{Dict{T,Float64},Cvoid} = nothing,
        cachesize::Float64 = 200.0,
        verbose::Bool = false
    ) where {T,U<:Real}

Train Support Vector Machine using LIBSVM using response vector `y`
and training data `X`. The shape of `X` needs to be `(nfeatures, nsamples)` or
`(nsamples, nsamples)` in the case of precomputed kernel (see below).
For one-class SVM use only `X`.

# Arguments

* `svmtype::Type = LIBSVM.SVC`: Type of SVM to train `SVC` (for C-SVM), `NuSVC`
    `OneClassSVM`, `EpsilonSVR` or `NuSVR`. Defaults to `OneClassSVM` if
    `y` is not used.
* `kernel::Kernels.KERNEL = Kernel.RadialBasis`: Model kernel `Linear`, `Polynomial`,
    `RadialBasis`, `Sigmoid` or `Precomputed`.
* `degree::Integer = 3`: Kernel degree. Used for polynomial kernel
* `gamma::Float64 = 1.0/size(X, 1)` : γ for kernels
* `coef0::Float64 = 0.0`: parameter for sigmoid and polynomial kernel
* `cost::Float64 = 1.0`: cost parameter C of C-SVC, epsilon-SVR, and nu-SVR
* `nu::Float64 = 0.5`: parameter nu of nu-SVC, one-class SVM, and nu-SVR
* `epsilon::Float64 = 0.1`: epsilon in loss function of epsilon-SVR
* `tolerance::Float64 = 0.001`: tolerance of termination criterion
* `shrinking::Bool = true`: whether to use the shrinking heuristics
* `probability::Bool = false`: whether to train a SVC or SVR model for probability estimates
* `weights::Union{Dict{T, Float64}, Cvoid} = nothing`: dictionary of class weights
* `cachesize::Float64 = 200.0`: cache memory size in MB
* `verbose::Bool = false`: print training output from LIBSVM if true
* `nt::Integer = 0`: number of OpenMP cores to use, if 0 it is set to OMP_NUM_THREADS, if negative it is set to the max number of threads

Consult LIBSVM documentation for advice on the choise of correct
parameters and model tuning.

# Precomputed kernel

In the case of precomputed kernel, the input matrix `X` should be a
(symmetric) matrix of shape `(n, n)` where column `i` contains values
`[K(x_i, x_1), K(x_i, x_2), ..., K(x_i, x_n)]`. For example, if matrix
`M` contains instances in its columns, then `M' * M` produces the correct
input matrix `X` for linear kernel.
"""
function svmtrain(
        X::AbstractMatrix{U}, y::AbstractVector{T} = [];
        svmtype::Type = SVC,
        kernel::Union{Kernel.KERNEL, Function} = Kernel.RadialBasis,
        degree::Integer = 3,
        gamma::Float64 = 1.0 / size(X, 1),
        coef0::Float64 = 0.0,
        cost::Float64 = 1.0,
        nu::Float64 = 0.5,
        epsilon::Float64 = 0.1,
        tolerance::Float64 = 0.001,
        shrinking::Bool = true,
        probability::Bool = false,
        weights::Union{Dict{T,Float64},Cvoid} = nothing,
        cachesize::Float64 = 200.0,
        verbose::Bool = false,
        nt::Integer = 1) where {T,U<:Real}
    set_num_threads(nt)

    isempty(y) && (svmtype = OneClassSVM)

    _svmtype = SVMTYPES[svmtype]

    if isa(kernel, Function)
        _kernel = Int32(Kernel.Precomputed)
    else
        _kernel = Int32(kernel)
    end

    wts = weights

    if svmtype ∈ (EpsilonSVR, NuSVR)
        check_train_input(X, y, kernel)
        idx = y
        weight_labels = Int32[]
        weights = Float64[]
        reverse_labels = Float64[]
    elseif svmtype == OneClassSVM
        idx = Float64[]
        weight_labels = Int32[]
        weights = Float64[]
        reverse_labels = Bool[]
    else
        check_train_input(X, y, kernel)
        idx, reverse_labels, weights, weight_labels = indices_and_weights(y, X, weights)
    end

    param = SVMParameter(
        _svmtype, _kernel, Int32(degree), Float64(gamma),
        coef0, cachesize, tolerance, cost, Int32(length(weights)),
        pointer(weight_labels), pointer(weights), nu, epsilon, Int32(shrinking),
        Int32(probability))

    ninstances = size(X, 2)

    # Construct SVMProblem
    if isa(kernel, Function)
        gram = data2gram(kernel, X)
        (nodes, nodeptrs) = gram2nodes(gram)
    else
        if kernel == Kernel.Precomputed
            (nodes, nodeptrs) = gram2nodes(X)
        else
            (nodes, nodeptrs) = instances2nodes(X)
        end
    end
    problem = SVMProblem(Int32(ninstances), pointer(idx), pointer(nodeptrs))

    libsvm_set_verbose(verbose)

    @GC.preserve nodes begin
        # Validate the given parameters
        libsvm_check_parameter(problem, param)

        ptr_model = libsvm_train(problem, param)
    end

    svm = SVM(unsafe_load(ptr_model), y, X, wts, reverse_labels, svmtype,
              kernel)

    libsvm_free_model(ptr_model)

    return svm
end

"""
    svmpredict(model::SVM{T}, X::AbstractMatrix{U}) where {T,U<:Real}

Predict values using `model` based on data `X`.
The shape of `X` needs to be `(nfeatures, nsamples)` (for precomputed kernel
see below).
The method returns tuple `(predictions, decisionvalues)`.

# Precomputed kernel

In the case of precomputed kernel, the input matrix `X` should be of shape `(l,
n)`, where `l` is the number of training instances and `n` is the number of
testing instances. Column `i` of `X` should contain values `[K(t_i, x_1),
K(t_i, x_2), ..., K(t_i, x_l)]`, where `t_i` is `i`-th testing instance and
`x_j` is `j`-th training instance. For linear kernel, `M' * T` produces the
correct matrix, where columns of `M` contain the training instances and columns
of `T` the testing instances.
"""
function svmpredict(model::SVM{T}, X::AbstractMatrix{U}; nt::Integer = 0) where {T,U<:Real}
    set_num_threads(nt)

    if model.kernel != Kernel.Precomputed && !isa(model.kernel, Function) && size(X, 1) != model.nfeatures
        throw(DimensionMismatch("Model has $(model.nfeatures) but $(size(X, 1)) provided"))
    end

    ninstances = size(X, 2)
    if isa(model.kernel, Function)
        gram = data2gram(model.kernel, X, model.SVs, model.nfeatures)
        (nodes, nodeptrs) = gram2nodes(gram)
    elseif model.kernel == Kernel.Precomputed
        (nodes, nodeptrs) = gram2nodes(X)
    else
        (nodes, nodeptrs) = instances2nodes(X)
    end

    pred = if model.SVMtype == OneClassSVM
        BitArray(undef, ninstances)
    else
        Array{T}(undef, ninstances)
    end

    nlabels = model.nclasses

    if model.SVMtype == EpsilonSVR || model.SVMtype == NuSVR || model.SVMtype == OneClassSVM || model.probability
        decvalues = zeros(Float64, nlabels, ninstances)
    else
        dcols = max(Int64(nlabels*(nlabels-1)/2), 2)
        decvalues = zeros(Float64, dcols, ninstances)
    end

    libsvm_set_verbose(false)

    cmod, data = svmmodel(model)

    predf = ifelse(model.probability, libsvm_predict_probability, libsvm_predict_values)
    decode = model.SVMtype ∈ (EpsilonSVR, NuSVR) ? identity :
             model.SVMtype == OneClassSVM        ? >(0)     :
             (x -> model.labels[round(Int, x)])

    @GC.preserve model nodes data begin
        # create function barrier, since `pred` is type unstable
        svmpredict_fill!(predf, decode, cmod, pred, decvalues, nodeptrs, nlabels)
    end

    (pred, decvalues)
end

function svmpredict_fill!(predf, decode, cmod, pred, decvalues, nodeptrs, nlabels)
    for i ∈ eachindex(pred)
        @inbounds pred[i] = decode(predf(cmod, nodeptrs[i], Ref(decvalues, nlabels * (i - 1) + 1)))
    end
end

include("ScikitLearnTypes.jl")
include("ScikitLearnAPI.jl")

end
