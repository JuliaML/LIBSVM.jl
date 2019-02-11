__precompile__()
module LIBSVM
import LIBLINEAR
using Compat
using SparseArrays
using Libdl

export svmtrain, svmpredict, fit!, predict, transform,
        SVC, NuSVC, OneClassSVM, NuSVR, EpsilonSVR, LinearSVC,
        Linearsolver, Kernel

include("LibSVMtypes.jl")
include("constants.jl")

verbosity = false

struct SupportVectors{T, U}
    l::Int32
    nSV::Vector{Int32}
    y::Vector{T}
    X::AbstractMatrix{U}
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

    SupportVectors(smc.l, nSV, yi , X[:,sv_indices],
                        sv_indices, nodes)
end

struct SVM{T}
    SVMtype::Type
    kernel::Kernel.KERNEL
    weights::Union{Dict{T, Float64}, Compat.Nothing}
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

function SVM(smc::SVMModel, y::T, X, weights, labels, svmtype, kernel) where T
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

    #Weights
    nw = smc.param.nr_weight
    libsvmweight = Array{Float64}(undef, nw)
    libsvmweight_label = Array{Int32}(undef, nw)

    if nw > 0
        unsafe_copyto!(pointer(libsvmweight), smc.param.weight, nw)
        unsafe_copyto!(pointer(libsvmweight_label), smc.param.weight_label, nw)
    end

    SVM(svmtype, kernel, weights, size(X,1),
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
    kernel = Int32(mod.kernel)

    param = SVMParameter(svm_type, kernel, mod.degree, mod.gamma,
                        mod.coef0, mod.cache_size, mod.tolerance, mod.cost,
                        length(mod.libsvmweight), pointer(mod.libsvmweightlabel), pointer(mod.libsvmweight),
                        mod.nu, mod.epsilon, Int32(mod.shrinking), Int32(mod.probability))

    n,m = size(mod.coefs)
    sv_coef = Vector{Ptr{Float64}}(undef, m)
    for i in 1:m
        sv_coef[i] = pointer(mod.coefs, (i-1)*n+1)
    end

    nodes, ptrs = LIBSVM.instances2nodes(mod.SVs.X)
    data = SVMData(sv_coef, nodes, ptrs)

    cmod = SVMModel(param, mod.nclasses, mod.SVs.l, pointer(data.nodeptrs), pointer(data.coefs),
                pointer(mod.rho), pointer(mod.probA), pointer(mod.probB), pointer(mod.SVs.indices),
                pointer(mod.libsvmlabel),
                pointer(mod.SVs.nSV), Int32(1))

    return cmod, data
end


function svmprint(str::Ptr{UInt8})
    if verbosity::Bool
        print(unsafe_string(str))
    end
    nothing
end


let libsvm = C_NULL
    global get_libsvm
    function get_libsvm()
        if libsvm == C_NULL
            if Sys.iswindows()
                libsvm = Libdl.dlopen(joinpath(dirname(@__FILE__),  "../deps",
                "libsvm.dll"))
            else
                libsvm = Libdl.dlopen(joinpath(dirname(@__FILE__),  "../deps",
                "libsvm.so.2"))
            end
            ccall(Libdl.dlsym(libsvm, :svm_set_print_string_function), Compat.Nothing,
                (Ptr{Compat.Nothing},), @cfunction(svmprint, Compat.Nothing, (Ptr{UInt8},) ))
        end
        libsvm
    end
end

macro cachedsym(symname)
    cached = gensym()
    quote
        let $cached = C_NULL
            global ($symname)
            ($symname)() = ($cached) == C_NULL ?
                ($cached = Libdl.dlsym(get_libsvm(), $(string(symname)))) : $cached
        end
    end
end


@cachedsym svm_train
@cachedsym svm_predict
@cachedsym svm_predict_values
@cachedsym svm_predict_probability
@cachedsym svm_free_model_content
@cachedsym svm_set_num_threads
@cachedsym svm_get_max_threads

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

function instances2nodes(instances::AbstractMatrix{U}) where U<:Real
    nfeatures = size(instances, 1)
    ninstances = size(instances, 2)
    nodeptrs = Array{Ptr{SVMNode}}(undef, ninstances)
    nodes = Array{SVMNode}(undef, nfeatures + 1, ninstances)

    for i=1:ninstances
        k = 1
        for j=1:nfeatures
            nodes[k, i] = SVMNode(Int32(j), Float64(instances[j, i]))
            k += 1
        end
        nodes[k, i] = SVMNode(Int32(-1), NaN)
        nodeptrs[i] = pointer(nodes, (i-1)*(nfeatures+1)+1)
    end

    (nodes, nodeptrs)
end

function instances2nodes(instances::SparseMatrixCSC{U}) where U<:Real
    ninstances = size(instances, 2)
    nodeptrs = Array{Ptr{SVMNode}}(undef, ninstances)
    nodes = Array{SVMNode}(undef, nnz(instances)+ninstances)

    j = 1
    k = 1
    for i=1:ninstances
        nodeptrs[i] = pointer(nodes, k)
        while j < instances.colptr[i+1]
            val = instances.nzval[j]
            nodes[k] = SVMNode(Int32(instances.rowval[j]), Float64(val))
            k += 1
            j += 1
        end
        nodes[k] = SVMNode(Int32(-1), NaN)
        k += 1
    end

    (nodes, nodeptrs)
end


function indices_and_weights(labels::AbstractVector{T},
        instances::AbstractMatrix{U},
        weights::Union{Dict{T, Float64}, Compat.Nothing}=nothing) where {T, U<:Real}
    label_dict = Dict{T, Int32}()
    reverse_labels = Array{T}(undef, 0)
    idx = grp2idx(Float64, labels, label_dict, reverse_labels)


    if length(labels) != size(instances, 2)
        error("""Size of second dimension of training instance matrix
        ($(size(instances, 2))) does not match length of labels
        ($(length(labels)))""")
    end

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
        if haskey(ENV,"OMP_NUM_THREADS")
            nt = parse(Int64, ENV["OMP_NUM_THREADS"])
        else
            nt = 1
        end
    end

    if nt < 0
        nt = ccall(svm_get_max_threads(), Cint, ())
    end

    ccall(svm_set_num_threads(), Compat.Nothing, (Cint,), nt)
end

"""
```julia
svmtrain{T, U<:Real}(X::AbstractMatrix{U}, y::AbstractVector{T}=[];
    svmtype::Type=SVC, kernel::Kernel.KERNEL=Kernel.RadialBasis, degree::Integer=3,
    gamma::Float64=1.0/size(X, 1), coef0::Float64=0.0,
    cost::Float64=1.0, nu::Float64=0.5, epsilon::Float64=0.1,
    tolerance::Float64=0.001, shrinking::Bool=true,
    probability::Bool=false, weights::Union{Dict{T, Float64}, Compat.Nothing}=nothing,
    cachesize::Float64=200.0, verbose::Bool=false)
```
Train Support Vector Machine using LIBSVM using response vector `y`
and training data `X`. The shape of `X` needs to be (nfeatures, nsamples).
For one-class SVM use only `X`.

# Arguments

* `svmtype::Type=LIBSVM.SVC`: Type of SVM to train `SVC` (for C-SVM), `NuSVC`
    `OneClassSVM`, `EpsilonSVR` or `NuSVR`. Defaults to `OneClassSVM` if
    `y` is not used.
* `kernel::Kernels.KERNEL=Kernel.RadialBasis`: Model kernel `Linear`, `polynomial`,
    `RadialBasis`, `Sigmoid` or `Precomputed`.
* `degree::Integer=3`: Kernel degree. Used for polynomial kernel
* `gamma::Float64=1.0/size(X, 1)` : γ for kernels
* `coef0::Float64=0.0`: parameter for sigmoid and polynomial kernel
* `cost::Float64=1.0`: cost parameter C of C-SVC, epsilon-SVR, and nu-SVR
* `nu::Float64=0.5`: parameter nu of nu-SVC, one-class SVM, and nu-SVR
* `epsilon::Float64=0.1`: epsilon in loss function of epsilon-SVR
* `tolerance::Float64=0.001`: tolerance of termination criterion
* `shrinking::Bool=true`: whether to use the shrinking heuristics
* `probability::Bool=false`: whether to train a SVC or SVR model for probability estimates
* `weights::Union{Dict{T, Float64}, Compat.Nothing}=nothing`: dictionary of class weights
* `cachesize::Float64=100.0`: cache memory size in MB
* `verbose::Bool=false`: print training output from LIBSVM if true
* `nt::Integer=0`: number of OpenMP cores to use, if 0 it is set to OMP_NUM_THREADS, if negative it is set to the max number of threads

Consult LIBSVM documentation for advice on the choise of correct
parameters and model tuning.
"""
function svmtrain(X::AbstractMatrix{U}, y::AbstractVector{T} = [];
        svmtype::Type=SVC, kernel::Kernel.KERNEL = Kernel.RadialBasis,
        degree::Integer=3, gamma::Float64=1.0/size(X, 1), coef0::Float64=0.0,
        cost::Float64=1.0, nu::Float64=0.5, epsilon::Float64=0.1,
        tolerance::Float64=0.001, shrinking::Bool=true,
        probability::Bool=false, weights::Union{Dict{T, Float64}, Compat.Nothing}=nothing,
        cachesize::Float64=200.0, verbose::Bool=false, nt::Integer=1) where {T, U<:Real}
    global verbosity

    set_num_threads(nt)

    isempty(y) && (svmtype = OneClassSVM)

    _svmtype = SVMTYPES[svmtype]
    _kernel = Int32(kernel)
    wts = weights

    if svmtype == EpsilonSVR || svmtype == NuSVR
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
        (idx, reverse_labels, weights, weight_labels) = indices_and_weights(y,
            X, weights)
    end

    param = Array{SVMParameter}(undef, 1)
    param[1] = SVMParameter(_svmtype, _kernel, Int32(degree), Float64(gamma),
        coef0, cachesize, tolerance, cost, Int32(length(weights)),
        pointer(weight_labels), pointer(weights), nu, epsilon, Int32(shrinking),
        Int32(probability))

    # Construct SVMProblem
    (nodes, nodeptrs) = instances2nodes(X)
    problem = SVMProblem[SVMProblem(Int32(size(X, 2)), pointer(idx),
        pointer(nodeptrs))]

    verbosity = verbose
    mod = ccall(svm_train(), Ptr{SVMModel}, (Ptr{SVMProblem},
        Ptr{SVMParameter}), problem, param)
    svm = SVM(unsafe_load(mod), y, X, wts, reverse_labels,
        svmtype, kernel)

    ccall(svm_free_model_content(), Compat.Nothing, (Ptr{Compat.Nothing},), mod)
    return (svm)
    #return(mod, weights, weight_labels)
end

"""
`svmpredict{T,U<:Real}(model::SVM{T}, X::AbstractMatrix{U})`

Predict values using `model` based on data `X`. The shape of `X`
needs to be (nfeatures, nsamples). The method returns tuple
(predictions, decisionvalues).
"""
function svmpredict(model::SVM{T}, X::AbstractMatrix{U}; nt::Integer=0) where {T,U<:Real}
    global verbosity

    set_num_threads(nt)

    if size(X,1) != model.nfeatures
        error("Model has $(model.nfeatures) but $(size(X, 1)) provided")
    end

    ninstances = size(X, 2)
    (nodes, nodeptrs) = instances2nodes(X)

    if model.SVMtype == OneClassSVM
        pred = BitArray(undef, ninstances)
    else
        pred = Array{T}(undef, ninstances)
    end

    nlabels = model.nclasses

    if model.SVMtype == EpsilonSVR || model.SVMtype == NuSVR || model.SVMtype == OneClassSVM || model.probability
        decvalues = zeros(Float64, nlabels, ninstances)
    else
        dcols = max(Int64(nlabels*(nlabels-1)/2), 2)
        decvalues = zeros(Float64, dcols, ninstances)
    end

    verbosity = false
    fn = model.probability ? svm_predict_probability() : svm_predict_values()

    cmod, data = svmmodel(model)
    ma = [cmod]

    for i = 1:ninstances
        output = ccall(fn, Float64, (Ptr{Compat.Nothing}, Ptr{SVMNode}, Ptr{Float64}),
                            ma, nodeptrs[i], pointer(decvalues, nlabels*(i-1)+1))

        if model.SVMtype == EpsilonSVR || model.SVMtype == NuSVR
            pred[i] = output
        elseif model.SVMtype == OneClassSVM
            pred[i] = output > 0
        else
            pred[i] = model.labels[round(Int,output)]
        end
    end

    (pred, decvalues)
end

include("ScikitLearnTypes.jl")
include("ScikitLearnAPI.jl")

end
