module SVMs

export svmtrain, svmpredict

include("LibSVMtypes.jl")

const SVMS = Dict{Symbol, Int32}(
    :CSVC => Int32(0),
    :nuSVC => Int32(1),
    :oneclassSVM => Int32(2),
    :epsilonSVR => Int32(3),
    :nuSVR => Int32(4)
    )

const KERNELS = Dict{Symbol, Int32}(
    :linear => Int32(0),
    :polynomial => Int32(1),
    :RBF => Int32(2),
    :sigmoid => Int32(3),
    :precomputed => Int32(4)
    )

verbosity = false

immutable SupportVectors{T, U}
    l::Int32
    nSV::Vector{Int32}
    y::Vector{T}
    X::AbstractMatrix{U}
    indices::Vector{Int32}
    SVnodes::Vector{SVMNode}
end

function SupportVectors(smc::SVMModel, y, X)
    println(smc.l)
    sv_indices = Array{Int32}(smc.l)
    unsafe_copy!(pointer(sv_indices), smc.sv_indices, smc.l)
    nodes = [unsafe_load(unsafe_load(smc.SV, i)) for i in 1:smc.l]

    if smc.nSV != C_NULL
        nSV = Array{Int32}(smc.nr_class)
        unsafe_copy!(pointer(nSV), smc.nSV, smc.nr_class)
    else
        nSV = Array{Int32}(0)
    end

    yi = smc.param.svm_type == SVMS[:oneclassSVM] ? Float64[] :
                                                    y[sv_indices]

    SupportVectors(smc.l, nSV, yi , X[:,sv_indices],
                        sv_indices, nodes)
end

immutable SVM{T}
    SVMtype::Symbol
    kernel::Symbol
    weights::Union{Dict{T, Float64}, Void}
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
    C::Float64
    nu::Float64
    epsilon::Float64
    shrinking::Bool
    probability::Bool
end

function SVM{T}(smc::SVMModel, y::T, X, weights, labels, svmtype, kernel)
    svs = SupportVectors(smc, y, X)
    coefs = zeros(smc.l, smc.nr_class-1)
    for k in 1:(smc.nr_class-1)
        unsafe_copy!(pointer(coefs, (k-1)*smc.l +1 ), unsafe_load(smc.sv_coef, k), smc.l)
    end
    k = smc.nr_class
    rs = Int(k*(k-1)/2)
    rho = Vector{Float64}(rs)
    unsafe_copy!(pointer(rho), smc.rho, rs)

    if smc.label == C_NULL
        libsvmlabel = Vector{Int32}(0)
    else
        libsvmlabel = Vector{Int32}(k)
        unsafe_copy!(pointer(libsvmlabel), smc.label, k)
    end

    if smc.probA == C_NULL
        probA = Float64[]
        probB = Float64[]
    else
        probA = Vector{Float64}(rs)
        probB = Vector{Float64}(rs)
        unsafe_copy!(pointer(probA), smc.probA, rs)
        unsafe_copy!(pointer(probB), smc.probB, rs)
    end

    #Weights
    nw = smc.param.nr_weight
    libsvmweight = Array{Float64}(nw)
    libsvmweight_label = Array{Int32}(nw)

    if nw > 0
        unsafe_copy!(pointer(libsvmweight), smc.param.weight, nw)
        unsafe_copy!(pointer(libsvmweight_label), smc.param.weight_label, nw)
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
immutable SVMData
    coefs::Vector{Ptr{Float64}}
    nodes::Array{SVMNode}
    nodeptrs::Array{Ptr{SVMNode}}
end

"""Convert SVM model to libsvm struct for prediction"""
function svmmodel(mod::SVM)
    svm_type = SVMS[mod.SVMtype]
    kernel = KERNELS[mod.kernel]

    param = SVMParameter(svm_type, kernel, mod.degree, mod.gamma,
                        mod.coef0, mod.cache_size, mod.tolerance, mod.C,
                        length(mod.libsvmweight), pointer(mod.libsvmweightlabel), pointer(mod.libsvmweight),
                        mod.nu, mod.epsilon, Int32(mod.shrinking), Int32(mod.probability))

    n,m = size(mod.coefs)
    sv_coef = Vector{Ptr{Float64}}(m)
    for i in 1:m
        sv_coef[i] = pointer(mod.coefs, (i-1)*n+1)
    end

    nodes, ptrs = SVMs.instances2nodes(mod.SVs.X)
    data = SVMData(sv_coef, nodes, ptrs)

    cmod = SVMModel(param, mod.nclasses, mod.SVs.l, pointer(data.nodeptrs), pointer(data.coefs),
                pointer(mod.rho), pointer(mod.probA), pointer(mod.probB), pointer(mod.SVs.indices),
                pointer(mod.libsvmlabel),
                pointer(mod.SVs.nSV), Int32(1))

    return cmod, data
end


let libsvm = C_NULL
    global get_libsvm
    function get_libsvm()
        if libsvm == C_NULL
            if is_windows()
                libsvm = Libdl.dlopen(joinpath(dirname(@__FILE__),  "../deps",
                "libsvm.dll"))
            else
                libsvm = Libdl.dlopen(joinpath(dirname(@__FILE__),  "../deps",
                "libsvm.so.2"))
            end
            # libsvm = Libdl.dlopen("/usr/local/Cellar/libsvm/3.21/lib/libsvm.2.dylib")
            ccall(Libdl.dlsym(libsvm, :svm_set_print_string_function), Void,
                (Ptr{Void},), cfunction(svmprint, Void, (Ptr{UInt8},)))
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
@cachedsym svm_predict_values
@cachedsym svm_predict_probability
@cachedsym svm_free_model_content

function grp2idx{T, S <: Real}(::Type{S}, labels::AbstractVector,
    label_dict::Dict{T, Int32}, reverse_labels::Vector{T})

    idx = Array{S}(length(labels))
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

function instances2nodes{U<:Real}(instances::AbstractMatrix{U})
    nfeatures = size(instances, 1)
    ninstances = size(instances, 2)
    nodeptrs = Array{Ptr{SVMNode}}(ninstances)
    nodes = Array{SVMNode}(nfeatures + 1, ninstances)

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

function instances2nodes{U<:Real}(instances::SparseMatrixCSC{U})
    ninstances = size(instances, 2)
    nodeptrs = Array{Ptr{SVMNode}}(ninstances)
    nodes = Array{SVMNode}(nnz(instances)+ninstances)

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

function svmprint(str::Ptr{UInt8})
    if verbosity::Bool
        print(unsafe_string(str))
    end
    nothing
end

function indices_and_weights{T, U<:Real}(labels::AbstractVector{T},
        instances::AbstractMatrix{U},
        weights::Union{Dict{T, Float64}, Void}=nothing)
    label_dict = Dict{T, Int32}()
    reverse_labels = Array{T}(0)
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

#Old LIBSVM
function _svmtrain{T, U<:Real}(labels::AbstractVector{T},
        instances::AbstractMatrix{U}; svm_type::Int32=Int32(0),
        kernel_type::Int32=Int32(2), degree::Integer=3,
        gamma::Float64=1.0/size(instances, 1), coef0::Float64=0.0,
        C::Float64=1.0, nu::Float64=0.5, p::Float64=0.1,
        cache_size::Float64=100.0, eps::Float64=0.001, shrinking::Bool=true,
        probability_estimates::Bool=false,
        weights::Union{Dict{T, Float64}, Void}=nothing,
        verbose::Bool=false)
    global verbosity

    (idx, reverse_labels, weights, weight_labels) = indices_and_weights(labels,
        instances, weights)

    param = Array{SVMParameter}(1)
    param[1] = SVMParameter(svm_type, kernel_type, Int32(degree), Float64(gamma),
        coef0, cache_size, eps, C, Int32(length(weights)),
        pointer(weight_labels), pointer(weights), nu, p, Int32(shrinking),
        Int32(probability_estimates))

    # Construct SVMProblem
    (nodes, nodeptrs) = instances2nodes(instances)
    problem = SVMProblem[SVMProblem(Int32(size(instances, 2)), pointer(idx),
        pointer(nodeptrs))]

    verbosity = verbose
    ptr = ccall(svm_train(), Ptr{SVMModel}, (Ptr{SVMProblem},
        Ptr{SVMParameter}), problem, param)

    return (ptr, nodes, nodeptrs)
end

"""
```julia
svmtrain{T, U<:Real}(X::AbstractMatrix{U}, y::AbstractVector{T}=[];
    svmtype::Symbol=:CSVC, kernel::Symbol=:RBF, degree::Integer=3,
    gamma::Float64=1.0/size(X, 1), coef0::Float64=0.0,
    cost::Float64=1.0, nu::Float64=0.5, epsilon::Float64=0.1,
    tolerance::Float64=0.001, shrinking::Bool=true,
    probability::Bool=false, weights::Union{Dict{T, Float64}, Void}=nothing,
    cachesize::Float64=200.0, verbose::Bool=false)
```
Train Support Vector Machine using LIBSVM using response vector `y`
and training data `X`. The shape of `X` needs to be (nsamples, nfeatures).
For one-class SVM use only `X`.

# Arguments

* `svmtype::Symbol=:CSVC`: Type of SVM to train `:CSVC`, `:nuSVC`
    `:oneclassSVM`, `:epsilonSVR` or `:nuSVR`. Defaults to `:oneclassSVM` if
    `y` is not used.
* `kernel::Symbol=:RBF`: Model kernel `:linear`, `:polynomial`, `:RBF`,
    `:sigmoid` or `:precomputed`.
* `degree::Integer=3`: Kernel degree. Used for polynomial kernel
* `gamma::Float64=1.0/size(X, 1)` : γ for kernels
* `coef0::Float64=0.0`: parameter for sigmoid and polynomial kernel
* `cost::Float64=1.0`: cost parameter C of C-SVC, epsilon-SVR, and nu-SVR
* `nu::Float64=0.5`: parameter nu of nu-SVC, one-class SVM, and nu-SVR
* `epsilon::Float64=0.1`: epsilon in loss function of epsilon-SVR
* `tolerance::Float64=0.001`: tolerance of termination criterion
* `shrinking::Bool=true`: whether to use the shrinking heuristics
* `probability::Bool=false`: whether to train a SVC or SVR model for probability estimates
* `weights::Union{Dict{T, Float64}, Void}=nothing`: dictionary of class weights
* `cachesize::Float64=100.0`: cache memory size in MB
* `verbose::Bool=false`: print training output from LIBSVM if true

Consult LIBSVM documentation for advice on the choise of correct
parameters and model tuning.
"""
function svmtrain{T, U<:Real}(X::AbstractMatrix{U}, y::AbstractVector{T} = [];
        svmtype::Symbol=:CSVC,
        kernel::Symbol=:RBF, degree::Integer=3,
        gamma::Float64=1.0/size(X, 1), coef0::Float64=0.0,
        cost::Float64=1.0, nu::Float64=0.5, epsilon::Float64=0.1,
        tolerance::Float64=0.001, shrinking::Bool=true,
        probability::Bool=false, weights::Union{Dict{T, Float64}, Void}=nothing,
        cachesize::Float64=200.0, verbose::Bool=false)
    global verbosity

    isempty(y) && (svmtype = :oneclassSVM)

    _svmtype = SVMS[svmtype]
    _kernel = KERNELS[kernel]
    wts = weights

    if svmtype == :epsilonSVR || svmtype == :nuSVR
        idx = labels
        weight_labels = Int32[]
        weights = Float64[]
        reverse_labels = Float64[]
    elseif svmtype == :oneclassSVM
        idx = Float64[]
        weight_labels = Int32[]
        weights = Float64[]
        reverse_labels = Bool[]
    else
        (idx, reverse_labels, weights, weight_labels) = indices_and_weights(y,
            X, weights)
    end

    param = Array{SVMParameter}(1)
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

    ccall(svm_free_model_content(), Void, (Ptr{Void},), mod)
    return (svm)
    #return(mod, weights, weight_labels)
end

"""
`svmpredict{T,U<:Real}(model::SVM{T}, X::AbstractMatrix{U})`

Predict values using `model` based on data `X`. The shape of `X`
needs to be (nsamples, nfeatures). The method returns tuple
(predictions, decisionvalues).
"""
function svmpredict{T,U<:Real}(model::SVM{T}, X::AbstractMatrix{U})
    global verbosity

    if size(X,1) != model.nfeatures
        error("Model has $(model.nfeatures) but $(size(X, 1)) provided")
    end

    ninstances = size(X, 2)
    (nodes, nodeptrs) = instances2nodes(X)

    pred = Array{T}(ninstances)
    nlabels = model.nclasses
    decvalues = Array{Float64}(nlabels, ninstances)

    verbosity = false
    fn = model.probability ? svm_predict_probability() : svm_predict_values()

    cmod, data = svmmodel(model)
    ma = [cmod]

    for i = 1:ninstances
        output = ccall(fn, Float64, (Ptr{Void}, Ptr{SVMNode}, Ptr{Float64}),
            ma, nodeptrs[i], pointer(decvalues, nlabels*(i-1)+1))
        if model.SVMtype == :epsilonSVR || model.SVMtype == :nuSVR
            pred[i] = output
        elseif model.SVMtype == :oneclassSVM
            pred[i] = output > 0
        else
            pred[i] = model.labels[round(Int,output)]
        end
    end

    (pred, decvalues)
end

function svmpredict2{U<:Real}(model::SVMModel,
        instances::AbstractMatrix{U})
    global verbosity

    ninstances = size(instances, 2)

    (nodes, nodeptrs) = instances2nodes(instances)
    class = Array{Int64}(ninstances)

    nlabels = model.nr_class
    decvalues = Array{Float64}(nlabels, ninstances)

    verbosity = false
    fn = model.param.probability == 1 ? svm_predict_probability() :
        svm_predict_values()
    ma = [model]
    for i = 1:ninstances
        output = ccall(fn, Float64, (Ptr{Void}, Ptr{SVMNode}, Ptr{Float64}),
            ma, nodeptrs[i], pointer(decvalues, nlabels*(i-1)+1))
        #class[i] = model.labels[round(Int,output)]
        class[i] = round(Int,output)
    end

    (class, decvalues)
end

end
