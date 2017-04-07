module SVMs

export svmtrain, svmpredict, svmcv, SVM, SupportVectors

include("LibSVMtypes.jl")

const SVMS = Dict{Symbol, Int32}(
    :CSVC => Int32(0),
    :NuSVC => Int32(1),
    :OneClassSVM => Int32(2),
    :EpsilonSVR => Int32(3),
    :NuSVR => Int32(4)
    )

const KERNELS = Dict{Symbol, Int32}(
    :Linear => Int32(0),
    :Polynomial => Int32(1),
    :RBF => Int32(2),
    :Sigmoid => Int32(3),
    :Precomputed => Int32(4)
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
    #Fix for regression!
    nodes = [unsafe_load(unsafe_load(smc.SV, i)) for i in 1:smc.l]

    if smc.nSV != C_NULL
        nSV = Array{Int32}(smc.nr_class)
        unsafe_copy!(pointer(nSV), smc.nSV, smc.nr_class)
    else
        nSV = Array{Int32}(0)
    end

    yi = smc.param.svm_type == SVMS[:OneClassSVM] ? Float64[] :
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
    eps::Float64
    C::Float64
    nu::Float64
    p::Float64
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
                        mod.coef0, mod.cache_size, mod.eps, mod.C,
                        length(mod.libsvmweight), pointer(mod.libsvmweightlabel), pointer(mod.libsvmweight),
                        mod.nu, mod.p, Int32(mod.shrinking), Int32(mod.probability))

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
            libsvm = Libdl.dlopen(joinpath(Pkg.dir(), "SVMs", "deps",
                "libsvm.so.2"))
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

function svmtrain{T, U<:Real}(labels::AbstractVector{T},
        instances::AbstractMatrix{U}; svm_type::Symbol=:CSVC,
        kernel_type::Symbol=:RBF, degree::Integer=3,
        gamma::Float64=1.0/size(instances, 1), coef0::Float64=0.0,
        C::Float64=1.0, nu::Float64=0.5, p::Float64=0.1,
        cache_size::Float64=100.0, eps::Float64=0.001, shrinking::Bool=true,
        probability_estimates::Bool=false,
        weights::Union{Dict{T, Float64}, Void}=nothing,
        verbose::Bool=false)
    global verbosity


    svm_tp = SVMS[svm_type]
    kernel = KERNELS[kernel_type]
    wts = weights

    if svm_type == :EpsilonSVR || svm_type == :NuSVR
        idx = labels
        weight_labels = Int32[]
        weights = Float64[]
        reverse_labels = Float64[]
    elseif svm_type == :OneClassSVM
        idx = Float64[]
        weight_labels = Int32[]
        weights = Float64[]
        reverse_labels = Bool[]
    else
        (idx, reverse_labels, weights, weight_labels) = indices_and_weights(labels,
            instances, weights)
    end

    param = Array{SVMParameter}(1)
    param[1] = SVMParameter(svm_tp, kernel, Int32(degree), Float64(gamma),
        coef0, cache_size, eps, C, Int32(length(weights)),
        pointer(weight_labels), pointer(weights), nu, p, Int32(shrinking),
        Int32(probability_estimates))

    # Construct SVMProblem
    (nodes, nodeptrs) = instances2nodes(instances)
    problem = SVMProblem[SVMProblem(Int32(size(instances, 2)), pointer(idx),
        pointer(nodeptrs))]

    verbosity = verbose
    mod = ccall(svm_train(), Ptr{SVMModel}, (Ptr{SVMProblem},
        Ptr{SVMParameter}), problem, param)
    svm = SVM(unsafe_load(mod), labels, instances, wts, reverse_labels,
        svm_type, kernel_type)

    ccall(svm_free_model_content(), Void, (Ptr{Void},), mod)
    return (svm)
    #return(mod, weights, weight_labels)
end


function svmcv{T, U<:Real, V<:Real, X<:Real}(labels::AbstractVector{T},
        instances::AbstractMatrix{U}, nfolds::Int=5,
        C::Union{V, AbstractArray{V}}=2.0.^(-5:2:15),
        gamma::Union{X, AbstractArray{X}}=2.0.^(3:-2:-15);
        svm_type::Int32=CSVC, kernel_type::Int32=RBF, degree::Integer=3,
        coef0::Float64=0.0, nu::Float64=0.5, p::Float64=0.1,
        cache_size::Float64=100.0, eps::Float64=0.001, shrinking::Bool=true,
        weights::Union{Dict{T, Float64}, Void}=nothing,
        verbose::Bool=false)
    global verbosity
    verbosity = verbose
    (idx, reverse_labels, weights, weight_labels) = indices_and_weights(labels,
        instances, weights)

    # Construct SVMParameters
    params = Array(SVMParameter, length(C), length(gamma))
    degree = Int32(degree)
    nweights = Int32(length(weights))
    shrinking = Int32(shrinking)
    for i = 1:length(C), j = 1:length(gamma)
        params[i, j] = SVMParameter(svm_type, kernel_type, Int32(degree),
            Float64(gamma[j]), coef0, cache_size, eps, Float64(C[i]), nweights,
            pointer(weight_labels), pointer(weights), nu, p, shrinking,
            Int32(0))
    end

    # Get information about classes
    (nodes, nodeptrs) = instances2nodes(instances)
    n_classes = length(reverse_labels)
    n_class = zeros(Int, n_classes)
    for id in idx
        n_class[Int64(id)] += 1
    end
    by_class = [Array(Int, n) for n in n_class]
    idx_class = zeros(Int, n_classes)
    for i = 1:length(idx)
        cl = Int64(idx[i])
        by_class[cl][(idx_class[cl] += 1)] = i
    end
    for i = 1:n_classes
        shuffle!(by_class[i])
    end
    prop_class = n_class/length(idx)

    # Perform cross-validation
    decvalues = Array(Float64, length(reverse_labels))
    fold_classes = Array(UnitRange{Int}, n_classes)
    perf = zeros(Float64, length(C), length(gamma))
    for i = 1:nfolds
        # Get range for test for each class
        fold_ntest = 0
        for j = 1:n_classes
            fold_classes[j] =
                div((i-1)*n_class[j], nfolds)+1:div(i*n_class[j], nfolds)
            fold_ntest += length(fold_classes[j])
        end

        # Get indices of test and training instances
        fold_ntrain = length(idx) - fold_ntest
        fold_train = Array(Int, fold_ntrain)
        fold_test = Array(Int, fold_ntest)
        itrain = 0
        itest = 0
        for j = 1:n_classes
            train_range = fold_classes[j]
            class_idx = by_class[j]

            n = first(train_range) - 1
            if n > 0
                fold_train[itest+1:itest+n] = class_idx[1:n]
                itest += n
            end
            n = n_class[j] - last(train_range)
            if n > 0
                fold_train[itest+1:itest+n] = class_idx[last(train_range)+1:end]
                itest += n
            end

            n = length(train_range)
            fold_test[itrain+1:itrain+n] = class_idx[train_range]
            itrain += n
        end

        fold_train_nodeptrs = nodeptrs[fold_train]
        fold_train_labels = idx[fold_train]
        problem = SVMProblem[SVMProblem(Int32(fold_ntrain),
            pointer(fold_train_labels),
            pointer(fold_train_nodeptrs))]

        for j = 1:length(params)
            ptr = ccall(svm_train(), Ptr{Void}, (Ptr{SVMProblem},
                Ptr{SVMParameter}), problem, pointer(params, j))
            correct = 0
            for k in fold_test
                correct += ccall(svm_predict_values(), Float64, (Ptr{Void},
                    Ptr{SVMNode}, Ptr{Float64}), ptr, nodeptrs[k], decvalues) ==
                    idx[k]
            end
            ccall(svm_free_model_content(), Void, (Ptr{Void},), ptr)
            perf[j] += correct/fold_ntest
        end
    end

    best = ind2sub(size(perf), indmax(perf))
    (C[best[1]], gamma[best[2]], perf/nfolds)
end

function svmpredict{T,U<:Real}(model::SVM{T}, instances::AbstractMatrix{U})
    global verbosity

    if size(instances,1) != model.nfeatures
        error("Model has $(model.nfeatures) but $(size(instances, 1)) provided")
    end

    ninstances = size(instances, 2)
    (nodes, nodeptrs) = instances2nodes(instances)

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
        if model.SVMtype == :EpsilonSVR || model.SVMtype == :NuSVR
            pred[i] = output
        elseif model.SVMtype == :OneClassSVM
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
