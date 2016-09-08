module LIBSVM

export svmtrain, svmpredict, svmcv, init_svmpredict, svmpredict!

const CSVC = Int32(0)
const NuSVC = Int32(1)
const OneClassSVM = Int32(2)
const EpsilonSVR = Int32(3)
const NuSVR = Int32(4)

const Linear = Int32(0)
const Polynomial = Int32(1)
const RBF = Int32(2)
const Sigmoid = Int32(3)
const Precomputed = Int32(4)

verbosity = false

immutable SVMNode
    index::Int32
    value::Float64
end

immutable SVMProblem
    l::Int32
    y::Ptr{Float64}
    x::Ptr{Ptr{SVMNode}}
end

immutable SVMParameter
    svm_type::Int32
    kernel_type::Int32
    degree::Int32
    gamma::Float64
    coef0::Float64

    cache_size::Float64
    eps::Float64
    C::Float64
    nr_weight::Int32
    weight_label::Ptr{Int32}
    weight::Ptr{Float64}
    nu::Float64
    p::Float64
    shrinking::Int32
    probability::Int32
end

# immutable SVMModel
#   param::SVMParameter
#   nr_class::Int32
#   l::Int32
#   SV::Ptr{Ptr{SVMNode}}
#   sv_coef::Ptr{Ptr{Float64}}
#   rho::Ptr{Float64}
#   probA::Ptr{Float64}
#   probB::Ptr{Float64}
#   sv_indices::Ptr{Int32}

#   label::Ptr{Int32}
#   nSV::Ptr{Int32}

#   free_sv::Int32
# end

type SVMModel{T}
    ptr::Ptr{Void}
    param::Vector{SVMParameter}

    # Prevent these from being garbage collected
    problem::Vector{SVMProblem}
    nodes::Array{SVMNode}
    nodeptr::Vector{Ptr{SVMNode}}

    labels::Vector{T}
    weight_labels::Vector{Int32}
    weights::Vector{Float64}
    nfeatures::Int
    verbose::Bool
end

type SVM_OUT
  class::Array{Any}
  decvalues::Array{Float64}
  nodes::Array{SVMNode}
  nodeptrs::Array{Ptr{SVMNode}}
end

let libsvm = C_NULL
    global get_libsvm
    function get_libsvm()
        if libsvm == C_NULL
            libsvm = Libdl.dlopen(joinpath(Pkg.dir(), "LIBSVM", "deps",
                "libsvm.so.2"))
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

    idx = Array(S, length(labels))
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
    nodeptrs = Array(Ptr{SVMNode}, ninstances)
    nodes = Array(SVMNode, nfeatures + 1, ninstances)

    instances2nodes!(nodes, nodeptrs, instances)

    (nodes, nodeptrs)
end


function instances2nodes!{U<:Real}(nodes, nodeptrs, instances::AbstractMatrix{U})
    nfeatures = size(instances, 1)
    ninstances = size(instances, 2)
    if(size(nodes) != (nfeatures+1,ninstances))
       error("size(nodes) has to be equal (size(instances, 1), size(instances, 2))")
    end
    if(size(nodeptrs,1) != ninstances)
      error("size(nodeptrs,1) has to be equal size(instances, 2)")
    end

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
    nodeptrs = Array(Ptr{SVMNode}, ninstances)
    nodes = Array(SVMNode, nnz(instances)+ninstances)

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
        print(bytestring(str))
    end
    nothing
end

function indices_and_weights{T, U<:Real}(labels::AbstractVector{T},
        instances::AbstractMatrix{U},
        weights::Union{Dict{T, Float64}, Void}=nothing)
    label_dict = Dict{T, Int32}()
    reverse_labels = Array(T, 0)
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
        weight_labels = grp2idx(Int32, keys(weights), label_dict,
            reverse_labels)
        weights = Float64(values(weights))
    end

    (idx, reverse_labels, weights, weight_labels)
end

function svmtrain{T, U<:Real}(labels::AbstractVector{T},
        instances::AbstractMatrix{U}; svm_type::Int32=CSVC,
        kernel_type::Int32=RBF, degree::Integer=3,
        gamma::Float64=1.0/size(instances, 1), coef0::Float64=0.0,
        C::Float64=1.0, nu::Float64=0.5, p::Float64=0.1,
        cache_size::Float64=100.0, eps::Float64=0.001, shrinking::Bool=true,
        probability_estimates::Bool=false,
        weights::Union{Dict{T, Float64}, Void}=nothing,
        verbose::Bool=false)
    global verbosity

    (idx, reverse_labels, weights, weight_labels) = indices_and_weights(labels,
        instances, weights)

    param = Array(SVMParameter, 1)
    param[1] = SVMParameter(svm_type, kernel_type, Int32(degree), Float64(gamma),
        coef0, cache_size, eps, C, Int32(length(weights)),
        pointer(weight_labels), pointer(weights), nu, p, Int32(shrinking),
        Int32(probability_estimates))

    # Construct SVMProblem
    (nodes, nodeptrs) = instances2nodes(instances)
    problem = SVMProblem[SVMProblem(Int32(size(instances, 2)), pointer(idx),
        pointer(nodeptrs))]

    verbosity = verbose
    ptr = ccall(svm_train(), Ptr{Void}, (Ptr{SVMProblem},
        Ptr{SVMParameter}), problem, param)

    model = SVMModel(ptr, param, problem, nodes, nodeptrs, reverse_labels,
        weight_labels, weights, size(instances, 1), verbose)
    finalizer(model, svmfree)
    model
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
        n_class[id] += 1
    end
    by_class = [Array(Int, n) for n in n_class]
    idx_class = zeros(Int, n_classes)
    for i = 1:length(idx)
        cl = idx[i]
        by_class[cl][(idx_class[cl] += 1)] = i
    end
    for i = 1:n_classes
        shuffle!(by_class[i])
    end
    prop_class = n_class/length(idx)

    # Perform cross-validation
    decvalues = Array(Float64, length(reverse_labels))
    fold_classes = Array(Range1{Int}, n_classes)
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

svmfree(model::SVMModel) = ccall(svm_free_model_content(), Void, (Ptr{Void},),
    model.ptr)

function svmpredict{T, U<:Real}(model::SVMModel{T},
        instances::AbstractMatrix{U})

    svmpredict_out = init_svmpredict(instances)
    svmpredict!(svmpredict_out, model, instances)

    svmpredict_out
end

function svmpredict!{T, U<:Real}(svmpredict_out::SVM_OUT
                                 , model::SVMModel{T}
                                 , instances::AbstractMatrix{U})
  (class, decvalues, nodes, nodeptrs) = (svmpredict_out.class, svmpredict_out.decvalues, svmpredict_out.nodes, svmpredict_out.nodeptrs)
  if(size(class, 1) != size(decvalues, 2))
    error("svmpredict_out provides size(class, 1) = $(size(class, 1)) but size(decvalues, 2) = $(size(decvalues, 2)). Has to be equal.")
  end
    global verbosity
    ninstances = size(instances, 2)

    if size(instances, 1) != model.nfeatures
        error("Model has $(model.nfeatures) but $(size(instances, 1)) provided")
    end

    instances2nodes!(nodes, nodeptrs, instances)
    nlabels = length(model.labels)

    verbosity = model.verbose
    fn = model.param[1].probability == 1 ? svm_predict_probability() :
        svm_predict_values()
    for i = 1:ninstances
        output = ccall(fn, Float64, (Ptr{Void}, Ptr{SVMNode}, Ptr{Float64}),
            model.ptr, nodeptrs[i], pointer(decvalues, nlabels*(i-1)+1))
        if model.param[1].svm_type==Int32(2) # if one class SVM
          class[i]=output
        else
          class[i] = model.labels[round(Int, output)]
        end
    end

    svmpredict_out
end

function init_svmpredict{U<:Real}(instances::AbstractMatrix{U})
  nfeatures = size(instances, 1)
  ninstances = size(instances, 2)
  nodeptrs = Array(Ptr{SVMNode}, ninstances)
  nodes = Array(SVMNode, nfeatures + 1, ninstances)
  class = Array(Any, ninstances);
  decvalues = Array(Float64, 1, ninstances);
  svmpredict_out = SVM_OUT(class, decvalues, nodes, nodeptrs)
  return(svmpredict_out)
end

end
