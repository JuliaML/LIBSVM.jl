module LIBSVM

export svmtrain, svmpredict

const CSVC = 0
const NuSVC = 1
const OneClassSVM = 2
const EpsilonSVR = 3
const NuSVR = 4

const Linear = 0 
const Polynomial = 1
const RBF = 2
const Sigmoid = 3
const Precomputed = 4

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
    labels::Vector{T}
    weight_labels::Vector{Int32}
    weights::Vector{Float64}
    nfeatures::Int
end

let libsvm = C_NULL
    global get_libsvm
    get_libsvm() = libsvm == C_NULL ?
        libsvm = dlopen(joinpath(Pkg.dir(), "LIBSVM", "deps", "libsvm.so.2")) :
        libsvm
end

macro cachedsym(symname)
    cached = gensym()
    quote
        let $cached = C_NULL
            global ($symname)
            ($symname)() = ($cached) == C_NULL ?
                ($cached = dlsym(get_libsvm(), $(string(symname)))) : $cached
        end
    end
end
@cachedsym svm_train
@cachedsym svm_predict_values
@cachedsym svm_predict_probability
@cachedsym svm_free_model_content

function grp2idx{T, S <: Real}(::Type{S}, labels::Vector,
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

function instances2nodes(instances::Array{Float64, 2})
    nfeatures = size(instances, 1)
    ninstances = size(instances, 2)
    nodeptrs = Array(Ptr{SVMNode}, ninstances)
    nodes = Array(SVMNode, nfeatures + 1, ninstances)

    for i=1:ninstances
        for j=1:nfeatures
            nodes[j, i] = SVMNode(int32(j), instances[j, i])
        end
        nodes[nfeatures+1, i] = SVMNode(int32(-1), NaN)
        nodeptrs[i] = pointer(sub(nodes, 1, i))
    end

    (nodes, nodeptrs)
end

function svmtrain{T}(labels::Vector{T}, instances::Array{Float64, 2};
    svm_type::Int=CSVC, kernel_type::Int=RBF, degree::Int=3, gamma::Float64=.00,
    coef0::Float64=0.0, C::Float64=1.0, nu::Float64=0.5, p::Float64=0.1,
    cache_size=100.0::Float64, eps::Float64=0.001, shrinking::Bool=true,
    probability_estimates::Bool=false,
    weights::Union(Dict{T, Float64}, Nothing)=nothing, cv_folds::Int=0,
    verbose::Bool=false)

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
        nr_weight = 0
        weight_labels = Int32[]
        weights = Float64[]
    else
        nr_weight = length(weights)
        weight_labels = grp2idx(Int32, keys(weights), label_dict,
            reverse_labels)
        weights = float64(values(weights))
    end

    if gamma == 0.0
        gamma = 1.0/size(instances, 1)
    end

    param = Array(SVMParameter, 1)
    param[1] = SVMParameter(int32(svm_type), int32(kernel_type), int32(degree),
        gamma, coef0, cache_size, eps, C, int32(length(weights)),
        pointer(weight_labels), pointer(weights), nu, p, int32(shrinking),
        int32(probability_estimates))

    # Construct SVMProblem
    (nodes, nodeptrs) = instances2nodes(instances)
    problem = Array(SVMProblem, 1)
    problem[1] = SVMProblem(int32(size(instances, 2)), pointer(idx),
        pointer(nodeptrs))

    ptr = ccall(svm_train(), Ptr{Void}, (Ptr{SVMProblem},
        Ptr{SVMParameter}), problem, param)

    model = SVMModel(ptr, param, reverse_labels, weight_labels, weights,
        size(instances, 1))
    finalizer(model, svmfree)
    model
end

svmfree(model::SVMModel) = ccall(svm_free_model_content(), Void, (Ptr{Void},),
    model.ptr)

function svmpredict{T}(model::SVMModel{T}, instances::Array{Float64, 2})
    ninstances = size(instances, 2)

    if size(instances, 1) != model.nfeatures
        error("Model has $(model.nfeatures) but $(size(instances, 1)) provided")
    end

    (nodes, nodeptrs) = instances2nodes(instances)
    class = Array(T, ninstances)
    decvalues = Array(Float64, length(model.labels), ninstances)

    fn = model.param[1].probability == 1 ? svm_predict_probability() :
        svm_predict_values()
    for i = 1:ninstances
        output = ccall(fn, Float64, (Ptr{Void}, Ptr{SVMNode}, Ptr{Float64}),
            model.ptr, nodeptrs[i], sub(decvalues, 1, i))
        class[i] = model.labels[int(output)]
    end

    (class, decvalues)
end
end