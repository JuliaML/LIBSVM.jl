using SparseArrays

function node_pointers(nodes::AbstractMatrix{SVMNode})
    nrows, ncols = size(nodes)
    pointers = Array{Ptr{SVMNode}}(undef, ncols)

    for i in 1:ncols
        pointers[i] = pointer(nodes, (i - 1) * nrows + 1)
    end

    return pointers
end

function fill_nodes!(nodes::AbstractMatrix{SVMNode},
                     X::AbstractMatrix{<:Real})
    nrows, ncols = size(X)

    # The last row will be filled with terminal nodes as required by
    # SVMLIB
    @assert size(nodes, 1) == nrows + 1
    @assert size(nodes, 2) == ncols

    for i in 1:ncols
        for j in 1:nrows
            nodes[j, i] = SVMNode(j, X[j, i])
        end
        nodes[end, i] = SVMNode(-1, NaN)
    end

    return nodes
end

function fill_ids!(ids::AbstractVector{SVMNode})
    n = length(ids)
    for i in 1:n
        ids[i] = SVMNode(0, i)
    end
end

function gram2nodes(X::AbstractMatrix{<:Real})
    # For training, n=l represents the number of training instances
    # For prediction, n is the number of instances to be predicted,
    # while l is the number of training instances (some of which are
    # the support vectors)
    l, n = size(X)

    # One extra row for instance IDs and one for terminal nodes
    # Instance IDs are required by LIBSVM
    nodes = Array{SVMNode}(undef, l + 2, n)

    # Create the nodes for instance IDs
    fill_ids!(@view(nodes[1, :]))
    fill_nodes!(@view(nodes[2:end, :]), X)

    nodeptrs = node_pointers(nodes)

    return nodes, nodeptrs
end

gram2nodes(X::SparseMatrixCSC{<:Real}) = gram2nodes(Matrix(X))

function instances2nodes(X::AbstractMatrix{<:Real})
    nrows, ncols = size(X)

    nodes = Array{SVMNode}(undef, nrows + 1, ncols)
    fill_nodes!(nodes, X)

    nodeptrs = node_pointers(nodes)

    return nodes, nodeptrs
end

function instances2nodes(instances::SparseMatrixCSC{<:Real})
    X = instances
    ncols = size(X, 2)

    nodes = Vector{SVMNode}(undef, nnz(X) + ncols)
    nodeptrs = Vector{Ptr{SVMNode}}(undef, ncols)

    rows = rowvals(X)
    vals = nonzeros(X)

    k = 1
    for i in 1:ncols
        nodeptrs[i] = pointer(nodes, k)
        for j in nzrange(X, i)
            nodes[k] = SVMNode(rows[j], vals[j])
            k += 1
        end
        nodes[k] = SVMNode(-1, NaN)
        k += 1
    end

    return nodes, nodeptrs
end
