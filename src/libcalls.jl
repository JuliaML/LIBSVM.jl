function libsvm_get_max_threads()
    return ccall((:svm_get_max_threads, libsvm), Cint, ())
end

function libsvm_set_num_threads(n::Integer)
    ccall((:svm_set_num_threads, libsvm), Cvoid, (Cint,), n)
end

noprint(str::Ptr{UInt8})::Cvoid = nothing

function libsvm_set_verbose(v::Bool)
    f = ifelse(v, C_NULL, @cfunction(noprint, Cvoid, (Ptr{UInt8},)))
    ccall((:svm_set_print_string_function, libsvm), Cvoid, (Ptr{Cvoid},), f)
end

function libsvm_check_parameter(problem::SVMProblem, param::SVMParameter)
    err = ccall((:svm_check_parameter, libsvm), Cstring,
                (Ref{SVMProblem}, Ref{SVMParameter}),
                problem, param)
    if err != C_NULL
        throw(ArgumentError("Incorrect parameter: $(unsafe_string(err))"))
    end
end

function libsvm_train(problem::SVMProblem, param::SVMParameter)
    return ccall((:svm_train, libsvm), Ptr{SVMModel},
                 (Ref{SVMProblem}, Ref{SVMParameter}),
                 problem, param)
end

function libsvm_free_model(model::Ptr{SVMModel})
    ccall((:svm_free_model_content, libsvm), Cvoid, (Ptr{SVMModel},),
          model)
end

function libsvm_predict_probability(model::SVMModel, nodes::Ptr{SVMNode},
        decisions::Vector{Float64})
    return ccall((:svm_predict_probability, libsvm), Float64,
                 (Ref{SVMModel}, Ptr{SVMNode}, Ref{Float64}),
                 model, node, decisions)
end

function libsvm_predict_values(model::SVMModel, nodes::Ptr{SVMNode},
        decisions::Vector{Float64})
    return ccall((:svm_predict_values, libsvm), Float64,
                 (Ref{SVMModel}, Ptr{SVMNode}, Ref{Float64}),
                 model, nodes, decisions)
end
