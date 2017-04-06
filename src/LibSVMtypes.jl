
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

immutable SVMModel
   param::SVMParameter
   nr_class::Int32
   l::Int32
   SV::Ptr{Ptr{SVMNode}}
   sv_coef::Ptr{Ptr{Float64}}
   rho::Ptr{Float64}
   probA::Ptr{Float64}
   probB::Ptr{Float64}
   sv_indices::Ptr{Int32}

   label::Ptr{Int32}
   nSV::Ptr{Int32}

   free_sv::Int32
end
