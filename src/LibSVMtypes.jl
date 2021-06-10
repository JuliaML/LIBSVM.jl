# https://github.com/cjlin1/libsvm/blob/557d85749aaf0ca83fd229af0f00e4f4cb7be85c/svm.h

struct SVMNode
    index::Cint
    value::Cdouble
end

struct SVMProblem
    l::Cint
    y::Ptr{Cdouble}
    x::Ptr{Ptr{SVMNode}}
end

struct SVMParameter
    svm_type   ::Cint
    kernel_type::Cint
    degree     ::Cint
    gamma      ::Cdouble
    coef0      ::Cdouble

    # for training only */
    cache_size  ::Cdouble
    eps         ::Cdouble
    C           ::Cdouble
    nr_weight   ::Cint
    weight_label::Ptr{Cint}
    weight      ::Ptr{Cdouble}
    nu          ::Cdouble
    p           ::Cdouble
    shrinking   ::Cint
    probability ::Cint
end

struct SVMModel
   param     ::SVMParameter
   nr_class  ::Cint
   l         ::Cint
   SV        ::Ptr{Ptr{SVMNode}}
   sv_coef   ::Ptr{Ptr{Cdouble}}
   rho       ::Ptr{Cdouble}
   probA     ::Ptr{Cdouble}
   probB     ::Ptr{Cdouble}
   sv_indices::Ptr{Cint}

   label     ::Ptr{Cint}
   nSV       ::Ptr{Cint}

   free_sv   ::Cint
end
