
if is_windows()
    lib = joinpath(joinpath(dirname(@__FILE__), "libsvm.dll"))
    if !isfile(lib)
        info("Downloading LIBSVM binary")
        download("https://mpastell.github.io/SVMs.jl/bindeps/libsvm.dll", lib)
    end
else
    cd(joinpath(dirname(@__FILE__), "libsvm-3.17"))
    run(`make lib`)
    run(`cp libsvm.so.2 ../libsvm.so.2`)
end
