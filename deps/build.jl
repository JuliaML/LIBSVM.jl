
if is_windows()
    lib = joinpath(joinpath(dirname(@__FILE__), "libsvm.dll"))
    info("Downloading LIBSVM binary")
    if Sys.WORD_SIZE == 64
        download("https://mpastell.github.io/LIBSVM.jl/bindeps/libsvm-3.22_0.dll", lib)
    else
        download("https://mpastell.github.io/LIBSVM.jl/bindeps/libsvm32-3.22_0.dll", lib)
    end
else
    cd(joinpath(dirname(@__FILE__), "libsvm-3.22"))
    run(`make lib`)
    run(`mv libsvm.so.2 ../libsvm.so.2`)
end
