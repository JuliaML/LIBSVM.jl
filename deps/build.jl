
if is_windows()
    lib = joinpath(joinpath(dirname(@__FILE__), "libsvm.dll"))
    if !isfile(lib)
        info("Downloading LIBSVM binary")
        if Sys.WORD_SIZE == 64
            download("https://mpastell.github.io/LIBSVM.jl/bindeps/libsvm.dll", lib)
        else
            download("https://mpastell.github.io/LIBSVM.jl/bindeps/libsvm32.dll", lib)
        end
    end
else
    cd(joinpath(dirname(@__FILE__), "libsvm-3.22"))
    run(`make lib`)
    run(`mv libsvm.so.2 ../libsvm.so.2`)
end
