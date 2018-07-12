
if is_windows()
    lib = joinpath(joinpath(dirname(@__FILE__), "libsvm.dll"))
    info("Downloading LIBSVM binary")
    if Sys.WORD_SIZE == 64
        download("http://web.ics.purdue.edu/~finej/libsvm-3.22_1.dll", lib)
    else
        download("http://web.ics.purdue.edu/~finej/libsvm32-3.22_1.dll", lib)
    end
else
    cd(joinpath(dirname(@__FILE__), "libsvm-3.22"))
    run(`make lib`)
    run(`mv libsvm.so.2 ../libsvm.so.2`)
end
