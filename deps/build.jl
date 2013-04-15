cd(joinpath(Pkg.dir(), "LIBSVM", "deps", "libsvm-3.17"))
run(`make lib`)
run(`cp libsvm.so.2 ../libsvm.so.2`)