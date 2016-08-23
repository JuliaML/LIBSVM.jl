using LIBSVM
iris = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
labels = iris[:, 5]
instances = convert(Matrix{Float64}, iris[:, 1:4]')
model = svmtrain(labels[1:2:end], instances[:, 1:2:end]; verbose=true)
gc()
(class, decvalues) = svmpredict(model, instances[:, 2:2:end])
correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
@assert (class .== labels[2:2:end]) == correct

svmpredict_out = init_svmpredict(instances[:,2:2:end])
svmpredict!(svmpredict_out, model, instances[:, 2:2:end])
@assert (svmpredict_out[1] .== labels[2:2:end]) == correct

model = svmtrain(labels[1:2:end], sparse(instances[:, 1:2:end]); verbose=true)
gc()
(class, decvalues) = svmpredict(model, sparse(instances[:, 2:2:end]))
correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
@assert (class .== labels[2:2:end]) == correct


