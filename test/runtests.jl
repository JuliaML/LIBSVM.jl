using RDatasets, LIBSVM
iris = data("datasets", "iris")
labels = iris["Species"].data
instances = transpose(matrix(iris[:, 2:5]))
model = svmtrain(labels[1:2:end], instances[:, 1:2:end]; verbose=true)
gc()
(class, decvalues) = svmpredict(model, instances[:, 2:2:end])
correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
@assert (class .== labels[2:2:end]) == correct

model = svmtrain(labels[1:2:end], sparse(instances[:, 1:2:end]); verbose=true)
gc()
(class, decvalues) = svmpredict(model, sparse(instances[:, 2:2:end]))
correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
@assert (class .== labels[2:2:end]) == correct