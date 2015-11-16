using LIBSVM
iris = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
labels = iris[:, 5]
instances = convert(Matrix{Float64}, iris[:, 1:4]')
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

#################################################################
# Testing regression
# Compare output of regression with stuff I've pre-generated
#  using the libsvm command line svm-train and svm-predict calls
# regression.csv stores the pre-generated values.
# The julia values should be close.
#################################################################
srand(0)
x = collect(0.0:1.0:10.0)
xnoise = randn(length(x))
y = x.^2 + xnoise
model = svmtrain(y, x', svm_type=LIBSVM.EpsilonSVR, C=100.0)
xtest = collect(0.5:1.0:9.5)
(ypredict, decvalues) = svmpredict(model, xtest')
ypredict_stored = vec(readcsv("regression.csv"))
@assert norm(ypredict - ypredict_stored, Inf)  < (1e-4)
