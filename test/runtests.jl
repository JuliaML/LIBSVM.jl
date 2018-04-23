using LIBSVM, Base.Test
import RDatasets

iris = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
labels = iris[:, 5]
instances = convert(Matrix{Float64}, iris[:, 1:4]')
model = svmtrain(instances[:, 1:2:end], labels[1:2:end]; verbose=true)
gc()
(class, decvalues) = svmpredict(model, instances[:, 2:2:end])
correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
@test (class .== labels[2:2:end]) == correct

skmodel = fit!(SVC(), instances[:,1:2:end]', labels[1:2:end])
skclass = predict(skmodel, instances[:, 2:2:end]')
@test skclass == class

model = svmtrain(sparse(instances[:, 1:2:end]), labels[1:2:end]; verbose=true)
gc()
(class, decvalues) = svmpredict(model, sparse(instances[:, 2:2:end]))
correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
@test (class .== labels[2:2:end]) == correct

#Regression tests, results confirmed using e1071 R-package
whiteside = RDatasets.dataset("MASS", "whiteside")
X = Array(whiteside[:Gas]')
y = Array(whiteside[:Temp])

m = svmtrain(X, y, svmtype = EpsilonSVR, cost = 10., gamma = 1.)
yeps, d = svmpredict(m, X)
@test sum(yeps - y) ≈ 7.455509045783046
skm = fit!(EpsilonSVR(cost = 10., gamma = 1.), X', y)
ysk = predict(skm, X')
@test isapprox(yeps,ysk)

nu1 = svmtrain(X, y, svmtype = NuSVR, cost = 10.,
                nu = .7, gamma = 2., tolerance = .001)
ynu1, d = svmpredict(nu1, X)
@test sum(ynu1 - y) ≈  14.184665717092
sknu1 = fit!(NuSVR(cost = 10., nu=.7, gamma = 2.), X', y)
ysknu1 = predict(sknu1, X')
@test isapprox(ysknu1,ynu1)

nu2 = svmtrain(X, y, svmtype = NuSVR, cost = 10., nu = .9)
ynu2, d =svmpredict(nu2, X)
@test sum(ynu2 - y) ≈ 6.686819661799177
sknu2 = fit!(NuSVR(cost = 10., nu=.9), X', y)
ysknu2 = predict(sknu2, X')
@test isapprox(ysknu2,ynu2)
