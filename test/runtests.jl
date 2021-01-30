using DelimitedFiles
using LIBSVM
using SparseArrays
using Test

@testset "LibSVM" begin


@testset "IRIS" begin
  iris = readdlm(joinpath(@__DIR__, "iris.csv"), ',')
  labels = iris[:, 5]

  instances = convert(Matrix{Float64}, iris[:, 1:4]')
  model = svmtrain(instances[:, 1:2:end], labels[1:2:end]; verbose = true)
  GC.gc()
  (class, decvalues) = svmpredict(model, instances[:, 2:2:end])
  correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  @test (class .== labels[2:2:end]) == correct

  skmodel = fit!(SVC(), instances[:,1:2:end]', labels[1:2:end])
  skclass = predict(skmodel, instances[:, 2:2:end]')
  @test skclass == class
  model = svmtrain(sparse(instances[:, 1:2:end]), labels[1:2:end]; verbose=true)
  GC.gc()
  (class, decvalues) = svmpredict(model, sparse(instances[:, 2:2:end]))
  correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  @test (class .== labels[2:2:end]) == correct
end


@testset "Whiteside" begin
  #Regression tests, results confirmed using e1071 R-package
  #whiteside = RDatasets.dataset("MASS", "whiteside")
  whiteside, hdr = readdlm(joinpath(@__DIR__, "whiteside.csv"), ',', header=true)
  ws = convert(Matrix{Float64}, whiteside[:,2:3])
  X = Array{Float64, 2}(ws[:, 2]')
  y = ws[:, 1]

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
  @test isapprox(ysknu2, ynu2)

  @testset "Multithreading" begin
    # Assign by maximum number of threads
    ntnu1 = svmtrain(X, y, svmtype = NuSVR, cost = 10.,
                    nu = .7, gamma = 2., tolerance = .001,
                    nt = -1)
    ntynu1, ntd = svmpredict(ntnu1, X)
    @test sum(ntynu1 - y) ≈ 14.184665717092

    # Assign by environment
    ENV["OMP_NUM_THREADS"] = 2

    ntm = svmtrain(X, y, svmtype = EpsilonSVR, cost = 10., gamma = 1.)
    ntyeps, ntd = svmpredict(m, X)
    @test sum(ntyeps - y) ≈ 7.455509045783046
  end
end


end  # @testset "LIBSVM"
