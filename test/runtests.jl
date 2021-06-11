using DelimitedFiles
using FileIO
using JLD2
using LIBSVM
using RDatasets
using SparseArrays
using Test

@testset "LibSVM" begin

function test_iris_model(model, X, y)
    ans = Bool[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    # TODO: check the decision value ?
    ŷ, _ = svmpredict(model, X)
    @test (ŷ .== y) == ans
    ŷ
end

@testset "libsvm version" begin
    @test LIBSVM.libsvm_version[] ≥ 322
end


@testset "IRIS" begin
    @info "test iris"

    iris = readdlm(joinpath(@__DIR__, "iris.csv"), ',')
    labels = iris[:, 5]

    instances = Matrix{Float64}(iris[:, 1:4]')
    model = svmtrain(instances[:, 1:2:end], labels[1:2:end]; verbose = true)
    GC.gc()
    class = test_iris_model(model, instances[:, 2:2:end], labels[2:2:end])

    @testset "sklearn API" begin
        skmodel = fit!(SVC(), instances[:,1:2:end]', labels[1:2:end])
        GC.gc()
        skclass = predict(skmodel, instances[:, 2:2:end]')
        @test skclass == class

        model = svmtrain(sparse(instances[:, 1:2:end]), labels[1:2:end]; verbose = true)
        test_iris_model(model, sparse(instances[:, 2:2:end]), labels[2:2:end])
    end

    @test_throws DimensionMismatch svmpredict(model, rand(3, 42))
end

@testset "AbstractVector as labels" begin
    @info "test AbstractVector labels"

    iris = dataset("datasets", "iris")
    X = Matrix(iris[:, 1:4])'
    y = iris.Species

    Xtrain = X[:, 1:2:end]
    Xtest  = X[:, 2:2:end]
    ytrain = y[1:2:end]
    ytest  = y[2:2:end]

    model = svmtrain(Xtrain, ytrain, verbose = true)
    ŷ = test_iris_model(model, Xtest, ytest)

    model = fit!(SVC(), Xtrain', ytrain)
    @test ŷ == predict(model, Xtest')
end

@testset "JLD2 save/load" begin
    @info "JLD2 save/load"

    iris = dataset("datasets", "iris")
    X = Matrix(iris[:, 1:4])'
    y = iris.Species

    Xtrain = X[:, 1:2:end]
    Xtest  = X[:, 2:2:end]
    ytrain = y[1:2:end]
    ytest  = y[2:2:end]

    model = svmtrain(Xtrain, ytrain, verbose = true)
    ŷ = test_iris_model(model, Xtest, ytest)

    mktempdir() do path
        f = joinpath(path, "test.jld2")
        @save f model

        model′ = load(f, "model")
        test_iris_model(model′, Xtest, ytest)
    end

    model = fit!(SVC(), Xtrain', ytrain)
    mktempdir() do path
        f = joinpath(path, "test.jld2")
        @save f model

        model′ = load(f, "model")
        @test ŷ == predict(model′, Xtest')
    end
end


@testset "Whiteside" begin
    #Regression tests, results confirmed using e1071 R-package
    @info "test Whiteside"

    whiteside = RDatasets.dataset("MASS", "whiteside")
    ws = Matrix{Float64}(whiteside[:,2:3])
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

@testset "Check parameters" begin
    bad_params = Dict(:svmtype => OneClassSVM, :probability => true)
    @test_throws ArgumentError svmtrain(rand(2, 5), ones(5); bad_params...)
end

@testset "Decision values" begin
    X = [-2 -1 -1 1 1 2;
         -1 -1 -2 1 2 1]
    y = [1, 1, 1, 2, 2, 2]
    d = [1.5 1.0 1.5 -1.0 -1.5 -1.5;
         0.0 0.0 0.0  0.0  0.0  0.0]

    model = svmtrain(X, y, kernel = Kernel.Linear)
    ỹ, d̃ = svmpredict(model, X)

    @test ỹ == y
    @test d ≈ d̃
end

@testset "Precomputed kernel" begin
    @testset "Input matrix validation" begin
        nonsquare_mat = rand(5, 6)
        y = rand(Bool, size(nonsquare_mat, 1))
        @test_throws DimensionMismatch svmtrain(nonsquare_mat, y,
                                                kernel=Kernel.Precomputed)
    end

    @testset "Trivial data" begin
        X = [-2 -1 -1 1 1 2; -1 -1 -2 1 2 1]
        y = [1, 1, 1, 2, 2, 2]

        K = X' * X

        model = svmtrain(K, y, kernel=Kernel.Precomputed)

        @test model.rho ≈ [0]
        @test model.coefs ≈ [0.25; -0.25]
        @test model.SVs.indices == [2, 4]

        ỹ, _ = svmpredict(model, K)
        @test y == ỹ

        T = [-1 2 3; -1 2 2]
        K = T' * X

        ỹ, _ = svmpredict(model, K)
        @test [1, 2, 2] == ỹ
    end

    @testset "Random data" begin
        K = [ 0.28 0.97 0.75 0.48 0.39 0.68 0.52 0.14
             ;0.97 0.23 0.01 0.76 0.06 0.06 0.25 0.23
             ;0.75 0.01 0.03 0.27 0.69 0.43 0.3  0.67
             ;0.48 0.76 0.27 0.36 0.35 0.39 0.56 0.80
             ;0.39 0.06 0.69 0.35 0.39 0.42 0.49 0.22
             ;0.68 0.06 0.43 0.39 0.42 0.13 0.37 0.66
             ;0.52 0.25 0.30 0.56 0.49 0.37 0.66 0.54
             ;0.14 0.23 0.67 0.80 0.22 0.66 0.54 0.13 ]
        y = [1, 1, 2, 2, 1, 1, 2, 1]

        model = svmtrain(K, y, kernel=Kernel.Precomputed)

        @test model.rho ≈ [-1.1449999948963523]
        @test model.coefs ≈ [1; 1; 1; -1; -1; -1]
        @test model.SVs.indices == [1, 5, 8, 3, 4, 7]

        ỹ, _ = svmpredict(model, K)
        @test ỹ == [1, 1, 1, 1, 1, 1, 1, 2]

        K = [ 0.34 0.75 0.97 0.39 0.23 0.52 0.68 0.14
             ;0.12 0.27 0.76 0.35 0.36 0.56 0.39 0.80
             ;0.45 0.90 0.32 0.12 0.05 1.49 0.42 0.22
             ;0.24 0.67 0.23 0.22 0.80 0.54 0.66 0.13 ]
        ỹ, _ = svmpredict(model, K)
        @test ỹ == [2, 1, 1, 1]
    end
end

end  # @testset "LIBSVM"
