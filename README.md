# LIBSVM.jl

[![Build Status](https://github.com/JuliaML/LIBSVM.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaML/LIBSVM.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaML/LIBSVM.jl/branch/master/graph/badge.svg?token=bGwzyTtNPn)](https://codecov.io/gh/JuliaML/LIBSVM.jl)

This is a Julia interface for [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).

**Features:**
* Supports all LIBSVM models: classification C-SVC, nu-SVC, regression: epsilon-SVR, nu-SVR
    and distribution estimation: one-class SVM
* Model objects are represented by Julia type SVM which gives you easy
  access to model features and can be saved e.g. as JLD file
* Supports ScikitLearn.jl API

## Usage

### LIBSVM API

This provides a lower level API similar to LIBSVM C-interface. See `?svmtrain`
for options.

```julia
using LIBSVM
using RDatasets
using Printf
using Statistics

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# First four dimension of input data is features
X = Matrix(iris[:, 1:4])'

# LIBSVM handles multi-class data automatically using a one-against-one strategy
y = iris.Species

# Split the dataset into training set and testing set
Xtrain = X[:, 1:2:end]
Xtest  = X[:, 2:2:end]
ytrain = y[1:2:end]
ytest  = y[2:2:end]

# Train SVM on half of the data using default parameters. See documentation
# of svmtrain for options
model = svmtrain(Xtrain, ytrain)

# Test model on the other half of the data.
ŷ, decision_values = svmpredict(model, Xtest);

# Compute accuracy
@printf "Accuracy: %.2f%%\n" mean(ŷ .== ytest) * 100
```

### ScikitLearn API

You can alternatively use `ScikitLearn.jl` API with same options as `svmtrain`:

```julia
using LIBSVM
using RDatasets

# Classification C-SVM
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = iris.Species

Xtrain = X[1:2:end, :]
Xtest  = X[2:2:end, :]
ytrain = y[1:2:end]
ytest  = y[2:2:end]

model = fit!(SVC(), Xtrain, ytrain)
ŷ = predict(model, Xtest)
```

```julia
# Epsilon-Regression

whiteside = RDatasets.dataset("MASS", "whiteside")
X = Matrix(whiteside[:, 3:3])  # the `Gas` column
y = whiteside.Temp

model = fit!(EpsilonSVR(cost = 10., gamma = 1.), X, y)
ŷ = predict(model, X)
```

## Credits

The library is currently developed and maintained by Matti Pastell. It was originally
developed by Simon Kornblith.

[LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) by Chih-Chung Chang and Chih-Jen Lin
