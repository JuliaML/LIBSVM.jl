# LIBSVM.jl

[![Build Status](https://github.com/JuliaML/LIBSVM.jl/workflows/CI/badge.svg)](https://github.com/JuliaML/LIBSVM.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/JuliaML/LIBSVM.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/LIBSVM.jl?branch=master)

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
using Printf, Statistics

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# LIBSVM handles multi-class data automatically using a one-against-one strategy
labels = levelcode.(iris[:Species])

# First dimension of input data is features; second is instances
instances = convert(Array, iris[:, 1:4])'

# Train SVM on half of the data using default parameters. See documentation
# of svmtrain for options
model = svmtrain(instances[:, 1:2:end], labels[1:2:end]);

# Test model on the other half of the data.
(predicted_labels, decision_values) = svmpredict(model, instances[:, 2:2:end]);

# Compute accuracy
@printf "Accuracy: %.2f%%\n" mean((predicted_labels .== labels[2:2:end]))*100
```

### ScikitLearn API

You can alternatively use `ScikitLearn.jl` API with same options as `svmtrain`:

```julia
using LIBSVM
using RDatasets

#Classification C-SVM
iris = dataset("datasets", "iris")
labels = levelcode.(iris[:, :Species])
instances = convert(Array, iris[:, 1:4])
model = fit!(SVC(), instances[1:2:end, :], labels[1:2:end])
yp = predict(model, instances[2:2:end, :])

#epsilon-regression
whiteside = RDatasets.dataset("MASS", "whiteside")
X = Array(whiteside[:Gas])
if typeof(X) <: AbstractVector
    X = reshape(X, (length(X),1))
end
y = Array(whiteside[:Temp])
svrmod = fit!(EpsilonSVR(cost = 10., gamma = 1.), X, y)
yp = predict(svrmod, X)
```

## Credits

The library is currently developed and maintained by Matti Pastell. It was originally
developed by Simon Kornblith.

[LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) by Chih-Chung Chang and Chih-Jen Lin
