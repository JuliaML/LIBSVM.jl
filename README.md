# LIBSVM.jl

[![Build Status](https://travis-ci.org/mpastell/LIBSVM.jl.svg?branch=master)](https://travis-ci.org/mpastell/LIBSVM.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/uo9ioc72cvowuy7f/branch/master?svg=true)](https://ci.appveyor.com/project/mpastell/svms-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/mpastell/LIBSVM.jl/badge.svg?branch=master)](https://coveralls.io/github/mpastell/LIBSVM.jl?branch=master)

This is a Julia interface for [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).

**Features:**
* Supports all LIBSVM models: classification C-SVC, nu-SVC, regression: epsilon-SVR, nu-SVR
    and distribution estimation: one-class SVM
* Model objects are represented by Julia type SVM which gives you easy
  access to model features and can be saved e.g. as JLD file

## Usage

```julia
using RDatasets, LIBSVM

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# LIBSVM handles multi-class data automatically using a one-against-one strategy
labels = convert(Vector, iris[:Species])

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

## Credits

The library is currently developed and maintained by Matti Pastell. It was originally
developed by Simon Kornblith.

[LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) by Chih-Chung Chang and Chih-Jen Lin
