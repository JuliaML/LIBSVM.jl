# LIBSVM.jl

[![Build Status](https://github.com/JuliaML/LIBSVM.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaML/LIBSVM.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaML/LIBSVM.jl/branch/master/graph/badge.svg?token=bGwzyTtNPn)](https://codecov.io/gh/JuliaML/LIBSVM.jl)

This is a Julia interface for
[LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) and for the linear
SVM model provided by
[LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).

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

### Precomputed kernel

It is possible to use different kernels than those that are provided. In such a
case, it is required to provide a matrix filled with precomputed kernel values.

For training, a symmetric matrix is expected:
```
K = [k(x_1, x_1)  k(x_1, x_2)  ...  k(x_1, x_l);
     k(x_2, x_1)
         ...                            ...
     k(x_l, x_1)        ...         k(x_l, x_l)]
```
where `x_i` is `i`-th training instance and `l` is the number of training
instances.

To predict `n` instances, a matrix of shape `(l, n)` is expected:
```
KK = [k(x_1, t_1)  k(x_1, t_2)  ...  k(x_1, t_n);
      k(x_2, t_1)
          ...                            ...
      k(x_l, t_1)        ...         k(x_l, t_n)]
```
where `t_i` is `i`-th instance to be predicted.

#### Example

```julia
# Training data
X = [-2 -1 -1 1 1 2;
     -1 -1 -2 1 2 1]
y = [1, 1, 1, 2, 2, 2]

# Testing data
T = [-1 2 3;
     -1 2 2]

# Precomputed matrix for training (corresponds to linear kernel)
K = X' * X

model = svmtrain(K, y, kernel=Kernel.Precomputed)

# Precomputed matrix for prediction
KK = X' * T

ỹ, _ = svmpredict(model, KK)
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

### MLJ API

The MLJ interface to LIBSVM.jl consists of the following models:

- classification: `LinearSVC`, `SVC`, `NuSVC` 
- regression: `EpsilonSVR`, `NuSVR`
- outlier detection: `OneClassSVM`

Each model has a detailed document string, which includes examples of
usage. Document strings can be accessed from MLJ without loading
`LIBSVM.jl` (or its MLJ interface) as shown in the following example:

```julia
using MLJ     # or MLJModels 
doc("NuSVC", pkg="LIBSVM")
```

This assumes the version of MLJModels loaded is 0.15.5 or higher.

Below we illustrate usage of the classic kernel `SVC` classifier. In
the example, `X` can be replaced with any table (e.g., `DataFrame`)
whose columns have `Float64` eltype. The target `y` can be any
`CategoricalVector`.

### Using a built-in kernel

```
using MLJ
import LIBSVM

SVC = @load SVC pkg=LIBSVM                   # model type
model = SVC(kernel=LIBSVM.Kernel.Polynomial) # instance

X, y = @load_iris # table, vector
mach = machine(model, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "virginica"
 "virginica"
 "virginica"
```

### User-defined kernels

```
k(x1, x2) = x1'*x2 # equivalent to `LIBSVM.Kernel.Linear`
model = SVC(kernel=k)
mach = machine(model, X, y) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "virginica"
 "virginica"
 "virginica"
```

### Incorporating class weights

In either scenario above, we can do:

```julia
weights = Dict("virginica" => 1, "versicolor" => 20, "setosa" => 1)
mach = machine(model, X, y, weights) |> fit!

julia> yhat = predict(mach, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "versicolor"
 "versicolor"
 "versicolor"
```


## Credits

The library is currently developed and maintained by Matti Pastell. It was originally
developed by Simon Kornblith.

[LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) by Chih-Chung Chang and Chih-Jen Lin
