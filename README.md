LIBSVM.jl
=========

Julia bindings for [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

## Usage

```julia
using RDatasets, LIBSVM

# Load Fisher's classic iris data
iris = data("datasets", "iris")

# LIBSVM handles multi-class data automatically using a one-against-one strategy
labels = iris["Species"].data

# First dimension of input data is features; second is instances
instances = transpose(matrix(iris[:, 2:5]))

# Train SVM on half of the data using default parameters. See the svmtrain
# function in LIBSVM.jl for optional parameter settings.
model = svmtrain(labels[1:2:end], instances[:, 1:2:end]);

# Test model on the other half of the data.
(predicted_labels, decision_values) = svmpredict(model, instances[:, 2:2:end]);

# Compute accuracy
@printf "Accuracy: %.2f%%\n" mean((predicted_labels .== labels[2:2:end]))*100
```

## Credits

Created by Simon Kornblith

[LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) by Chih-Chung Chang and Chih-Jen Lin