# Neural-Networks
Feed Forward NNs or Multilayer Perceptrons (MLPs) &amp; Deep Feed Forward (DFF) Neural Networks
It is essential to remember that Neural Networks are most frequently employed to solve classification and regression problems using labeled training data. Hence, an alternative approach could be to put them under the Supervised branch of Machine Learning.

•	Input Layer — contains one or more input nodes. For example, suppose you want to predict whether it will rain tomorrow and base your decision on two variables, humidity and wind speed. In that case, your first input would be the value for humidity, and the second input would be the value for wind speed.

•	Hidden Layer — this layer houses hidden nodes, each containing an activation function (more on these later). Note that a Neural Network with multiple hidden layers is known as Deep Neural Network.

•	Output Layer — contains one or more output nodes. Following the same weather prediction example above, you could choose to have only one output node generating a rain probability (where >0.5 means rain tomorrow, and ≤0.5 no rain tomorrow). Alternatively, you could have two output nodes, one for rain and another for no rain. Note, you can use a different activation function for output nodes vs. hidden nodes.

•	Connections — lines joining different nodes are known as connections. These contain kernels (weights) and biases, the parameters that get optimized during the training of a neural network.

Paramters and activation functions: 

•	Kernels (weights) — used to scale input and hidden node values. Each connection typically holds a different weight.

•	Biases — used to adjust scaled values before passing them through an activation function.

•	Activation functions — think of activation functions as standard curves (building blocks) used by the Neural Network to create a custom curve to fit the training data. Passing different input values through the network selects different sections of the standard curve, which are then assembled into a final custom-fit curve.There are many activation functions to choose from, with Softplus, ReLU, and Sigmoid being the most commonly used. 

Loss functions, optimizers, and training: 

Training Neural Networks involves a complicated process known as backpropagation.

This algorithm is called backpropagation because it tries to reduce errors from output to input. It looks for the minimum value of the error function in the weight field using a technique called gradient descent.
Backpropagation is the core of neural network training. It is a method of adjusting the weights of a neural network based on the loss value obtained in the previous epoch. Correctly adjusting the weights allows us to reduce the error rate and increase its generalization, making the model reliable.

Let me briefly introduce you to loss functions and optimizers and summarize what happens when we “train” a Neural Network.

•	Loss — represents the “size” of error between the true values/labels and the predicted values/labels. The goal of training a Neural Network is to minimize this loss. The smaller the loss, the closer the match between the true and the predicted data. There are many loss functions to choose from, with BinaryCrossentropy, CategoricalCrossentropy, and MeanSquaredError being the most common.

•	Optimizers — are the algorithms used in backpropagation. The goal of an optimizer is to find the optimum set of kernels (weights) and biases to minimize the loss. Optimizers typically use a gradient descent approach, which allows them to iteratively find the “best” possible configuration of weights and biases. The most commonly used ones are SGD, ADAM, and RMSProp.
Gradient descent is an iterative optimization algorithm used in machine learning to minimize a loss function. The loss function describes how well the model will perform given the current set of parameters (weights and biases), and gradient descent is used to find the best set of parameters.

•	In general, the wide selection of activation functions combined with the ability to add as many hidden nodes as we wish (provided we have sufficient computational power) means that Neural Networks can create a curve of any shape to fit the data.

•	However, having this extreme flexibility may sometimes lead to overfitting the data. Hence, we must always ensure that we validate the model on the test/validation set before using it to make predictions.

Neural Networks take one or multiple input values and apply transformations using kernels (weights) and biases before passing results through activation functions. In the end, we get an output (prediction), which is a result of this complex set of transformations optimized through training.

We train Neural Networks by fitting a custom curve through the training data, guided by loss minimization and achieved through parameter (kernels and biases) optimization using Gradient descent algorithm. 
