
# Requirement
tensorflow  
python 3.5.x

# About
This model is a multilayer perceptron model that identifies a voice as male or female with reference to Kaggle.

# Architecture
A neural network with 4 hidden layers each containing 300 nodes. A relu activation function is used at previous three layers and softmax function is used at last one layer. The batch size is set to 40, and a total of 10 epochs are performed. To prevent overfitting, I dropped out 0.3 percent of nodes for each layer. 

<img src="Stacked-autoencoder-with-softmax-classifier.png">

# Optimizer / Loss function
Adam optimizer / categorical_cross_entropy

# Reference
Bronson, Simple NN with Keras, https://www.kaggle.com/jsultan/simple-nn-with-keras.  
Vineet Ahirkar and Naveen Bansal, Gender Recognition using Voice, University of Maryland, Baltimore Country.
