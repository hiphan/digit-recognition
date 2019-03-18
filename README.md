# Grayscale digit recognition
Implement feedforward propagation, back propagation to train a neural network to recognize digit. The neural net consists 
4 layers: input layer, 2 hidden layers with 25 neurons each, and output layer. The activation functions used are leaky relu -> leaky relu (both with alpha = 0.01) -> sigmoid. 
Other activation functions are also available in activation_func.py.  
The train and test sets are from Kaggle (https://www.kaggle.com/c/digit-recognizer).  
The neural net is trained using batch gradient descent in 5000 iterations. Training accuracy is 93% on average.