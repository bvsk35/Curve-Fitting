# This file generates data i.e. Points and
# Initial guess weights of the Neural Network
# and stores in them in a .txt file

# Import Required Libraries
import numpy

# NN Architecture
# Layers 3
# No. of neurons in Hidden Layer N = 24
# No. of points n = 300
# Points
X = numpy.random.uniform(0, 1, (300, 1))
V = numpy.random.uniform(-0.1, 0.1, (300, 1))
D = numpy.array([])
for i in range(0, 300):
    d = numpy.sin(20 * X[i]) + 3 * X[i] + V[i]
    D = numpy.concatenate((D, d), axis=0)
x = numpy.ones((300, 1))
X_Prime = numpy.concatenate((x, X), axis=1)

# Weights
# Layer 1 Input Layer - No weights and Bias
# Layer 2 Hidden Layer - N weights and N bias
# Layer 3 Output Layer - N weights and 1 bias
# W_Layer_2 = 24 x 2 (including bias as first column)
# W_Layer_3 = 1 x 25 (including bias as first entry)
W_Layer_2 = numpy.random.normal(0, 0.20412, (24, 2)) # Xavier Initialisation
W_Layer_3 = numpy.random.normal(0, 1.0, (1, 25)) # Xavier Initialisation
# W_Layer_2 = numpy.random.uniform(-5, 5, (24, 2))
# W_Layer_3 = numpy.random.normal(-5, 5, (1, 25))

# Save everything in text file
# numpy.savetxt('X.txt', X)
# numpy.savetxt('D.txt', D)
numpy.savetxt('Layer2WeightsInitialGuess.txt', W_Layer_2)
numpy.savetxt('Layer3WeightsInitialGuess.txt', W_Layer_3)