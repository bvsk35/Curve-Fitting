# This file applied Back Propagation Algorithm
# for curve fitting problem
# Neural Network Architecture has
# 3 layers: Input, Hidden and Output

# Import Required Libraries
import numpy
import matplotlib.pyplot as plt

# Parameters
Epoch = numpy.array([]) # No. of Training Iterations
MSE = numpy.array([]) # Mean Squared Error
M2 = 0 # Momentum Vector for Layer 2
M3 = 0 # Momentum Vector for Layer 3
beta = 0.9 # Momentum Parameter
eta = 4 # Learning Rate
tol = 0.005 # Terminating Condition
iterations = 0
max_iter = 1e6 # Maximum Allowed Iterations

# Load the Data: Points X, D, Weights for 2nd and 3rd layer
X = numpy.loadtxt('X.txt')
D = numpy.loadtxt('D.txt')
W_Layer_2_Guess = numpy.loadtxt('Layer2WeightsInitialGuess.txt')
W_Layer_3_Guess = numpy.loadtxt('Layer3WeightsInitialGuess.txt')

# Required Functions
def ForwardPass(x, W_Layer_2, W_Layer_3):
    # V2 and V3 are locally induced fields at Layer 2 and 3
    # Y2 and Y3 are locally induced fields at Layer 2 and 3
    temp_x = numpy.concatenate(([1], [x]), axis=0)
    V2 = numpy.dot(W_Layer_2, temp_x)
    Y2 = numpy.tanh(V2)
    temp_y = numpy.concatenate(([1], [Y2]), axis=0)	
    V3 = numpy.dot(W_Layer_3, temp_y)
    Y3 = V3
    return V2, V3, Y2, Y3

def BackwardPass(x, d, W_Layer_2, W_Layer_3, V2, V3, Y2, Y3):
    feedback_error = (2/300) * (d - Y3)
    delta3 = feedback_error * 1
    Derivative_Layer2 = numpy.array([1 - numpy.square(numpy.tanh(i)) for i in V2])
    delta2 = numpy.multiply(W_Layer_3[1:]*delta3, Derivative_Layer2)
    Gradient_Layer_2 = numpy.dot(-delta2[:, numpy.newaxis], numpy.concatenate(([[1]], [[x]]), axis=1))
    Gradient_Layer_3 = numpy.dot(-delta3, numpy.concatenate(([1], Y2), axis=0))
    return Gradient_Layer_2, Gradient_Layer_3

def UpdateWeights(eta, beta, M2, M3, Gradient_Layer_2, Gradient_Layer_3, W_Layer_2, W_Layer_3):
    M2 = (beta * M2) - (eta * Gradient_Layer_2) # Momentum
    M3 = (beta * M3) - (eta * Gradient_Layer_3) # Momentum
    W_Layer_2 = W_Layer_2 + M2
    W_Layer_3 = W_Layer_3 + M3
    return W_Layer_2, W_Layer_3, M2, M3

def CalculateMSE(X, D, W_Layer_2, W_Layer_3):
    row = numpy.size(X)
    n = 300
    sum = 0
    for i in range(0, row):
        V2, V3, Y2, Y3 = ForwardPass(X[i], W_Layer_2, W_Layer_3)
        sum = sum + numpy.square(D[i] - Y3)/n
    return sum

def CheckLearningrate(eta, MSE):
    if MSE[-1] > MSE[-2]:
        eta = 0.4 * eta
    return eta

# Main Loop
while iterations <= max_iter:
    if iterations == 0:
        # Back Propagation
        row = numpy.size(X)
        temp_w2 = numpy.array([W_Layer_2_Guess])
        temp_w3 = numpy.array([W_Layer_3_Guess])
        for i in range(0, row):
            V2, V3, Y2, Y3 = ForwardPass(X[i], temp_w2[-1], temp_w3[-1])
            g2, g3 = BackwardPass(X[i], D[i], temp_w2[-1], temp_w3[-1], V2, V3, Y2, Y3)
            w2, w3, M2, M3 = UpdateWeights(eta, beta, M2, M3, g2, g3, temp_w2[-1], temp_w3[-1])
            temp_w2 = numpy.concatenate((temp_w2, [w2]), axis=0)
            temp_w3 = numpy.concatenate((temp_w3, [w3]), axis=0)
        # Book Keeping
        W2 = numpy.concatenate(([W_Layer_2_Guess], [temp_w2[-1]]), axis=0)
        W3 = numpy.concatenate(([W_Layer_3_Guess], [temp_w3[-1]]), axis=0)
        Epoch = numpy.concatenate((Epoch, [iterations]), axis=0)
        mse = CalculateMSE(X, D, W2[-1], W3[-1])
        MSE = numpy.concatenate((MSE, [mse]), axis=0)
        # Print
        print('Epoch: ', iterations, ' MSE: ', mse, ' Learning Rate: ', eta, '\n')
        # Next...
        iterations += 1
    else:
        # Back Propagation
        row = numpy.size(X)
        temp_w2 = numpy.array([W2[-1]])
        temp_w3 = numpy.array([W3[-1]])
        for i in range(0, row):
            V2, V3, Y2, Y3 = ForwardPass(X[i], temp_w2[-1], temp_w3[-1])
            g2, g3 = BackwardPass(X[i], D[i], temp_w2[-1], temp_w3[-1], V2, V3, Y2, Y3)
            w2, w3, M2, M3 = UpdateWeights(eta, beta, M2, M3, g2, g3, temp_w2[-1], temp_w3[-1])
            temp_w2 = numpy.concatenate((temp_w2, [w2]), axis=0)
            temp_w3 = numpy.concatenate((temp_w3, [w3]), axis=0)
        # Book Keeping
        W2 = numpy.concatenate((W2, [temp_w2[-1]]), axis=0)
        W3 = numpy.concatenate((W3, [temp_w3[-1]]), axis=0)
        Epoch = numpy.concatenate((Epoch, [iterations]), axis=0)
        mse = CalculateMSE(X, D, W2[-1], W3[-1])
        MSE = numpy.concatenate((MSE, [mse]), axis=0)
        # Print
        print('Epoch: ', iterations, ' MSE: ', mse, ' Learning Rate: ', eta, '\n')
        # Check
        # if (MSE[-1] - MSE[-2]).all() < tol:
        #     print('Optimal Weights Reached')
        #     break
        if MSE[-1] < tol:
            print('Optimal Weights Reached')
            break
        eta = CheckLearningrate(eta, MSE)
        # Next...
        iterations += 1

# Save Final Weights
numpy.savetxt('FinalOptimalWeights2.txt', W2[-1])
numpy.savetxt('FinalOptimalWeights3.txt', W3[-1])

# Plot
# Plot 1
fig, ax1 = plt.subplots()
row = numpy.size(X)
Y = numpy.array([])
for i in range(0, row):
    V2, V3, Y2, Y3 = ForwardPass(X[i], W2[-1], W3[-1])
    Y = numpy.concatenate((Y, [Y3]), axis=0)
ax1.plot(X, D, 'b.', label='Actual Curve')
ax1.plot(X, Y, 'r.', label='Curve Fit')
plt.title(r'Curve Fitting for the equation $ y = \sin{20x} + 3 x + v $ where v is random variable and v $ \in [-0.1, 0.1] $')
plt.xlabel(r'X $\rightarrow$')
plt.ylabel(r'Y $\rightarrow$')
plt.legend()
# Plot 2
fig, ax2 = plt.subplots()
ax2.plot(Epoch, MSE, label='Mean Squared Error')
plt.title('No of Training Iterations VS Mean Squared Error (MSE)')
plt.xlabel(r'Epoch $\rightarrow$')
plt.ylabel(r'MSE $\rightarrow$')
plt.legend()
plt.show()
