# Python Neural Network
The purpose of this project was to apply self training on Python, as well as have a deeper understanding of how neural networks work by implementing a simple neural network.

This sample is based off of the Python AI: How to [Build a Neural Network & Make Predictions](https://realpython.com/python-ai-neural-network/) web page.

**Python Concepts**
- How to install python modules: numpy and matplotlib
- Structure for defining a Python class and implementing use of this class
- Differences between scalars (3.14), lists ([1.0,2.0,3.0]) and numpy arrays (np.array([1.0,2.0,3.0])
- Looping over "collections" (such as lists or numpy arrays)
- Applying mathematical operations for collections
- Python printing formats, such as: print(f'...

**Neural Network Concepts**
- Use of dot product to determine how similar two vectors are.  Definitions for this [dot product](https://en.wikipedia.org/wiki/Dot_product) are available on wikipedia. A useful definition is that the dot product of two vectors, a and b, is||a|| * ||b|| * cos(angle), where ||a|| and ||b|| are the magnitudes (lengths) of the related vectors and angle is the angle betwen the two vectors.  A high dot product indicates that the vectors are aligned, with the largest value being when the angle between them is zero (cos(0)=1).  If the vectors are not aligned, having an angle approaching 90 degress, the this value will be small and become zero if the angle is 90 degrees (cos(90)=0).
- Applying the sigmoid function.  From previous study of neural networks I was already aware of the "S" shaped sigmoid function for implementing the neural network neuron.  However, I wasn't aware that the derivative of this function (1/(1+e**-x)) is calculated using the results of this function.  If we say that sigmoid(x)=1/(1+e**-x), then the derivative of sigmoid(x) is sigmoid(x) * (1-sigmoid(x)).  Once the value for sigmoid(x) is calculated it can be reused to define the derivate.
- How to calculate the gradient descent (slope). To determine the gradient descent at a particular location a [Chain Rule](https://en.wikipedia.org/wiki/Chain_rule) concept is used which allows for aggregating partial differential values.  For example, dz/dx=dz/dy * dy/dx, if dz/dy and dy/dx can be determined then they can be combined to calculate dz/dx.  This is critical when training the neural network for determining the gradient used to adjust the parameters when training by applying backwards propogation.
   * Since the error for the dot product function is the resul value minus the expectd value squared (MSE), the derivative is two times the difference (differntial of x**2 is 2*x)
   * The derivative of the sigmoid function (1/(1+e**-x)) is sigmoid(x) * (1-sigmoid(x))
   * Since the bias is a constant, the derivative of this constant is 1: derivative of x**n is n * x**(n-1)
   * The derivative of the weights vector is the weights vector
   * bias gradient=dot product derivative * (2 * (calculated value - acutal value)) * 1
   * weight gradient=dot product derivative * (sigmoid(x) * (1-sigmoid(x)) * weights vector
- To train this neural network a sample set of data with known expected outcomes is processed through the neural network to identify errors and adjust accordingly
   * To avoid overfitting of neural network to the sample data (resulting network fits supplied data but doesn't work well with values that it hasn't seen before), a stochiastic gradient descent approach is used. This involved iterating over a large number of iterations (10,000), randomly selecting training data and then processing the gradient to adjust the bias and weight values
   * Every 10th iteration the cumulative error across the input set is calculated and recorded to see how the network is doing

**Conclusions**

This simple network was extremely useful for understanding how neural networks work, as well as providing a practical application for implementing a program using Python.

