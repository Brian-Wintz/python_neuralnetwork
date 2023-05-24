import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self,learning_rate):
        self.weights=np.array([np.random.randn(),np.random.randn()])
        self.bias=np.random.randn()
        self.learning_rate=learning_rate

    # sigmoid=1/(1+e**-x)
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    # derivative of sigmoid function=sigmoid(x)*(1-sigmoid(x))
    def _sigmoid_deriv(self,x):
        sigmoid_value=_sigmoid(x)
        return self.sigmoid_value*(1-sigmoid_value)

    def predict(self,input_vector):
        layer_1=np.dot(input_vector,self.weights)+self.bias
        return self._sigmoid(layer_1)

    # calculate weight and bias error gradients
    def _compute_gradients(self,input_vector,target):
        layer_1=np.dot(input_vector,self.weights)+self.bias
        layer_2=self._sigmoid(layer_1)
        prediction=layer_2

        derror_dprediction=2*(prediction-target)
        dprediction_dlayer1=layer_2*(1-layer_2)
        dlayer1_dbias=1
        dlayer1_dweights=(0*self.weights)+(1*input_vector)

        derror_dbias=(derror_dprediction*dprediction_dlayer1*dlayer1_dbias)
        derror_dweights=(derror_dprediction*dprediction_dlayer1*dlayer1_dweights)

        return derror_dbias,derror_dweights

    # adjust bias and weight values, using learning rate to "dampen" adjustment
    def _update_parameters(self,derror_dbias,derror_dweights):
        self.bias=self.bias-(derror_dbias*self.learning_rate)
        self.weights=self.weights-(derror_dweights*self.learning_rate)

    # use the input_vectors and expected targets to train this network running through iterations number of cycles by applying stochiastic gradient descent
    def train(self,input_vectors,targets,iterations):
        cumulative_errors=[]
        for current_iteration in range(iterations):
            random_data_index=np.random.randint(len(input_vectors))

            input_vector=input_vectors[random_data_index]
            target=targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias,derror_dweights=self._compute_gradients(input_vector,target)

            self._update_parameters(derror_dbias,derror_dweights)

            # Measure the cumulative error for every 10th iteration
            if current_iteration % 10==0:
                cumulative_error=0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point=input_vectors[data_instance_index]
                    target=targets[data_instance_index]

                    prediction=self.predict(data_point)
                    error=np.square(prediction-target)

                    cumulative_error=cumulative_error+error
                cumulative_errors.append(cumulative_error)
        return cumulative_errors

if __name__ == "__main__":
    # Input Vector         Target Result
    # [3.0,1.5]            0
    # [2.0,1.0]            1
    # [4.0,1.5]            0
    # [3.0,4.0]            1
    # [3.5,0.5]            0
    # [2.0,0.5]            1
    # [5.5,1.0]            1
    # [1.0,1.0]            0
    input_vectors=np.array([[3,1.5],[2,1],[4,1.5],[3,4],[3.5,0.5],[2,0.5],[5.5,1],[1,1]])
    targets=np.array([0,1,0,1,0,1,1,0])

    learning_rate=0.01
    iterations=10000
    neural_network=NeuralNetwork(learning_rate)
    training_error=neural_network.train(input_vectors,targets,iterations)

    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.savefig("cum_error.png")

    print(neural_network.bias)
    print(neural_network.weights)
    print(training_error[len(training_error)-1])

    for index in range(len(input_vectors)):
        input_vector=input_vectors[index]
        target=targets[index]
        result=neural_network.predict(input_vector)
        correct=(result>0.5 and target==1)

        print(f'Input: {input_vector[0]:.2f} {input_vector[1]:.2f} Result: {result:.2f} Target: {target:.2f} Correct: {correct}')

