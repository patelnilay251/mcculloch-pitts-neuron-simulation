class ActivationFunction:
    def activate(self, weighted_sum):
        raise NotImplementedError("Subclasses must implement the activate method")


class StepActivation(ActivationFunction):
    def activate(self, weighted_sum):
        return 1 if weighted_sum >= 0 else 0


class Neuron:
    def __init__(self, activation_function, inputs, weights, biases):
        self.activation_function = activation_function
        self.inputs = inputs
        self.weights = weights
        self.biases = biases

    def activate(self):
        if len(self.inputs) != len(self.weights):
            raise ValueError("Number of inputs must match the number of weights")

        # Calculate the weighted sum and apply activation function
        weighted_sum = sum(x * w for x, w in zip(self.inputs, self.weights))
        weighted_sum += sum(self.biases)
        output = self.activation_function.activate(weighted_sum)

        return output


class Inputs:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs

    def get_inputs(self):
        return [float(input(f"Enter input {i + 1}: ")) for i in range(self.num_inputs)]


class Weights:
    def __init__(self, num_weights):
        self.num_weights = num_weights

    def get_weights(self):
        return [
            float(input(f"Enter weight {i + 1}: ")) for i in range(self.num_weights)
        ]


class Biases:
    def __init__(self, num_biases):
        self.num_biases = num_biases

    def get_biases(self):
        return [float(input(f"Enter bias {i + 1}: ")) for i in range(self.num_biases)]


# Example usage:
if __name__ == "__main__":
    # User inputs for the number of inputs, weights, and biases
    num_inputs = int(input("Enter the number of inputs: "))
    num_weights = int(input("Enter the number of weights: "))
    num_biases = int(input("Enter the number of biases: "))

    # num_inputs = 8
    # num_weights = 8
    # num_biases = 1

    # Instantiate input, weights, biases, and activation function objects
    inputs = Inputs(num_inputs)
    weights = Weights(num_weights)
    biases = Biases(num_biases)
    activation_function = StepActivation()

    # Get user inputs
    input_values = inputs.get_inputs()
    weight_values = weights.get_weights()
    bias_values = biases.get_biases()

    # input_values = [1, 0, 1, 1, 0, 1, 0, 1]
    # weight_values = [0.2] * 8  # All weights are set to 0.2
    # bias_values = [-0.8]
    # Instantiate the neuron with the provided inputs, weights, and biases
    neuron = Neuron(activation_function, input_values, weight_values, bias_values)

    # Activate the neuron and print the result
    output = neuron.activate()
    print(f"Output: {output}")
