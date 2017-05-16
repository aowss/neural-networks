package ml.neuralnetwork.configuration;

import java.util.function.BiFunction;

/**
 * A function that determines the effective input of a neuron from its external inputs
 * Created by aowss.ibrahim on 2017-05-04.
 */
public interface PropagationFunction extends BiFunction<double[], double[], Double> {
}

enum PropagationFunctionEnum implements PropagationFunction {

    WEIGHTED_SUM((input, weights) ->  {

        if (input == null) throw new RuntimeException("The input can't be null");
        if (input.length < 3) throw new RuntimeException("The input array must contain at least two values : one input and the bias");
        if (weights == null) throw new RuntimeException("The weigths can't be null");
        if (weights.length != input.length) throw new RuntimeException("The input array and the weight array must be of the same size");

        double result = 0;
        for (int i = 0; i < input.length; i++) {
            result += input[i] * weights[i];
        }
        return result;

    });

    private BiFunction<double[], double[], Double> function;

    PropagationFunctionEnum(BiFunction<double[], double[], Double> function) {
        this.function = function;
    }

    @Override
    public Double apply(double[] inputs, double[] weights) {
        return function.apply(inputs, weights);
    }

}