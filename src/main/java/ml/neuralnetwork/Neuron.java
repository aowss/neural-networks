package ml.neuralnetwork;

import ml.neuralnetwork.configuration.ActivationFunction;
import ml.neuralnetwork.configuration.PropagationFunction;
import ml.neuralnetwork.configuration.WeightInitializationFunction;

import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * Each neuron performs a relatively simple job :
 * <ol>
 *     <li>receive input from neighbours or external sources and use this to compute an output signal which is propagated to other units</li>
 *     <li>adjust its weights</li>
 * </ol>
 * Created by aowss.ibrahim on 2017-05-04.
 */
public class Neuron implements Function<double[], Double> {

    private final ActivationFunction activationFunction;
    private final PropagationFunction propagationFunction;
    private final WeightInitializationFunction weightInitializationFunction;

    private double[] currentWeights;

//    @Inject
    public Neuron(ActivationFunction activationFunction, PropagationFunction propagationFunction, WeightInitializationFunction weightInitializationFunction) {
        this.activationFunction = activationFunction;
        this.propagationFunction = propagationFunction;
        this.weightInitializationFunction = weightInitializationFunction;
    }

    //  TODO: make it thread safe
    @Override
    public Double apply(double[] input) {
        if (currentWeights == null) initializeWeights(input.length);
        return activationFunction.apply(
                propagationFunction.apply(input, currentWeights)
        );
    }

    private void initializeWeights(int length) {
        //DoubleStream.generate(weightInitializationFunction).limit(length).collect(Collectors::toList)
        currentWeights = new double[length];
        for (int i = 0; i < length; i++) {
            currentWeights[i] = weightInitializationFunction.getAsDouble();
        }
    }

    //  TODO: Check if the adjustment of the weight is the responsibility of the neuron as suggested by the description or the responsibility of the network
    //  TODO: Should this adjustment be a side effect of the apply method ?
}
