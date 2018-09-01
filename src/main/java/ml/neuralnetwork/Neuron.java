package ml.neuralnetwork;

import ml.neuralnetwork.configuration.ActivationFunction;
import ml.neuralnetwork.configuration.IntegrationFunction;

import java.util.Arrays;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import static ml.neuralnetwork.Utils.addValue;

/**
 * Neurons are computing units.
 * They are split into two functional parts:
 * <ol>
 *     <li>an integration function g reduces the n arguments to a single value</li>
 *     <li>an activation function f produces the output of this node taking that single value as its argument</li>
 * </ol>
 * Each neuron performs a relatively simple job :
 * <ol>
 *     <li>receive input from neighbours or external sources and use this to compute an output signal which is propagated to other units</li>
 *     <li>adjust its weights</li>
 * </ol>
 * Created by aowss.ibrahim on 2017-05-04.
 */
public class Neuron implements Function<double[], Double>, BiConsumer<Double, double[]> {

    private final ActivationFunction activationFunction;
    private final IntegrationFunction integrationFunction;

    /**
     * The last element in the array is the bias
     */
    private double[] currentWeights;

    /**
     * A Neuron is a statefull computing unit.
     * Its state is composed of its current weights as returned by {@link #getCurrentWeights}
     * @param activationFunction the neuron's activation function, e.g. sigmoid
     * @param integrationFunction the neuron's integration function, e.g. wieghted sum
     * @param initialWeights the neuron's initial weights as initialized by the network
     */
    public Neuron(ActivationFunction activationFunction, IntegrationFunction integrationFunction, double[] initialWeights, double bias) {
        this.activationFunction = activationFunction;
        this.integrationFunction = integrationFunction;
        currentWeights = addValue.apply(initialWeights, bias);
    }

    /**
     * Compute the neuron's output
     * @param input the input is the output of the previous layer or the input to the network as a whole.
     *              The size of the input is 1 less than the size of the {@link #getCurrentWeights weights} since an artificial input with a value of <code>1</code> is added to model the neuron's threshold as a bias
     * @return the neuron's output
     */
    @Override
    public Double apply(double[] input) {
        return activationFunction.apply(
                integrationFunction.apply(addValue.apply(input, 1.0), currentWeights)
        );
    }

    //  TODO: add getBias, getExtendedWeights
    //  TODO: make defensive copies

    public int getInputSize() {
        return currentWeights.length;
    }

    /**
     * The neuron's current weights
     * @return the current weights including the bias which is the last element in the array
     */
    public double[] getCurrentWeights() {
        return currentWeights;
    }

    /**
     * Updates the neuron's weights
     * @param error the error as calculated by the network
     * @param input the training sample
     */
    @Override
    public void accept(Double error, double[] input) {

    }

}
