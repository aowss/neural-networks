package ml.neuralnetwork;

import ml.neuralnetwork.configuration.ActivationFunction;
import ml.neuralnetwork.configuration.ErrorFunction;
import ml.neuralnetwork.configuration.IntegrationFunction;
import ml.neuralnetwork.configuration.WeightInitializationFunction;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

import static ml.neuralnetwork.Utils.zip;

/**
 * An artificial network consists of a pool of simple processing units which communicate by sending signals to each other over a large number of weighted connections.
 * The system is inherently parallel in the sense that many units can carry out their computations at the same time.
 * Created by aowss.ibrahim on 2017-05-09.
 */
public class Network {

    //  The input layer is not part of the network
    private final Neuron[][] network;
    private final int inputSize;
    private final double learningRate;
    private final ErrorFunction errorFunction;

    private Network(int inputSize, int[] numberOfNeurons, double learningRate, ActivationFunction activationFunction, IntegrationFunction integrationFunction, ErrorFunction errorFunction, WeightInitializationFunction weightInitializationFunction, WeightInitializationFunction biasInitializationFunction) {

        network = new Neuron[numberOfNeurons.length][];
        this.inputSize = inputSize;
        this.learningRate = learningRate;
        this.errorFunction = errorFunction;

        int layerIndex = 0;
        for (int layerSize : numberOfNeurons) {
            network[layerIndex] = new Neuron[layerSize];
            for (int neuronIndex = 0; neuronIndex < layerSize; neuronIndex++) {
                int neuronInputSize = layerIndex == 0 ? inputSize : network[layerIndex].length;
                network[layerIndex][neuronIndex] = new Neuron(activationFunction, integrationFunction, initializeWeights(neuronInputSize, weightInitializationFunction), biasInitializationFunction.getAsDouble());
            }
            layerIndex++;
        }

    }

    private double[] initializeWeights(int length, WeightInitializationFunction weightInitializationFunction) {
        double[] weights = new double[length];
        for (int i = 0; i < length; i++) {
            weights[i] = weightInitializationFunction.getAsDouble();
        }
        return weights;
    }

    //  Supervised Learning ( corrective learning )
    public void train(TrainingSet trainingSet) {

        if (trainingSet.getInputSize() != inputSize) throw new RuntimeException("The training data size [ " + trainingSet.getInputSize() + " ] doesn't match what is expected by the network [ " + inputSize + " ]");

        for (TrainingSet.TrainingSample sample : trainingSet.getSamples()) {

            //  Forward Propagation : Get networks' output for the given sample
            double[] output = process.apply(network, 0).apply(sample.getInput());
            assert output.length == network[network.length - 1].length;

            //  Cost calculation
            double error = errorFunction.apply(output, sample.getOutput());
        }

    }

    private Function<Neuron[], Function<double[], double[]>> propagate = layer -> input -> {
        double[] output = new double[layer.length];
        for (int neuron = 0; neuron < layer.length; neuron++) {
            output[neuron] = layer[neuron].apply(input);
        }
        return output;
    };

    private BiFunction<Neuron[][], Integer, Function<double[], double[]>> process = (network, layerIndex) -> input -> {
        if (layerIndex < network.length - 1) {
            this.process.apply(network, layerIndex + 1).apply(propagate.apply(network[layerIndex]).apply(input));
        }
        return propagate.apply(network[layerIndex]).apply(input);
    };

    public class Builder {

        private ActivationFunction activationFunction;
        private IntegrationFunction integrationFunction;
        private ErrorFunction errorFunction;
        private WeightInitializationFunction weightInitializationFunction = () -> 0.5;
        private WeightInitializationFunction biasInitializationFunction = () -> 0.5;

        private int inputSize;
        private int[] numberOfNeurons;
        private double learningRate;

        public Builder withActivationFunction(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder withIntegrationFunction(IntegrationFunction integrationFunction) {
            this.integrationFunction = integrationFunction;
            return this;
        }

        public Builder withErrorFunction(ErrorFunction errorFunction) {
            this.errorFunction = errorFunction;
            return this;
        }

        public Builder withWeightInitializationFunction(WeightInitializationFunction weightInitializationFunction) {
            this.weightInitializationFunction = weightInitializationFunction;
            return this;
        }

        public Builder withBiasInitializationFunction(WeightInitializationFunction biasInitializationFunction) {
            this.biasInitializationFunction = biasInitializationFunction;
            return this;
        }

        public Builder withLayers(int... numberOfNeurons) {
            if (numberOfNeurons == null || numberOfNeurons.length == 0) throw new RuntimeException("The network should have at least one layer");
            this.numberOfNeurons = numberOfNeurons;
            return this;
        }

        public Builder withInputSize(int inputSize) {
            if (inputSize < 2) throw new RuntimeException("The network should have at least 2 inputs");
            this.inputSize = inputSize;
            return this;
        }

        public Builder withLearningRate(double learningRate) {
            if (learningRate <= 0 || learningRate >= 1) throw new RuntimeException("The learning rate should be between 0.0 and 1.0 exclusive");
            this.learningRate = learningRate;
            return this;
        }

        //  TODO: add validation
        public Network build() {
            return new Network(inputSize, numberOfNeurons, learningRate, activationFunction, integrationFunction, errorFunction, weightInitializationFunction, biasInitializationFunction);
        }

    }

}
