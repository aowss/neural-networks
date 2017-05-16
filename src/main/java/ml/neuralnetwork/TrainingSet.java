package ml.neuralnetwork;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * Created by aowss.ibrahim on 2017-04-18.
 */
public class TrainingSet {

    private TrainingSample[] samples;

    public class TrainingSample {

        private double[] input;
        private double output;

        private TrainingSample(double[] input, double output) {
            //  Adding the bias, a.k.a offset
            this.input = new double[input.length + 1];
            this.input[0] = 1.0;
            System.arraycopy(input, 0, this.input, 1, input.length);
            this.output = output;
        }

        public int getSampleSize() {
            return input.length;
        }

        public double[] getInput() {
            return input;
        }

        public double getOutput() {
            return output;
        }

        @Override
        public String toString() {
            return  "{ \"input\" : \""
                    + DoubleStream.of(input).mapToObj(Double::toString).collect(Collectors.joining(", ", "[", "]"))
                    + "\", \"output\" : \"" + output + "\" }";
        }

    }

    public TrainingSet(double[][] input, double[] output) {

        if (input == null || input.length == 0) throw new RuntimeException("The training set's input data can't be null or empty");
        if (output == null || output.length == 0) throw new RuntimeException("The training set's output data can't be null or empty");
        if (input.length != output.length) throw new RuntimeException("The training set's input and output sizes must be identical");

        samples = new TrainingSample[input.length];
        for (int i = 0; i < input.length; i++) {
            samples[i] = new TrainingSample(input[i], output[i]);
        }

    }

    public TrainingSample getSample(int sampleNumber) {
        return samples[sampleNumber];
    }

    public TrainingSample[] getSamples() {
        return samples;
    }

    public int getTrainingSetSize() {
        return samples.length;
    }

    //  This assumes that the samples are all the same size
    public int getInputSize() {
        return samples[0].getSampleSize();
    }

    @Override
    public String toString() {
        return  Arrays.stream(samples).map(TrainingSample::toString).collect(Collectors.joining(",", "[", "]"));
    }

}
