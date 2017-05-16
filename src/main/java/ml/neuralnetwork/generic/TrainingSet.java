package ml.neuralnetwork.generic;

import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * Created by aowss.ibrahim on 2017-04-18.
 */
public class TrainingSet<I,O> {

    private TrainingSample<I,O>[] samples;

    public class TrainingSample<I,O> {

        private I[] input;
        private O output;

        private TrainingSample(I[] input, O output) {
            this.input = input;
            this.output = output;
        }

        public int getSampleSize() {
            return input.length;
        }

        public I[] getInput() {
            return input;
        }

        public O getOutput() {
            return output;
        }

        @Override
        public String toString() {
            return  "{ \"input\" : \""
                    + Arrays.stream(input).map(Object::toString).collect(Collectors.joining(",", "[", "]"))
                    + "\", \"output\" : \"" + output + "\" }";
        }

    }

    public TrainingSet(I[][] input, O[] output) {

        if (input == null || input.length == 0) throw new RuntimeException("The training set's input data can't be null or empty");
        if (output == null || output.length == 0) throw new RuntimeException("The training set's output data can't be null or empty");
        if (input.length != output.length) throw new RuntimeException("The training set's input and output sizes must be identical");

        samples = new TrainingSample[input.length];
        for (int i = 0; i < input.length; i++) {
            samples[i] = new TrainingSample<>(input[i], output[i]);
        }

    }

    public TrainingSample<I,O> getSample(int sampleNumber) {
        return samples[sampleNumber];
    }

    public TrainingSample<I,O>[] getSamples() {
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
