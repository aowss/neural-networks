package ml.neuralnetwork;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by aowss.ibrahim on 2017-04-20.
 */
public class Engine {

    private static final double learningRate = 0.01;
    private static final Logger logger = Logger.getLogger("Engine");

    static {
        logger.setLevel(Level.FINEST);
    }

    public static BiFunction<Double[], Function<Double,Double>, Double[]> computeWeights = (initialWeights, computeFunction) -> Stream.of(initialWeights).
            map(computeFunction::apply).
            collect(Collectors.toList()).
            toArray(new Double[initialWeights.length]);

    //  get a value from a set of input and a set of weights
    public static BiFunction<double[], double[], Double> inputCombiner = (input, weights) -> Utils.weightedInput.apply(input).apply(weights);  //  e.g. weighted input
    public static Function<Double, Double> activationFunction = weightedInput -> ActivatationFunctions.sigmoid.apply(weightedInput);  //  e.g. sigmoid

    //  calculate the error based on the expected value for the given training set
    //public static BiFunction<Double, Double, Double> residualError = (output, expectedOutput) -> Math.pow(output - expectedOutput, 2) / 2;
    public static BiFunction<Double, Double, Double> residualError = (output, expectedOutput) -> expectedOutput - output;  //  e.g. diff

    //  learn, i.e. modify the weights for the next iteration ( next set of input )
    public static Function<Double, Function<Double, BiFunction<double[], double[], double[]>>> calculateNewWeights = learningRate -> error -> (input, currentWeights) -> {      // delta rule
        BiFunction<Double, Double, Double> delta = (weight, in) -> weight + learningRate * error * in;
        return Utils.zip.apply(delta).apply(currentWeights,input);
    };

    public static double[] process(TrainingSet input) {
        //logger.info("processing " + input);

        double[] weights = new double[input.getSample(0).getSampleSize()];
        Arrays.fill(weights, 0.5);
        logger.info("initial weights " + Arrays.asList(weights));

        for (TrainingSet.TrainingSample sample : input.getSamples()) {
            logger.info("processing : " + sample);
            logger.info("weighted input : " + inputCombiner.apply(sample.getInput(), weights));
            logger.info("output : " + activationFunction.apply(inputCombiner.apply(sample.getInput(), weights)));
            logger.info("target output : " + (Double)sample.getOutput());
            weights = calculateNewWeights.
                    apply(learningRate).
                    apply(residualError.apply(
                            activationFunction.apply(inputCombiner.apply(sample.getInput(), weights)),
                            (Double)sample.getOutput()
                    )).
                    apply(sample.getInput(), weights);
            logger.info("current weights " + Arrays.asList(weights));
        }
        logger.info("final weights " + Arrays.asList(weights));
        return weights;
    }

    public static void main(String[] args) {

        int sampleSize = 3;
        int inputSize = args.length != 0 ? Integer.valueOf(args[0]) * sampleSize : 1000 * sampleSize;
        double[][] input = new double[inputSize][sampleSize];
        double[] target = new double[inputSize];

        //Double[] perfectWeights = new Double[] { 150.0, 50.0, 100.0};
        Double[] perfectWeights = new Double[] { 0.15, 0.05, 0.1};

        Random random = new Random();
        for (int i = 0; i < inputSize; i++) {
            double expectedValue = 0.0;
            for (int j = 0; j < sampleSize; j++) {
                //input[i][j] = new Double(random.nextInt(1000));
                input[i][j] = random.nextInt(1000) / 1000.0;
                expectedValue += input[i][j] * perfectWeights[j];
            }
            target[i] = expectedValue / 3;
        }
/*
        IntStream stream = random.ints(inputSize);
        Double[] input1 = new Double[] { 2.0, 5.0, 3.0};
        double target1 = 850;
        Double[] input2 = new Double[] { 3.0, 2.0, 1.0};
        double target2 = 650;
        Double[] input3 = new Double[] { 0.0, 2.0, 2.0};
        double target3 = 300;
        Double[] input4 = new Double[] { 3.0, 3.0, 3.0};
        double target4 = 900;
        Double[][] input = new Double[4][3];
        input[0] = input1;
        input[1] = input2;
        input[2] = input3;
        input[3] = input4;
        Double[] target = new Double[4];
        target[0] = target1;
        target[1] = target2;
        target[2] = target3;
        target[3] = target4;
*/
        TrainingSet trainingSet = new TrainingSet(input, target);
        process(trainingSet);
    }
}
