package ml.neuralnetwork.configuration;

import java.util.Arrays;
import java.util.function.BiFunction;

import static ml.neuralnetwork.Utils.zip;

/**
 * A function that determines the effective input of a neuron from its external inputs
 * Created by aowss.ibrahim on 2017-05-04.
 */
public interface ErrorFunction extends BiFunction<double[], double[], Double> {
}

enum ErrorFunctionEnum implements ErrorFunction {

    SQUARE {

        public Double apply(double[] output, double[] targetOutput) {

            if (output == null) throw new RuntimeException("The output can't be null");
            if (targetOutput == null) throw new RuntimeException("The target output can't be null");
            if (output.length != targetOutput.length) throw new RuntimeException("The output array and the target output array must be of the same size");

            return 0.5 * Arrays.stream(zip.apply((Double a, Double b) -> a - b).apply(targetOutput, output)).map(diff -> Math.pow(diff, 2)).sum();

        }

    };

}