package ml.neuralnetwork;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Created by aowss.ibrahim on 2017-04-18.
 */
public class Utils {

    public static final Function<double[], Function<double[], Double>> weightedInput = input -> weights -> {

        if (input == null) throw new RuntimeException("The input can't be null");
        if (input.length < 3) throw new RuntimeException("The input array must contain at least two values : one input and the bias");
        if (weights == null) throw new RuntimeException("The weigths can't be null");
        if (weights.length != input.length) throw new RuntimeException("The input array and the weight array must be of the same size");

        double result = 0;
        for (int i = 0; i < input.length; i++) {
            result += input[i] * weights[i];
        }
        return result;

    };

    public static final Function<BiFunction<Double,Double,Double>, BiFunction<double[], double[], double[]>> zip = computeFunction -> (input1, input2) -> {
        if (input1 == null || input2 == null) throw new RuntimeException("The inputs can't be null");
        if (input1.length != input2.length) throw new RuntimeException("Both inputs must be of the same size");
        double[] result = new double[input1.length];
        for (int i = 0; i < input1.length; i++) {
            result[i] = computeFunction.apply(input1[i], input2[i]);
        }
        return result;
    };

    public static final <I1,I2,O> O[] zip(I1[] input1, I2[] input2, BiFunction<I1,I2,O> computeFunction) {
        if (input1 == null || input2 == null) throw new RuntimeException("The inputs can't be null");
        if (input1.length != input2.length) throw new RuntimeException("Both inputs must be of the same size");
        O[] result = (O[])new Object[input1.length];
        for (int i = 0; i < input1.length; i++) {
            result[i] = computeFunction.apply(input1[i], input2[i]);
        }
        return result;
    };

}
