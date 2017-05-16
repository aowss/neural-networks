package ml.neuralnetwork;

import java.util.function.Function;

/**
 * Created by aowss.ibrahim on 2017-04-18.
 */
public class ActivatationFunctions {

    public static final Function<Double, Double> sigmoid = input -> 1.0 / (1.0 + Math.exp(-input));

    public static final Function<Double, Function<Double, Double>> binaryThreshold = treshold -> input -> input >= treshold ? 1.0 : 0.0;

    public static final Function<Double, Double> linearThreshold = input -> input > 0 ? input : 0.0;

}
