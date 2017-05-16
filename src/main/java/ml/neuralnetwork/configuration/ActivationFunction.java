package ml.neuralnetwork.configuration;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * A function that determines the new level of activation based on the effective input and the current activation.
 * Note that the activation functions don't seem to take the current activation into account.
 * Created by aowss.ibrahim on 2017-05-02.
 */
public interface ActivationFunction extends Function<Double, Double> {
}

enum ActivationFunctionEnum implements ActivationFunction {

    SIGMOID(input -> 1.0 / (1.0 + Math.exp(-input))),
    BINARY_THRESHOLD(treshold -> input -> input >= treshold ? 1.0 : 0.0, () -> 2.0),
    LINEAR_THRESHOLD(input -> input > 0 ? input : 0.0);

    private Function<Double, Double> function;

    ActivationFunctionEnum(Function<Double, Double> function) {
        this.function = function;
    }

    ActivationFunctionEnum(Function<Double, Function<Double, Double>> function, Supplier<Double> threshold) {
        this.function = function.apply(threshold.get());
    }

    @Override
    public Double apply(Double in) {
        return function.apply(in);
    }

}