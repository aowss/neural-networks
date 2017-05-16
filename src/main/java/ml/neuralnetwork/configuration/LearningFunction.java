package ml.neuralnetwork.configuration;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Created by aowss.ibrahim on 2017-05-04.
 */
public interface LearningFunction {

    /**
     *
     * @param error the error
     * @param weight the current weight
     * @param in the current input
     * @return the new weight
     */
    public double update(double error, double weight, double in);

}

enum LearningFunctionEnum implements LearningFunction {

    DELTA_RULE(learningRate -> error -> (weight, in) -> weight + learningRate * error * in, () -> 0.2);

    private Function<Double, BiFunction<Double, Double, Double>> function;

    LearningFunctionEnum(Function<Double, Function<Double, BiFunction<Double, Double, Double>>> function, Supplier<Double> learningRate) {
        this.function = function.apply(learningRate.get());
    }

    @Override
    public double update(double error, double weight, double in) {
        return function.apply(error).apply(weight, in);
    }

}