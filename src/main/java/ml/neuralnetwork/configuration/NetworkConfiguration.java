package ml.neuralnetwork.configuration;

import java.util.function.DoubleSupplier;
import java.util.function.Supplier;

/**
 * Created by aowss.ibrahim on 2017-05-08.
 */
public class NetworkConfiguration {

    private Supplier<ActivationFunction> activationFunctionSupplier;
    private Supplier<IntegrationFunction> integrationFunctionSupplier;
    private Supplier<LearningFunction> learningFunctionSupplier;
    private Supplier<WeightInitializationFunction> weightInitializationFunctionSupplier;
    private DoubleSupplier tresholdSupplier;
    private DoubleSupplier learningRateSupplier;

}
