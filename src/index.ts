import { ActivationFunctionType } from './activation-functions/activation-function';
import { Backpropagation } from './backpropagation';
import { Perceptron } from './perceptron';

export { Perceptron, Backpropagation, ActivationFunctionType };

export type {
  BackpropagationConfig,
  BackpropagationHistory,
  NumericArray,
  PerceptronTrainingStats,
  SerializedModel,
  TrainingSample
} from './types';
