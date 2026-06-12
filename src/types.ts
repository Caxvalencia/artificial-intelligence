import { ActivationFunctionType } from './activation-functions/activation-function';

export type NumericArray = ArrayLike<number>;

export interface TrainingSample {
  input: NumericArray;
  output: number | NumericArray;
}

export interface BackpropagationConfig {
  epochs?: number;
  activationFunction?: ActivationFunctionType;
  learningRate?: number;
  momentum?: number;
  seed?: number;
  shuffle?: boolean;
  targetLoss?: number;
  patience?: number;
  verbose?: boolean;
}

export interface BackpropagationHistory {
  loss: number[];
  epochs: number;
  stoppedEarly: boolean;
}

export interface SerializedModel {
  version: 1;
  config: {
    activationFunction: ActivationFunctionType;
    learningRate: number;
    momentum: number;
  };
  layers: number[];
  biases: number[][];
  weights: number[][][];
}

export interface LegacySerializedModel {
  layers: number[];
  thresholds: number[][];
  weights: number[][][];
}

export interface PerceptronTrainingStats {
  epochs: number;
  errors: number;
  converged: boolean;
}
