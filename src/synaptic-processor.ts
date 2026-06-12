import {
  ActivationFunction,
  ActivationFunctionType
} from './activation-functions/activation-function';

export class SynapticProcessor {
  activationFunction: ActivationFunction;
  error: number;
  synapse: number;
  data: Float64Array;
  learningRate: number;
  delta: number;
  outputExpected: number;

  constructor(activationFunction: ActivationFunctionType, learningRate: number = 0.3) {
    this.error = 0;
    this.activationFunction = ActivationFunction.init(activationFunction);
    this.setLearningFactor(learningRate);
  }

  /**
   * @returns {number}
   */
  output(): number {
    return this.activationFunction.activation(this.synapse);
  }

  /**
   * @param {Float64Array} weights
   */
  recalculateWeights(weights: Float64Array) {
    const error = this.outputExpected - this.output();
    this.delta = this.learningRate * error;

    for (let i = 0; i < weights.length; i++) {
      weights[i] += this.data[i] * this.delta;
    }
  }

  /**
   * @param {Float64Array} weights
   * @returns
   */
  calculateSynapses(weights: Float64Array, threshold) {
    this.synapse = 0;

    for (let i = 0; i < weights.length; i++) {
      this.synapse += this.data[i] * weights[i];
    }

    this.synapse += threshold;

    return this;
  }

  calculateError() {
    this.error = this.outputExpected - this.output();

    return this;
  }

  setData(data: Float64Array) {
    this.data = data.slice();

    return this;
  }

  setOutputExpected(expectedOutput) {
    this.outputExpected = expectedOutput;

    return this;
  }

  setLearningFactor(learningFactor) {
    this.learningRate = learningFactor;

    return this;
  }
}
