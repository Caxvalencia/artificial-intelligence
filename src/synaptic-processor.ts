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
    const output = this.activationFunction.activation(this.synapse);

    this.assertFinite(output, 'Activation output');

    return output;
  }

  /**
   * @param {Float64Array} weights
   */
  recalculateWeights(weights: Float64Array) {
    const error = this.outputExpected - this.output();
    this.delta = this.learningRate * error;
    this.assertFinite(this.delta, 'Weight update delta');

    for (let i = 0; i < weights.length; i++) {
      const updatedWeight = weights[i] + this.data[i] * this.delta;

      this.assertFinite(updatedWeight, `Weight ${i}`);
      weights[i] = updatedWeight;
    }
  }

  /**
   * @param {Float64Array} weights
   * @returns
   */
  calculateSynapses(weights: Float64Array, threshold: number) {
    if (this.data.length !== weights.length) {
      throw new RangeError(
        `Data dimension must be ${weights.length}; received ${this.data.length}`
      );
    }

    this.synapse = 0;

    for (let i = 0; i < weights.length; i++) {
      this.synapse += this.data[i] * weights[i];
    }

    this.synapse += threshold;
    this.assertFinite(this.synapse, 'Synaptic calculation');

    return this;
  }

  calculateError() {
    this.error = this.outputExpected - this.output();
    this.assertFinite(this.error, 'Prediction error');

    return this;
  }

  setData(data: Float64Array) {
    this.data = data.slice();

    return this;
  }

  setOutputExpected(expectedOutput: number) {
    this.outputExpected = expectedOutput;

    return this;
  }

  setLearningFactor(learningFactor: number) {
    if (!Number.isFinite(learningFactor) || learningFactor <= 0) {
      throw new RangeError('Learning factor must be a positive finite number');
    }

    this.learningRate = learningFactor;

    return this;
  }

  private assertFinite(value: number, label: string) {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${label} produced a non-finite value`);
    }
  }
}
