import { ActivationFunctionType } from './activation-functions/activation-function';
import { SynapticProcessor } from './synaptic-processor';
import { PerceptronTrainingStats } from './types';

type PerceptronSample = [Float64Array, number];

export class Perceptron {
  dataStack: PerceptronSample[];
  weights: Float64Array;
  threshold: number;
  synapticProcessor: SynapticProcessor;
  maxEpochs: number;
  trainingStats: PerceptronTrainingStats;

  funcBack: () => void;

  constructor(activationFunction?: ActivationFunctionType, callback = () => {}) {
    this.weights = null;
    this.funcBack = callback;

    this.synapticProcessor = new SynapticProcessor(activationFunction);
    this.dataStack = [];
    this.maxEpochs = 8000;
    this.trainingStats = { epochs: 0, errors: 0, converged: false };
  }

  addData(data: ArrayLike<number>, output: number) {
    const normalizedData = this.normalizeData(data, 'Training data');

    if (output !== 0 && output !== 1) {
      throw new RangeError('Perceptron output must be either 0 or 1');
    }

    if (this.dataStack.length > 0 && normalizedData.length !== this.dataStack[0][0].length) {
      throw new RangeError(
        `Training data dimension must be ${this.dataStack[0][0].length}; received ${normalizedData.length}`
      );
    }

    this.dataStack.push([normalizedData, output]);

    return this;
  }

  learn() {
    if (this.dataStack.length === 0) {
      throw new Error('Perceptron requires at least one training sample');
    }

    if (!this.weights) {
      this.assignWeights();
    }

    this.trainingStats = { epochs: 0, errors: 0, converged: false };

    for (let epoch = 1; epoch <= this.maxEpochs; epoch++) {
      let errors = 0;

      for (let i = 0; i < this.dataStack.length; i++) {
        this.synapticProcessor
          .setData(this.dataStack[i][0])
          .setOutputExpected(this.dataStack[i][1])
          .calculateSynapses(this.weights, this.threshold)
          .calculateError();

        if (this.synapticProcessor.error !== 0) {
          this.synapticProcessor.recalculateWeights(this.weights);
          this.threshold += this.synapticProcessor.delta;

          errors++;
          this.funcBack();
        }
      }

      this.trainingStats = {
        epochs: epoch,
        errors,
        converged: errors === 0
      };

      if (this.trainingStats.converged) {
        return this;
      }
    }

    throw new Error(
      `Perceptron did not converge after ${this.maxEpochs} epochs; ` +
        `last epoch errors: ${this.trainingStats.errors}`
    );
  }

  process(data: ArrayLike<number>) {
    if (!this.weights || !Number.isFinite(this.threshold)) {
      throw new Error('Perceptron must be trained or configured before processing data');
    }

    const normalizedData = this.normalizeData(data, 'Process data');

    if (normalizedData.length !== this.weights.length) {
      throw new RangeError(
        `Process data dimension must be ${this.weights.length}; received ${normalizedData.length}`
      );
    }

    return this.synapticProcessor
      .setData(normalizedData)
      .calculateSynapses(this.weights, this.threshold)
      .output();
  }

  setWeights(weights: ArrayLike<number>) {
    this.weights = this.normalizeData(weights, 'Weights');

    return this;
  }

  setMaxEpochs(maxEpochs: number) {
    if (!Number.isInteger(maxEpochs) || maxEpochs < 1) {
      throw new RangeError('maxEpochs must be a positive integer');
    }

    this.maxEpochs = maxEpochs;

    return this;
  }

  protected createWeight() {
    const rangeWeight = { MIN: -0.5, MAX: 0.49 };
    const rangeDiff = rangeWeight.MAX - rangeWeight.MIN;
    let weight = 0;

    while (!weight) {
      weight = parseFloat((Math.random() * rangeDiff + rangeWeight.MIN).toFixed(4));
    }

    return weight;
  }

  protected assignWeights(dataSize: number = null) {
    if (!dataSize) {
      dataSize = this.dataStack[0][0].length;
    }

    const weights = new Float64Array(dataSize);

    for (let i = 0; i < dataSize; i++) {
      weights[i] = this.createWeight();
    }

    this.setWeights(weights);
    this.threshold = this.createWeight();

    this.funcBack();

    return this;
  }

  private normalizeData(data: ArrayLike<number>, label: string): Float64Array {
    if (data == null || typeof data.length !== 'number' || data.length === 0) {
      throw new RangeError(`${label} must contain at least one value`);
    }

    const normalizedData = new Float64Array(data);

    for (let index = 0; index < normalizedData.length; index++) {
      if (!Number.isFinite(normalizedData[index])) {
        throw new RangeError(`${label} must contain only finite numbers`);
      }
    }

    return normalizedData;
  }
}
