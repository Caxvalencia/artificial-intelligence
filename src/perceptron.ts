import { ActivationFunctionType } from './activation-functions/activation-function';
import { SynapticProcessor } from './synaptic-processor';

interface PerceptronTrainingStats {
  epochs: number;
  errors: number;
  converged: boolean;
}

export class Perceptron {
  dataStack: any[];
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

  addData(data: Float64Array, output: number) {
    this.dataStack.push([data, output]);

    return this;
  }

  learn() {
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

  process(data: Float64Array) {
    return this.synapticProcessor
      .setData(data)
      .calculateSynapses(this.weights, this.threshold)
      .output();
  }

  setWeights(weights: Float64Array) {
    this.weights = weights;

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
}
