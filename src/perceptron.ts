import { ActivationFunctionType } from './activation-functions/activation-function';
import { SynapticProcessor } from './synaptic-processor';

export class Perceptron {
  dataStack: any[];
  weights: Float64Array;
  threshold: number;
  synapticProcessor: SynapticProcessor;

  funcBack: () => void;

  constructor(activationFunction?: ActivationFunctionType, callback = () => {}) {
    this.weights = null;
    this.funcBack = callback;

    this.synapticProcessor = new SynapticProcessor(activationFunction);
    this.dataStack = [];
  }

  addData(data: Float64Array, output: number) {
    this.dataStack.push([data, output]);

    return this;
  }

  learn() {
    if (!this.weights) {
      this.assignWeights();
    }

    const LIMIT_ERRORS: number = 8000;
    let counterErrors = 0;

    const runEpochs = () => {
      let hasError = false;

      for (let i = 0; i < this.dataStack.length; i++) {
        this.synapticProcessor
          .setData(this.dataStack[i][0])
          .setOutputExpected(this.dataStack[i][1])
          .calculateSynapses(this.weights, this.threshold)
          .calculateError();

        if (this.synapticProcessor.error !== 0) {
          this.synapticProcessor.recalculateWeights(this.weights);
          this.threshold += this.synapticProcessor.delta;

          hasError = true;
          this.funcBack();
        }
      }

      if (hasError) {
        counterErrors++;

        if (counterErrors >= LIMIT_ERRORS) {
          counterErrors = 0;

          throw Error('Maximum error limit reached');
        }

        return runEpochs();
      }

      counterErrors = 0;
    };

    runEpochs();

    return this;
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
