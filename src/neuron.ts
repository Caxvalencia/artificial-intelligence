import {
  ActivationFunction,
  ActivationFunctionType
} from './activation-functions/activation-function';

export class Neuron {
  weights: Float64Array | null;
  bias: number;
  output: number;
  delta: number;

  private activationFunction: ActivationFunction;
  private input: Float64Array | null;
  private previousWeightUpdates: Float64Array | null;
  private previousBiasUpdate: number;
  private learningRate: number;
  private momentum: number;
  private random: () => number;

  constructor(
    activationFunction: ActivationFunctionType,
    learningRate: number,
    momentum: number,
    random: () => number
  ) {
    this.activationFunction = ActivationFunction.init(activationFunction);
    this.learningRate = learningRate;
    this.momentum = momentum;
    this.random = random;
    this.weights = null;
    this.input = null;
    this.previousWeightUpdates = null;
    this.previousBiasUpdate = 0;
    this.bias = 0;
    this.output = 0;
    this.delta = 0;
  }

  initialize(inputSize: number): this {
    if (this.weights) {
      return this;
    }

    const limit = Math.sqrt(6 / inputSize);
    this.weights = new Float64Array(inputSize);
    this.previousWeightUpdates = new Float64Array(inputSize);

    for (let index = 0; index < inputSize; index++) {
      this.weights[index] = (this.random() * 2 - 1) * limit;
    }

    this.bias = (this.random() * 2 - 1) * limit;

    return this;
  }

  setParameters(weights: ArrayLike<number>, bias: number): this {
    this.weights = new Float64Array(weights);
    this.previousWeightUpdates = new Float64Array(weights.length);
    this.previousBiasUpdate = 0;
    this.bias = bias;

    return this;
  }

  forward(input: Float64Array): number {
    if (!this.weights) {
      this.initialize(input.length);
    }

    if (input.length !== this.weights.length) {
      throw new RangeError(
        `Neuron input dimension must be ${this.weights.length}; received ${input.length}`
      );
    }

    this.input = input;
    let synapse = this.bias;

    for (let index = 0; index < input.length; index++) {
      synapse += input[index] * this.weights[index];
    }

    this.assertFinite(synapse, 'Neuron synapse');
    this.output = this.activationFunction.activation(synapse);
    this.assertFinite(this.output, 'Neuron activation output');

    return this.output;
  }

  calculateOutputDelta(target: number): number {
    const difference = target - this.output;
    this.delta = difference * this.activationFunction.primeFromOutput(this.output);
    this.assertFinite(this.delta, 'Neuron output delta');

    return (difference * difference) / 2;
  }

  calculateHiddenDelta(nextLayer: Neuron[], neuronIndex: number): void {
    let propagatedError = 0;

    for (const neuron of nextLayer) {
      if (!neuron.weights) {
        throw new Error('Next layer must be initialized before calculating hidden deltas');
      }

      propagatedError += neuron.weights[neuronIndex] * neuron.delta;
    }

    this.delta = propagatedError * this.activationFunction.primeFromOutput(this.output);
    this.assertFinite(this.delta, 'Neuron hidden delta');
  }

  update(): void {
    if (!this.weights || !this.input || !this.previousWeightUpdates) {
      throw new Error('Neuron must complete a forward pass before updating weights');
    }

    for (let index = 0; index < this.weights.length; index++) {
      const update =
        this.learningRate * this.delta * this.input[index] +
        this.momentum * this.previousWeightUpdates[index];
      const weight = this.weights[index] + update;

      this.assertFinite(weight, `Neuron weight ${index}`);
      this.weights[index] = weight;
      this.previousWeightUpdates[index] = update;
    }

    const biasUpdate = this.learningRate * this.delta + this.momentum * this.previousBiasUpdate;
    this.bias += biasUpdate;
    this.assertFinite(this.bias, 'Neuron bias');
    this.previousBiasUpdate = biasUpdate;
  }

  private assertFinite(value: number, label: string): void {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${label} produced a non-finite value`);
    }
  }
}
