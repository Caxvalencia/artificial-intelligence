import { ActivationFunctionType } from './activation-functions/activation-function';
import { Neuron } from './neuron';

export class Layer {
  neurons: Neuron[];

  constructor(
    size: number,
    activationFunction: ActivationFunctionType,
    learningRate: number,
    momentum: number,
    random: () => number
  ) {
    if (!Number.isInteger(size) || size < 1) {
      throw new RangeError('A layer must contain at least one neuron');
    }

    this.neurons = Array.from(
      { length: size },
      () => new Neuron(activationFunction, learningRate, momentum, random)
    );
  }

  get size(): number {
    return this.neurons.length;
  }

  get initialized(): boolean {
    return this.neurons.every((neuron) => neuron.weights !== null);
  }

  forward(input: Float64Array): Float64Array {
    return Float64Array.from(this.neurons, (neuron) => neuron.forward(input));
  }

  calculateOutputDeltas(targets: Float64Array): number {
    let loss = 0;

    for (let index = 0; index < this.neurons.length; index++) {
      loss += this.neurons[index].calculateOutputDelta(targets[index]);
    }

    return loss / this.neurons.length;
  }

  calculateHiddenDeltas(nextLayer: Layer): void {
    for (let index = 0; index < this.neurons.length; index++) {
      this.neurons[index].calculateHiddenDelta(nextLayer.neurons, index);
    }
  }

  update(): void {
    this.neurons.forEach((neuron) => neuron.update());
  }

  setParameters(weights: number[][], biases: number[]): void {
    this.neurons.forEach((neuron, index) => neuron.setParameters(weights[index], biases[index]));
  }

  exportWeights(): number[][] {
    return this.neurons.map((neuron) => {
      if (!neuron.weights) {
        throw new Error('Cannot export an untrained network');
      }

      return Array.from(neuron.weights);
    });
  }

  exportBiases(): number[] {
    return this.neurons.map((neuron) => neuron.bias);
  }
}
