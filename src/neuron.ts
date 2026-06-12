import { ActivationFunctionType } from './activation-functions/activation-function';
import { Perceptron } from './perceptron';

export class Neuron extends Perceptron {
  beforeWeights: Float64Array;
  error: number;
  inputNeurons: Neuron[];
  outputNeurons: Neuron[];
  synapse: number;
  currentData: Float64Array;

  /**
   * @construtor
   */
  constructor(activationFunction: ActivationFunctionType = ActivationFunctionType.SIGMOIDAL) {
    super(activationFunction);

    this.error = 0;
    this.synapse = 0;
    this.outputNeurons = [];
    this.inputNeurons = [];
  }

  learn(currentData?: Float64Array) {
    if (currentData) {
      this.synapticProcessor.data = currentData;
    }

    this.currentData = this.synapticProcessor.data;

    if (!this.weights) {
      this.assignWeights(this.currentData.length);
      this.setBeforeWeights(this.weights.slice());
    }

    this.synapticProcessor.calculateSynapses(this.weights, this.threshold);
    this.synapse = this.synapticProcessor.synapse;

    return this;
  }

  /**
   * Error on hidden layers
   */
  backpropagation() {
    for (let index = 0; index < this.inputNeurons.length; index++) {
      this.inputNeurons[index].calculateHiddenError(index);
    }

    if (this.inputNeurons.length > 0) {
      this.inputNeurons[0].backpropagation();
    }
  }

  recalculateWeights() {
    const delta = this.synapticProcessor.learningRate * this.error;
    const momentumFactor = 0.77;
    let deltaWeights: number = 0;

    for (let i = 0; i < this.weights.length; i++) {
      if (this.currentData[i] === 0) {
        continue;
      }

      deltaWeights = momentumFactor * (this.weights[i] - this.beforeWeights[i]);

      this.beforeWeights[i] = this.weights[i];
      this.weights[i] += this.currentData[i] * delta + deltaWeights;
    }

    this.threshold += delta;
  }

  output(): number {
    return this.synapticProcessor.activationFunction.activation(this.synapse);
  }

  process() {
    this.synapticProcessor.calculateSynapses(this.weights, this.threshold);
    this.synapse = this.synapticProcessor.synapse;

    return this.output();
  }

  calculateErrorOfOutput(outputExpected) {
    const error = outputExpected - this.output();

    this.calculateErrorDerivated(error);
  }

  calculateHiddenError(neuronIndex) {
    let sumError = 0;

    for (let index = 0; index < this.outputNeurons.length; index++) {
      const neuron = this.outputNeurons[index];
      sumError += neuron.weights[neuronIndex] * neuron.error;
    }

    this.calculateErrorDerivated(sumError);

    return this;
  }

  calculateErrorDerivated(factorDelta) {
    this.error = factorDelta * this.prime();

    return this;
  }

  /**
   * @returns {number}
   */
  prime(): number {
    return this.synapticProcessor.activationFunction.prime(this.synapse);
  }

  /**
   * @param {Float64Array} beforeWeights
   * @returns {this}
   */
  setBeforeWeights(beforeWeights: Float64Array): this {
    this.beforeWeights = beforeWeights;

    return this;
  }

  /**
   * @returns {this}
   */
  setThreshold(threshold: number): this {
    this.threshold = threshold;

    return this;
  }
}
