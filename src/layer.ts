import { ActivationFunctionType } from './activation-functions/activation-function';
import { Neuron } from './neuron';
import { SynapticProcessor } from './synaptic-processor';

export class Layer {
  private layers: Neuron[][];
  private activationFunction: ActivationFunctionType;

  synapticProcessor: SynapticProcessor;

  constructor(
    activationFunction: ActivationFunctionType = ActivationFunctionType.SIGMOIDAL,
    learningRate: number
  ) {
    this.layers = [];
    this.activationFunction = activationFunction;
    this.synapticProcessor = new SynapticProcessor(this.activationFunction, learningRate);
  }

  /**
   * @param {number} numberNeurons
   * @returns
   */
  add(numberNeurons: number) {
    if (!Number.isInteger(numberNeurons) || numberNeurons < 1) {
      throw new RangeError('A layer must contain at least one neuron');
    }

    const layer = this.create(numberNeurons);
    const indexNewLayer = this.layers.push(layer) - 1;
    const beforeLayer = this.layers[indexNewLayer - 1];

    if (beforeLayer === undefined) {
      return this;
    }

    // Apuntar con cada Neuron de la nueva capa a la anterior
    layer.forEach((neuron: Neuron) => {
      neuron.inputNeurons = beforeLayer;
    });

    // Apuntar con cada neurona de la capa anterior a la nueva capa
    beforeLayer.forEach((neuron: Neuron) => {
      neuron.outputNeurons = layer;
    });

    return this;
  }

  forEach(callback: (layer: Neuron[], index?: number) => void) {
    for (let idx = 0; idx < this.layers.length; idx++) {
      callback(this.layers[idx], idx);
    }
  }

  /**
   * @param {number} index
   * @returns {Neuron[]}
   */
  get(index: number): Neuron[] {
    return this.layers[index];
  }

  /**
   * @returns {Neuron[]}
   */
  getLast(): Neuron[] {
    return this.get(this.layers.length - 1);
  }

  get length(): number {
    return this.layers.length;
  }

  /**
   * @private
   * @param {number} numberNeurons
   * @returns {Neuron[]}
   */
  private create(numberNeurons: number): Neuron[] {
    const layer: Neuron[] = [];

    for (let i = 0; i < numberNeurons; i++) {
      layer[i] = new Neuron(this.activationFunction);
      layer[i].synapticProcessor = this.synapticProcessor;
    }

    return layer;
  }
}
