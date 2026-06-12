import { ActivationFunctionType } from './activation-functions/activation-function';
import { Layer } from './layer';
import { Neuron } from './neuron';

interface ModelType {
  layers: number[];
  thresholds: number[][];
  weights: number[][][];
}

interface BackpropagationConfig {
  epochs: number;
  activationFunction?: ActivationFunctionType;
  learningRate?: number;
  verbose?: boolean;
}

interface BackpropagationHistory {
  loss: number[];
}

interface TrainingSample {
  input: ArrayLike<number>;
  output: number;
}

interface NormalizedTrainingSample {
  input: Float64Array;
  output: number;
}

('use strict');
export class Backpropagation {
  layers: Layer;
  epochs: number;
  activationFunction: ActivationFunctionType;
  error: number;
  verbose: boolean;

  history: BackpropagationHistory;

  /**
   * @construtor
   */
  constructor(
    config: BackpropagationConfig = {
      epochs: 1000,
      activationFunction: ActivationFunctionType.SIGMOIDAL,
      learningRate: 0.3,
      verbose: false
    }
  ) {
    const epochs = config.epochs;
    const activationFunction = config.activationFunction ?? ActivationFunctionType.SIGMOIDAL;
    const learningRate = config.learningRate ?? 0.3;
    const verbose = config.verbose ?? false;

    if (!Number.isInteger(epochs) || epochs < 0) {
      throw new RangeError('epochs must be a non-negative integer');
    }

    if (!Object.values(ActivationFunctionType).includes(activationFunction)) {
      throw new RangeError(`Unknown activation function: ${activationFunction}`);
    }

    if (!Number.isFinite(learningRate) || learningRate <= 0) {
      throw new RangeError('learningRate must be a positive finite number');
    }

    this.error = 0;
    this.activationFunction = activationFunction;
    this.layers = new Layer(this.activationFunction, learningRate);
    this.epochs = epochs;
    this.verbose = verbose;
    this.history = { loss: [] };
  }

  learn(dataset: TrainingSample[]) {
    if (this.layers.length === 0) {
      throw new Error('Backpropagation requires at least one layer before training');
    }

    const normalizedDataset = this.normalizeDataset(dataset);
    this.history.loss = [];
    this.error = 0;

    for (let epoch = 1; epoch <= this.epochs; epoch++) {
      this.error = this.runEpoch(normalizedDataset);
      this.history.loss.push(this.error);

      if (this.verbose && epoch % 1000 === 0) {
        console.log(this.error, epoch);
      }
    }

    return this;
  }

  process(data: ArrayLike<number>) {
    if (this.layers.length === 0) {
      throw new Error('Backpropagation requires at least one layer before processing data');
    }

    const firstNeuron = this.layers.get(0)[0];

    if (!firstNeuron.weights) {
      throw new Error('Backpropagation must be trained or imported before processing data');
    }

    this.assertFiniteValues(data, 'Process data');

    if (data.length !== firstNeuron.weights.length) {
      throw new RangeError(
        `Process data dimension must be ${firstNeuron.weights.length}; received ${data.length}`
      );
    }

    let outputs: number[] = [];
    let currentData = new Float64Array(data);

    if (this.verbose) {
      console.log(currentData);
    }

    this.layers.forEach((layer: Neuron[]) => {
      if (outputs.length > 0) {
        currentData = new Float64Array(outputs);
        outputs = [];
      }

      this.layers.synapticProcessor.setData(currentData);

      for (let index = 0; index < layer.length; index++) {
        const neuron = layer[index];
        outputs[index] = neuron.process();
      }
    });

    if (this.verbose) {
      console.log(outputs);
    }

    return outputs;
  }

  addLayer(numberNeurons: number) {
    this.layers.add(numberNeurons);

    return this;
  }

  /**
   * @param {{ layers: number[]; weights: number[][] }} model
   * @returns {this}
   */
  importModel(model: ModelType): this {
    this.validateModel(model);

    if (this.layers.length > 0) {
      throw new Error('Cannot import a model into a network that already has layers');
    }

    model.layers.forEach((layer) => {
      this.addLayer(layer);
    });

    model.weights.forEach((layerWeights, index) => {
      this.layers.get(index).forEach((neuron: Neuron, neuronIndex) => {
        neuron
          .setWeights(new Float64Array(layerWeights[neuronIndex]))
          .setBeforeWeights(neuron.weights.slice())
          .setThreshold(model.thresholds[index][neuronIndex]);
      });
    });

    return this;
  }

  /**
   * @returns {ModelType}
   */
  exportModel(): ModelType {
    if (this.layers.length === 0) {
      throw new Error('Cannot export a network without layers');
    }

    let model: ModelType = {
      layers: [],
      thresholds: [],
      weights: []
    };

    this.layers.forEach((layer) => {
      model.layers.push(layer.length);

      const indexLayerThresholds = model.thresholds.push([]);
      const indexLayerWeights = model.weights.push([]);

      let layerThresholds = model.thresholds[indexLayerThresholds - 1];
      let layerWeights = model.weights[indexLayerWeights - 1];

      layer.forEach((neuron) => {
        if (!neuron.weights || !Number.isFinite(neuron.threshold)) {
          throw new Error('Cannot export an untrained network');
        }

        layerWeights.push(Array.from(neuron.weights));
        layerThresholds.push(neuron.threshold);
      });
    });

    return model;
  }

  /**
   * @private
   * @param {Array<{ input: Float64Array; output: number }>} dataset
   */
  private runEpoch(dataset: NormalizedTrainingSample[]) {
    let totalLoss = 0;

    for (let dataIdx = 0; dataIdx < dataset.length; dataIdx++) {
      const data = dataset[dataIdx];

      this.forwardpropagation(data);
      totalLoss += this.backpropagation(data.output);

      if (!Number.isFinite(totalLoss)) {
        throw new RangeError('Epoch loss produced a non-finite value');
      }

      this.layers.forEach((layer) => {
        for (let neuronIdx = 0; neuronIdx < layer.length; neuronIdx++) {
          const neuron = layer[neuronIdx];
          neuron.recalculateWeights();
        }
      });
    }

    const averageLoss = totalLoss / dataset.length;

    if (!Number.isFinite(averageLoss)) {
      throw new RangeError('Average epoch loss produced a non-finite value');
    }

    return averageLoss;
  }

  /**
   * @private
   * @param {*} { input, output }
   * @returns
   */
  private forwardpropagation({ input }: any) {
    let outputs = [];
    let data = new Float64Array(input);

    this.layers.forEach((layer: Neuron[]) => {
      if (outputs.length > 0) {
        data = new Float64Array(outputs);
        outputs = [];
      }

      this.layers.synapticProcessor.setData(data);

      for (let index = 0; index < layer.length; index++) {
        const neuron = layer[index];
        outputs[index] = neuron.learn().output();
      }
    });

    return this;
  }

  /**
   * @private
   * @param {number} output
   */
  private backpropagation(output: number): number {
    const lastLayer = this.layers.getLast();
    let sumErrors = 0;

    for (let index = 0; index < lastLayer.length; index++) {
      const neuron = lastLayer[index];

      neuron.calculateErrorOfOutput(output);
      neuron.backpropagation();

      sumErrors += neuron.error * neuron.error;
    }

    return sumErrors / 2;
  }

  private normalizeDataset(dataset: TrainingSample[]): NormalizedTrainingSample[] {
    if (!Array.isArray(dataset) || dataset.length === 0) {
      throw new RangeError('Dataset must contain at least one training sample');
    }

    let inputDimension: number;

    return dataset.map((sample, index) => {
      if (!sample || sample.input == null) {
        throw new TypeError(`Dataset sample ${index} must contain input and output`);
      }

      this.assertFiniteValues(sample.input, `Dataset sample ${index} input`);

      if (!Number.isFinite(sample.output)) {
        throw new RangeError(`Dataset sample ${index} output must be a finite number`);
      }

      if (inputDimension === undefined) {
        inputDimension = sample.input.length;
      } else if (sample.input.length !== inputDimension) {
        throw new RangeError(
          `Dataset sample ${index} input dimension must be ${inputDimension}; ` +
            `received ${sample.input.length}`
        );
      }

      return {
        input: new Float64Array(sample.input),
        output: sample.output
      };
    });
  }

  private validateModel(model: ModelType) {
    if (!model || !Array.isArray(model.layers) || model.layers.length === 0) {
      throw new TypeError('Model must contain at least one layer');
    }

    if (
      !Array.isArray(model.weights) ||
      !Array.isArray(model.thresholds) ||
      model.weights.length !== model.layers.length ||
      model.thresholds.length !== model.layers.length
    ) {
      throw new RangeError('Model layers, weights and thresholds must have matching lengths');
    }

    let expectedWeightCount: number;

    model.layers.forEach((numberNeurons, layerIndex) => {
      if (!Number.isInteger(numberNeurons) || numberNeurons < 1) {
        throw new RangeError(`Model layer ${layerIndex} must contain at least one neuron`);
      }

      const layerWeights = model.weights[layerIndex];
      const layerThresholds = model.thresholds[layerIndex];

      if (
        !Array.isArray(layerWeights) ||
        !Array.isArray(layerThresholds) ||
        layerWeights.length !== numberNeurons ||
        layerThresholds.length !== numberNeurons
      ) {
        throw new RangeError(`Model layer ${layerIndex} dimensions do not match its neuron count`);
      }

      layerThresholds.forEach((threshold) => {
        if (!Number.isFinite(threshold)) {
          throw new RangeError(`Model layer ${layerIndex} thresholds must be finite numbers`);
        }
      });

      layerWeights.forEach((weights, neuronIndex) => {
        this.assertFiniteValues(weights, `Model layer ${layerIndex} neuron ${neuronIndex} weights`);

        if (expectedWeightCount === undefined) {
          expectedWeightCount = weights.length;
        }

        if (weights.length !== expectedWeightCount) {
          throw new RangeError(
            `Model layer ${layerIndex} neuron ${neuronIndex} must have ` +
              `${expectedWeightCount} weights; received ${weights.length}`
          );
        }
      });

      expectedWeightCount = numberNeurons;
    });
  }

  private assertFiniteValues(values: ArrayLike<number>, label: string) {
    if (values == null || typeof values.length !== 'number' || values.length === 0) {
      throw new RangeError(`${label} must contain at least one value`);
    }

    for (let index = 0; index < values.length; index++) {
      if (!Number.isFinite(values[index])) {
        throw new RangeError(`${label} must contain only finite numbers`);
      }
    }
  }
}
