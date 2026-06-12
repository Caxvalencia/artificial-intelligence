import { ActivationFunctionType } from './activation-functions/activation-function';
import { Layer } from './layer';
import {
  BackpropagationConfig,
  BackpropagationHistory,
  LegacySerializedModel,
  SerializedModel,
  TrainingSample
} from './types';

interface NormalizedTrainingSample {
  input: Float64Array;
  output: Float64Array;
}

export class Backpropagation {
  layers: Layer[];
  epochs: number;
  activationFunction: ActivationFunctionType;
  learningRate: number;
  momentum: number;
  error: number;
  verbose: boolean;
  shuffle: boolean;
  targetLoss?: number;
  patience?: number;
  history: BackpropagationHistory;

  private random: () => number;

  constructor(config: BackpropagationConfig = {}) {
    this.epochs = config.epochs ?? 1000;
    this.activationFunction = config.activationFunction ?? ActivationFunctionType.SIGMOIDAL;
    this.learningRate = config.learningRate ?? 0.3;
    this.momentum = config.momentum ?? 0.77;
    this.verbose = config.verbose ?? false;
    this.shuffle = config.shuffle ?? false;
    this.targetLoss = config.targetLoss;
    this.patience = config.patience;
    this.random = this.createRandom(config.seed);
    this.layers = [];
    this.error = 0;
    this.history = { loss: [], epochs: 0, stoppedEarly: false };

    this.validateConfig(config.seed);
  }

  addLayer(numberNeurons: number): this {
    this.layers.push(
      new Layer(
        numberNeurons,
        this.activationFunction,
        this.learningRate,
        this.momentum,
        this.random
      )
    );

    return this;
  }

  learn(dataset: TrainingSample[]): this {
    if (this.layers.length === 0) {
      throw new Error('Backpropagation requires at least one layer before training');
    }

    const normalizedDataset = this.normalizeDataset(dataset);

    this.history = { loss: [], epochs: 0, stoppedEarly: false };
    this.error = 0;
    let bestLoss = Number.POSITIVE_INFINITY;
    let staleEpochs = 0;

    for (let epoch = 1; epoch <= this.epochs; epoch++) {
      this.error = this.runEpoch(normalizedDataset);
      this.history.loss.push(this.error);
      this.history.epochs = epoch;

      if (this.verbose && epoch % 1000 === 0) {
        console.log(this.error, epoch);
      }

      if (this.error < bestLoss - Number.EPSILON) {
        bestLoss = this.error;
        staleEpochs = 0;
      } else {
        staleEpochs++;
      }

      const reachedTarget = this.targetLoss !== undefined && this.error <= this.targetLoss;
      const exhaustedPatience = this.patience !== undefined && staleEpochs >= this.patience;

      if (reachedTarget || exhaustedPatience) {
        this.history.stoppedEarly = true;

        break;
      }
    }

    return this;
  }

  process(data: ArrayLike<number>): number[] {
    if (this.layers.length === 0) {
      throw new Error('Backpropagation requires at least one layer before processing data');
    }

    const firstNeuron = this.layers[0].neurons[0];

    if (!firstNeuron.weights) {
      throw new Error('Backpropagation must be trained or imported before processing data');
    }

    this.assertFiniteValues(data, 'Process data');

    if (data.length !== firstNeuron.weights.length) {
      throw new RangeError(
        `Process data dimension must be ${firstNeuron.weights.length}; received ${data.length}`
      );
    }

    return Array.from(this.forward(new Float64Array(data)));
  }

  importModel(model: SerializedModel | LegacySerializedModel): this {
    const normalizedModel = this.normalizeModel(model);

    if (this.layers.length > 0) {
      throw new Error('Cannot import a model into a network that already has layers');
    }

    this.activationFunction = normalizedModel.config.activationFunction;
    this.learningRate = normalizedModel.config.learningRate;
    this.momentum = normalizedModel.config.momentum;

    normalizedModel.layers.forEach((size) => this.addLayer(size));
    this.layers.forEach((layer, index) => {
      layer.setParameters(normalizedModel.weights[index], normalizedModel.biases[index]);
    });

    return this;
  }

  exportModel(): SerializedModel {
    if (this.layers.length === 0) {
      throw new Error('Cannot export a network without layers');
    }

    return {
      version: 1,
      config: {
        activationFunction: this.activationFunction,
        learningRate: this.learningRate,
        momentum: this.momentum
      },
      layers: this.layers.map((layer) => layer.size),
      biases: this.layers.map((layer) => layer.exportBiases()),
      weights: this.layers.map((layer) => layer.exportWeights())
    };
  }

  private runEpoch(dataset: NormalizedTrainingSample[]): number {
    let totalLoss = 0;
    const samples = this.shuffle ? this.shuffleSamples(dataset) : dataset;

    for (const sample of samples) {
      this.forward(sample.input);
      const outputLayer = this.layers[this.layers.length - 1];
      totalLoss += outputLayer.calculateOutputDeltas(sample.output);

      for (let index = this.layers.length - 2; index >= 0; index--) {
        this.layers[index].calculateHiddenDeltas(this.layers[index + 1]);
      }

      this.layers.forEach((layer) => layer.update());
      this.assertFinite(totalLoss, 'Epoch loss');
    }

    const averageLoss = totalLoss / samples.length;
    this.assertFinite(averageLoss, 'Average epoch loss');

    return averageLoss;
  }

  private forward(input: Float64Array): Float64Array {
    let output = input;

    for (const layer of this.layers) {
      output = layer.forward(output);
    }

    return output;
  }

  private normalizeDataset(dataset: TrainingSample[]): NormalizedTrainingSample[] {
    if (!Array.isArray(dataset) || dataset.length === 0) {
      throw new RangeError('Dataset must contain at least one training sample');
    }

    const outputDimension = this.layers[this.layers.length - 1].size;
    let inputDimension: number | undefined;

    return dataset.map((sample, index) => {
      if (!sample || sample.input == null || sample.output == null) {
        throw new TypeError(`Dataset sample ${index} must contain input and output`);
      }

      this.assertFiniteValues(sample.input, `Dataset sample ${index} input`);

      if (inputDimension === undefined) {
        inputDimension = sample.input.length;
      } else if (sample.input.length !== inputDimension) {
        throw new RangeError(
          `Dataset sample ${index} input dimension must be ${inputDimension}; ` +
            `received ${sample.input.length}`
        );
      }

      const output =
        typeof sample.output === 'number' ? new Float64Array([sample.output]) : sample.output;
      this.assertFiniteValues(output, `Dataset sample ${index} output`);

      if (output.length !== outputDimension) {
        throw new RangeError(
          `Dataset sample ${index} output dimension must be ${outputDimension}; ` +
            `received ${output.length}`
        );
      }

      return { input: new Float64Array(sample.input), output: new Float64Array(output) };
    });
  }

  private normalizeModel(model: SerializedModel | LegacySerializedModel): SerializedModel {
    if (!model || !Array.isArray(model.layers) || model.layers.length === 0) {
      throw new TypeError('Model must contain at least one layer');
    }

    const versioned = 'version' in model;

    if (versioned && model.version !== 1) {
      throw new RangeError(`Unsupported model version: ${model.version}`);
    }

    const activationFunction = versioned
      ? model.config?.activationFunction
      : this.activationFunction;
    const learningRate = versioned ? model.config?.learningRate : this.learningRate;
    const momentum = versioned ? model.config?.momentum : this.momentum;
    const biases = versioned ? model.biases : model.thresholds;

    this.validateModelParts(model.layers, model.weights, biases);

    if (!Object.values(ActivationFunctionType).includes(activationFunction)) {
      throw new RangeError(`Unknown activation function: ${activationFunction}`);
    }

    this.assertPositiveFinite(learningRate, 'Model learningRate');
    this.assertRange(momentum, 'Model momentum', 0, 1);

    return {
      version: 1,
      config: { activationFunction, learningRate, momentum },
      layers: model.layers.map((size) => size),
      biases: biases.map((layer) => layer.slice()),
      weights: model.weights.map((layer) => layer.map((weights) => weights.slice()))
    };
  }

  private validateModelParts(layers: number[], weights: number[][][], biases: number[][]): void {
    if (
      !Array.isArray(weights) ||
      !Array.isArray(biases) ||
      weights.length !== layers.length ||
      biases.length !== layers.length
    ) {
      throw new RangeError('Model layers, weights and biases must have matching lengths');
    }

    let expectedWeightCount: number | undefined;

    layers.forEach((size, layerIndex) => {
      if (!Number.isInteger(size) || size < 1) {
        throw new RangeError(`Model layer ${layerIndex} must contain at least one neuron`);
      }

      if (
        !Array.isArray(weights[layerIndex]) ||
        !Array.isArray(biases[layerIndex]) ||
        weights[layerIndex].length !== size ||
        biases[layerIndex].length !== size
      ) {
        throw new RangeError(`Model layer ${layerIndex} dimensions do not match its neuron count`);
      }

      this.assertFiniteValues(biases[layerIndex], `Model layer ${layerIndex} biases`);

      weights[layerIndex].forEach((neuronWeights, neuronIndex) => {
        this.assertFiniteValues(
          neuronWeights,
          `Model layer ${layerIndex} neuron ${neuronIndex} weights`
        );

        if (expectedWeightCount === undefined) {
          expectedWeightCount = neuronWeights.length;
        }

        if (neuronWeights.length !== expectedWeightCount) {
          throw new RangeError(
            `Model layer ${layerIndex} neuron ${neuronIndex} must have ` +
              `${expectedWeightCount} weights; received ${neuronWeights.length}`
          );
        }
      });

      expectedWeightCount = size;
    });
  }

  private validateConfig(seed?: number): void {
    if (!Number.isInteger(this.epochs) || this.epochs < 0) {
      throw new RangeError('epochs must be a non-negative integer');
    }

    if (!Object.values(ActivationFunctionType).includes(this.activationFunction)) {
      throw new RangeError(`Unknown activation function: ${this.activationFunction}`);
    }

    if (this.activationFunction === ActivationFunctionType.BINARY) {
      throw new RangeError('Binary activation cannot be used with backpropagation');
    }

    this.assertPositiveFinite(this.learningRate, 'learningRate');
    this.assertRange(this.momentum, 'momentum', 0, 1);

    if (seed !== undefined && !Number.isInteger(seed)) {
      throw new RangeError('seed must be an integer');
    }

    if (
      this.targetLoss !== undefined &&
      (!Number.isFinite(this.targetLoss) || this.targetLoss < 0)
    ) {
      throw new RangeError('targetLoss must be a non-negative finite number');
    }

    if (this.patience !== undefined && (!Number.isInteger(this.patience) || this.patience < 1)) {
      throw new RangeError('patience must be a positive integer');
    }
  }

  private shuffleSamples(dataset: NormalizedTrainingSample[]): NormalizedTrainingSample[] {
    const shuffled = dataset.slice();

    for (let index = shuffled.length - 1; index > 0; index--) {
      const randomIndex = Math.floor(this.random() * (index + 1));
      [shuffled[index], shuffled[randomIndex]] = [shuffled[randomIndex], shuffled[index]];
    }

    return shuffled;
  }

  private createRandom(seed?: number): () => number {
    if (seed === undefined) {
      return Math.random;
    }

    let state = seed >>> 0;

    return () => {
      state = (state * 1664525 + 1013904223) >>> 0;

      return state / 4294967296;
    };
  }

  private assertPositiveFinite(value: number, label: string): void {
    if (!Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${label} must be a positive finite number`);
    }
  }

  private assertRange(value: number, label: string, minimum: number, maximum: number): void {
    if (!Number.isFinite(value) || value < minimum || value >= maximum) {
      throw new RangeError(
        `${label} must be between ${minimum} inclusive and ${maximum} exclusive`
      );
    }
  }

  private assertFiniteValues(values: ArrayLike<number>, label: string): void {
    if (values == null || typeof values.length !== 'number' || values.length === 0) {
      throw new RangeError(`${label} must contain at least one value`);
    }

    for (let index = 0; index < values.length; index++) {
      this.assertFinite(values[index], label);
    }
  }

  private assertFinite(value: number, label: string): void {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${label} must contain only finite numbers`);
    }
  }
}
