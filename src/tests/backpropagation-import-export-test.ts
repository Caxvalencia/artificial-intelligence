import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

import { ActivationFunctionType, Backpropagation, SerializedModel, TrainingSample } from '../index';

@suite
export class BackpropagationImportExportTest {
  @test
  public validatesModelsBeforeImportingOrExporting(): void {
    const network = new Backpropagation();

    assert.throws(
      () => network.importModel({} as SerializedModel),
      'Model must contain at least one layer'
    );
    assert.equal(network.layers.length, 0);
    assert.throws(() => network.exportModel(), 'Cannot export a network without layers');
    assert.throws(
      () => new Backpropagation().addLayer(1).exportModel(),
      'Cannot export an untrained network'
    );
    assert.throws(
      () =>
        network.importModel({
          version: 2,
          config: {
            activationFunction: ActivationFunctionType.SIGMOIDAL,
            learningRate: 0.3,
            momentum: 0.77
          },
          layers: [1],
          biases: [[0]],
          weights: [[[1]]]
        } as unknown as SerializedModel),
      'Unsupported model version: 2'
    );
    assert.throws(
      () =>
        network.importModel({
          layers: [1],
          thresholds: [[0]],
          weights: [[[Number.NaN]]]
        }),
      'Model layer 0 neuron 0 weights must contain only finite numbers'
    );
    assert.equal(network.layers.length, 0);
  }

  @test
  public importsLegacyModelsAndExportsVersionedModels(): void {
    const model = new Backpropagation().importModel({
      layers: [1],
      thresholds: [[0.5]],
      weights: [[[1, -1]]]
    });
    const exported = model.exportModel();

    assert.equal(exported.version, 1);
    assert.deepEqual(exported.layers, [1]);
    assert.deepEqual(exported.biases, [[0.5]]);
    assert.deepEqual(exported.weights, [[[1, -1]]]);
  }

  @test
  public preservesSigmoidalAndTanhPredictionsAfterRoundTrip(): void {
    this.assertRoundTrip(ActivationFunctionType.SIGMOIDAL, this.sigmoidDataset());
    this.assertRoundTrip(ActivationFunctionType.HYPERBOLIC_TANGENT, this.tanhDataset());
  }

  @test
  public persistsTrainingConfiguration(): void {
    const network = new Backpropagation({
      epochs: 1,
      activationFunction: ActivationFunctionType.HYPERBOLIC_TANGENT,
      learningRate: 0.15,
      momentum: 0.4,
      seed: 9
    })
      .addLayer(2)
      .addLayer(1)
      .learn(this.tanhDataset());
    const restored = new Backpropagation().importModel(network.exportModel());

    assert.equal(restored.activationFunction, ActivationFunctionType.HYPERBOLIC_TANGENT);
    assert.equal(restored.learningRate, 0.15);
    assert.equal(restored.momentum, 0.4);
  }

  private assertRoundTrip(
    activationFunction: ActivationFunctionType,
    dataset: TrainingSample[]
  ): void {
    const network = new Backpropagation({
      epochs: 100,
      activationFunction,
      seed: 4,
      shuffle: true
    })
      .addLayer(3)
      .addLayer(1)
      .learn(dataset);
    const predictions = dataset.map(({ input }) => network.process(input));
    const restored = new Backpropagation().importModel(
      JSON.parse(JSON.stringify(network.exportModel())) as SerializedModel
    );

    assert.deepEqual(
      dataset.map(({ input }) => restored.process(input)),
      predictions
    );
  }

  private sigmoidDataset(): TrainingSample[] {
    return [
      { input: [0, 0], output: 1 },
      { input: [0, 1], output: 0 },
      { input: [1, 0], output: 0 },
      { input: [1, 1], output: 1 }
    ];
  }

  private tanhDataset(): TrainingSample[] {
    return [
      { input: [-1, -1], output: -1 },
      { input: [-1, 1], output: 1 },
      { input: [1, -1], output: 1 },
      { input: [1, 1], output: -1 }
    ];
  }
}
