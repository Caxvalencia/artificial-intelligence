import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

import { ActivationFunctionType, Backpropagation, TrainingSample } from '../index';

@suite
export class BackpropagationTest {
  @test
  public validatesConfigurationAndLayers(): void {
    assert.throws(
      () => new Backpropagation({ epochs: -1 }),
      'epochs must be a non-negative integer'
    );
    assert.throws(
      () => new Backpropagation({ learningRate: 0 }),
      'learningRate must be a positive finite number'
    );
    assert.throws(
      () => new Backpropagation({ momentum: 1 }),
      'momentum must be between 0 inclusive and 1 exclusive'
    );
    assert.throws(() => new Backpropagation({ seed: 1.5 }), 'seed must be an integer');
    assert.throws(
      () => new Backpropagation({ targetLoss: -1 }),
      'targetLoss must be a non-negative finite number'
    );
    assert.throws(
      () => new Backpropagation({ patience: 0 }),
      'patience must be a positive integer'
    );
    assert.throws(
      () => new Backpropagation({ activationFunction: ActivationFunctionType.BINARY }),
      'Binary activation cannot be used with backpropagation'
    );
    assert.throws(
      () => new Backpropagation().addLayer(0),
      'A layer must contain at least one neuron'
    );
  }

  @test
  public validatesDatasetWithoutMutatingIt(): void {
    const network = new Backpropagation({ epochs: 1, seed: 1 }).addLayer(1);
    const input = [1, 0];
    const dataset = [{ input, output: 1 }];

    network.learn(dataset);

    assert.strictEqual(dataset[0].input, input);
    assert.isArray(dataset[0].input);
    assert.throws(() => network.learn([]), 'Dataset must contain at least one training sample');
    assert.throws(
      () =>
        network.learn([
          { input: [1, 0], output: 1 },
          { input: [1], output: 0 }
        ]),
      'Dataset sample 1 input dimension must be 2; received 1'
    );
    assert.throws(
      () => network.learn([{ input: [1], output: [0, 1] }]),
      'Dataset sample 0 output dimension must be 1; received 2'
    );
    assert.throws(
      () => network.learn([{ input: [Number.NaN], output: 1 }]),
      'Dataset sample 0 input must contain only finite numbers'
    );
  }

  @test
  public validatesTrainingAndProcessState(): void {
    assert.throws(
      () => new Backpropagation().learn([{ input: [1], output: 1 }]),
      'Backpropagation requires at least one layer before training'
    );
    assert.throws(
      () => new Backpropagation().process([1]),
      'Backpropagation requires at least one layer before processing data'
    );

    const network = new Backpropagation({ epochs: 1, seed: 1 }).addLayer(1);
    assert.throws(
      () => network.process([1]),
      'Backpropagation must be trained or imported before processing data'
    );

    network.learn([{ input: [1, 0], output: 1 }]);
    assert.throws(() => network.process([1]), 'Process data dimension must be 2; received 1');
  }

  @test
  public runsExactlyConfiguredEpochsAndSupportsEarlyStopping(): void {
    const dataset = [{ input: [1], output: 1 }];
    const zeroEpochNetwork = new Backpropagation({ epochs: 0, seed: 1 }).addLayer(1);

    zeroEpochNetwork.learn(dataset);

    assert.deepEqual(zeroEpochNetwork.history.loss, []);
    assert.equal(zeroEpochNetwork.history.epochs, 0);
    assert.isNull(zeroEpochNetwork.layers[0].neurons[0].weights);

    const network = new Backpropagation({ epochs: 10, targetLoss: 1, seed: 1 }).addLayer(1);
    network.learn(dataset);

    assert.lengthOf(network.history.loss, 1);
    assert.equal(network.history.epochs, 1);
    assert.isTrue(network.history.stoppedEarly);
    assert.equal(network.error, network.history.loss[0]);
  }

  @test
  public producesDeterministicModelsWithTheSameSeed(): void {
    const dataset = this.xnorDataset();
    const createNetwork = () =>
      new Backpropagation({ epochs: 20, seed: 42, shuffle: true, momentum: 0.5 })
        .addLayer(3)
        .addLayer(1)
        .learn(dataset)
        .exportModel();

    assert.deepEqual(createNetwork(), createNetwork());
  }

  @test
  public learnsTwoOutputs(): void {
    const dataset: TrainingSample[] = [
      { input: [0, 0], output: [0, 0] },
      { input: [0, 1], output: [0, 1] },
      { input: [1, 0], output: [1, 0] },
      { input: [1, 1], output: [1, 1] }
    ];
    const network = new Backpropagation({
      epochs: 5000,
      learningRate: 0.5,
      momentum: 0.2,
      seed: 7,
      shuffle: true,
      targetLoss: 0.001
    })
      .addLayer(3)
      .addLayer(2)
      .learn(dataset);

    dataset.forEach(({ input, output }) => {
      const prediction = network.process(input).map(Math.round);
      assert.deepEqual(prediction, Array.from(output as ArrayLike<number>));
    });
  }

  @test
  public learnsNonLinearProblemsWithScalarTargets(): void {
    const network = new Backpropagation({
      epochs: 10000,
      seed: 2,
      shuffle: true,
      targetLoss: 0.002
    })
      .addLayer(3)
      .addLayer(1)
      .learn(this.xnorDataset());

    this.xnorDataset().forEach(({ input, output }) => {
      assert.equal(Math.round(network.process(input)[0]), output);
    });
  }

  private xnorDataset(): Array<{ input: number[]; output: number }> {
    return [
      { input: [0, 0], output: 1 },
      { input: [0, 1], output: 0 },
      { input: [1, 0], output: 0 },
      { input: [1, 1], output: 1 }
    ];
  }
}
