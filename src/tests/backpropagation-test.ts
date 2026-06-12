import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

import { ActivationFunctionType } from '../activation-functions/activation-function';
import { Backpropagation } from '../backpropagation';

function useSeededRandom(seed: number = 2): () => void {
  const originalRandom = Math.random;

  Math.random = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  return () => {
    Math.random = originalRandom;
  };
}

@suite
export class BackpropagationTest {
  @test
  public validatesConfigurationAndLayers() {
    assert.throws(
      () => new Backpropagation({ epochs: -1 }),
      'epochs must be a non-negative integer'
    );
    assert.throws(
      () => new Backpropagation({ epochs: 1, learningRate: 0 }),
      'learningRate must be a positive finite number'
    );
    assert.throws(
      () => new Backpropagation({ epochs: 1 }).addLayer(0),
      'A layer must contain at least one neuron'
    );
  }

  @test
  public validatesDatasetWithoutMutatingIt() {
    const network = new Backpropagation({ epochs: 1 }).addLayer(1);
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
      () => network.learn([{ input: [Number.NaN], output: 1 }]),
      'Dataset sample 0 input must contain only finite numbers'
    );
    assert.throws(
      () => network.learn([{ input: [1], output: Number.POSITIVE_INFINITY }]),
      'Dataset sample 0 output must be a finite number'
    );
  }

  @test
  public validatesTrainingAndProcessState() {
    assert.throws(
      () => new Backpropagation({ epochs: 1 }).learn([{ input: [1], output: 1 }]),
      'Backpropagation requires at least one layer before training'
    );
    assert.throws(
      () => new Backpropagation({ epochs: 1 }).process([1]),
      'Backpropagation requires at least one layer before processing data'
    );

    const network = new Backpropagation({ epochs: 1 }).addLayer(1);
    assert.throws(
      () => network.process([1]),
      'Backpropagation must be trained or imported before processing data'
    );

    network.learn([{ input: [1, 0], output: 1 }]);
    assert.throws(() => network.process([1]), 'Process data dimension must be 2; received 1');
  }

  @test
  public runsExactlyConfiguredEpochs() {
    const dataset = [{ input: [1], output: 1 }];
    const zeroEpochNetwork = new Backpropagation({ epochs: 0 }).addLayer(1);

    zeroEpochNetwork.learn(dataset);

    assert.deepEqual(zeroEpochNetwork.history.loss, []);
    assert.equal(zeroEpochNetwork.error, 0);
    assert.isNull(zeroEpochNetwork.layers.get(0)[0].weights);

    const network = new Backpropagation({ epochs: 3 }).addLayer(1);
    network.learn(dataset);

    assert.lengthOf(network.history.loss, 3);
    assert.equal(network.error, network.history.loss[2]);
  }

  @test
  public storesAverageLossForEveryEpoch() {
    const network = new Backpropagation({ epochs: 1 }).addLayer(1);
    const sampleLosses = [2, 4];
    const backpropagation = (network as any).backpropagation.bind(network);

    (network as any).backpropagation = () => sampleLosses.shift();
    network.learn([
      { input: [0], output: 0 },
      { input: [1], output: 1 }
    ]);
    (network as any).backpropagation = backpropagation;

    assert.deepEqual(network.history.loss, [3]);
    assert.equal(network.error, 3);
  }

  @test
  public rejectsBinaryActivation() {
    const network = new Backpropagation({
      epochs: 1,
      activationFunction: ActivationFunctionType.BINARY
    });

    assert.throws(
      () => network.addLayer(1).learn([{ input: [1], output: 1 }]),
      'Binary activation is not differentiable and cannot be used with backpropagation'
    );
  }

  @test
  public testOR() {
    const dataset = [
      { input: [0, 0], output: 0 },
      { input: [0, 1], output: 1 },
      { input: [1, 0], output: 1 },
      { input: [1, 1], output: 1 }
    ];

    const OR = new Backpropagation({
      epochs: 1000,
      learningRate: 10,
      verbose: true
    });
    OR.addLayer(2).addLayer(1).learn(dataset);

    dataset.forEach(({ input, output }) => {
      const outputActual = Math.round(OR.process(input)[0]);

      assert.equal(outputActual, output, input + ' -> ' + output);
    });
  }

  @test
  public testAND() {
    const dataset = [
      { input: [0, 0], output: 0 },
      { input: [0, 1], output: 0 },
      { input: [1, 0], output: 0 },
      { input: [1, 1], output: 1 }
    ];

    const AND = new Backpropagation({ epochs: 1000 });
    AND.addLayer(2).addLayer(1).learn(dataset);

    dataset.forEach(({ input, output }) => {
      const outputActual = Math.round(AND.process(input)[0]);

      assert.equal(outputActual, output, input + ' -> ' + output);
    });
  }

  @test
  public testXOR() {
    const dataset = [
      { input: [0, 0], output: 1 },
      { input: [0, 1], output: 0 },
      { input: [1, 0], output: 0 },
      { input: [1, 1], output: 1 }
    ];

    const XOR = new Backpropagation({ epochs: 15000 });
    XOR.addLayer(3).addLayer(1).learn(dataset);

    dataset.forEach(({ input, output }) => {
      const outputActual = Math.round(XOR.process(input)[0]);

      assert.equal(outputActual, output, input + ' -> ' + output);
    });
  }

  @test
  public testHyperbolicTangentForXOR() {
    const restoreRandom = useSeededRandom();
    const dataset = [
      { input: [-1, -1], output: -1 },
      { input: [-1, 1], output: 1 },
      { input: [1, -1], output: 1 },
      { input: [1, 1], output: -1 }
    ];

    const XOR = new Backpropagation({
      epochs: 10000,
      activationFunction: ActivationFunctionType.HYPERBOLIC_TANGENT
    });
    XOR.addLayer(3).addLayer(1).learn(dataset);
    restoreRandom();

    dataset.forEach(({ input, output }) => {
      const outputActual = Math.round(XOR.process(input)[0]);

      assert.equal(outputActual, output, input + ' -> ' + output);
    });
  }
}
