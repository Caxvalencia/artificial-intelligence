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
