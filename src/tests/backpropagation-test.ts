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
