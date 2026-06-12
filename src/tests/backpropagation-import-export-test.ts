import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

import { Backpropagation } from '../backpropagation';
import { Neuron } from '../neuron';
import { ActivationFunctionType } from '../activation-functions/activation-function';

function useSeededRandom(seed: number = 1): () => void {
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
export class BackpropagationImportExportTest {
  @test
  public validatesModelsBeforeImportingOrExporting() {
    const network = new Backpropagation();

    assert.throws(() => network.importModel({} as any), 'Model must contain at least one layer');
    assert.equal(network.layers.length, 0);
    assert.throws(() => network.exportModel(), 'Cannot export a network without layers');

    const untrainedNetwork = new Backpropagation().addLayer(1);
    assert.throws(() => untrainedNetwork.exportModel(), 'Cannot export an untrained network');
    assert.throws(
      () => untrainedNetwork.importModel(this.modelData()),
      'Cannot import a model into a network that already has layers'
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
  public importExportModel() {
    const model = this.modelData();
    const XOR = new Backpropagation();
    const modelExported = XOR.importModel(model).exportModel();

    assert.sameDeepMembers(model.layers, modelExported.layers, 'Layers');
    assert.sameDeepMembers(model.weights, modelExported.weights, 'Weights');
    assert.sameDeepMembers(model.thresholds, modelExported.thresholds, 'Thresholds');
  }

  @test('given one model data imported should relearning')
  public givenOneModelDataImportedShouldRelearning() {
    const model = this.modelData();

    const dataset = [
      { input: [0, 0], output: 1 },
      { input: [0, 1], output: 0 },
      { input: [1, 0], output: 0 },
      { input: [1, 1], output: 1 }
    ];

    const resultExpected = [
      { input: [0, 0], output: 0.9822190845688125 },
      { input: [0, 1], output: 0.01560094500947688 },
      { input: [1, 0], output: 0.015274557756339846 },
      { input: [1, 1], output: 0.9951608411056504 }
    ];

    const XOR = new Backpropagation();
    XOR.importModel(model).learn(dataset);

    assert.isBelow(XOR.error, 1e-7, 'error');

    resultExpected.forEach(({ input, output }) => {
      const outputActual = XOR.process(input)[0];

      assert.closeTo(outputActual, output, 1e-3, input + ' -> ' + output);
    });
  }

  // @test('given one new arquitecture should to learning one XOR logic')
  // public givenOneNewAcquitectureShouldToLearningOneXorLogic() {
  //     const dataset = [
  //         { input: [0, 0], output: 1 },
  //         { input: [0, 1], output: 0 },
  //         { input: [1, 0], output: 0 },
  //         { input: [1, 1], output: 1 }
  //     ];

  //     const neuronA = new Neuron(ActivationFunctionType.RELU);
  //     const neuronB = new Neuron(ActivationFunctionType.RELU);
  //     const neuronC = new Neuron(ActivationFunctionType.RELU);

  //     neuronA.outputNeurons.push(neuronC);
  //     neuronB.outputNeurons.push(neuronC);

  //     neuronC.outputNeurons.push(neuronA);
  //     neuronC.outputNeurons.push(neuronB);

  //     function forwardpropagation(data) {
  //         for (let index = 1; index <= 3; index++) {
  //             data[2] = neuronC.output();

  //             neuronA.learn(Float64Array.from(data));
  //             neuronB.learn(Float64Array.from(data));
  //             neuronC.learn(
  //                 Float64Array.from([neuronA.output(), neuronB.output()])
  //             );

  //             if (neuronC.synapse <= neuronC.threshold) {
  //                 break;
  //             }
  //         }

  //         return neuronC.output();
  //     }

  //     function backpropagation(output) {
  //         neuronC.error = output - neuronC.output();
  //         neuronA.error = neuronC.error * neuronC.weights[0];
  //         neuronB.error = neuronC.error * neuronC.weights[1];

  //         neuronA.recalculateWeights();
  //         neuronB.recalculateWeights();
  //         neuronC.recalculateWeights();
  //     }

  //     const epochs = 2500;

  //     for (let epoch = 1; epoch <= epochs; epoch++) {
  //         console.log('\nEpoch: ' + epoch);

  //         for (let index = 0; index < dataset.length; index++) {
  //             const data = dataset[index];
  //             forwardpropagation(data.input);
  //             backpropagation(data.output);

  //             console.log(neuronC.output(), data.output);
  //         }
  //     }
  // }

  @test('given one new arquitecture should to learning one NOT logic')
  public givenOneNewAcquitectureShouldToLearningOneNotLogic() {
    const restoreRandom = useSeededRandom();
    const dataset = [
      { input: [0, 0], output: 1 },
      { input: [0, 1], output: 0 },
      { input: [1, 0], output: 0 },
      { input: [1, 1], output: 1 }
    ];

    const neuronA = new Neuron(ActivationFunctionType.HYPERBOLIC_TANGENT);
    neuronA.outputNeurons.push(neuronA);

    // neuronA.output = function(): number {
    //     return this.synapse;
    // };

    function forwardpropagation(data) {
      data[2] = 0;

      for (let index = 1; index <= 100; index++) {
        neuronA.learn(Float64Array.from(data));

        if (data[2] === neuronA.output()) {
          break;
        }

        data[2] = neuronA.output();

        if (data[2] < neuronA.threshold) {
          break;
        }
      }

      return neuronA.output();
    }

    function backpropagation(output) {
      neuronA.error = output - neuronA.output();
      neuronA.recalculateWeights();
    }

    const epochs = 200;

    for (let epoch = 1; epoch <= epochs; epoch++) {
      console.log('\nEpoch: ' + epoch);
      console.log(neuronA.error);

      for (let index = 0; index < dataset.length; index++) {
        const data = dataset[index];
        forwardpropagation(data.input);
        backpropagation(data.output);
      }
    }
    restoreRandom();

    dataset.forEach((data) => {
      console.log(data, forwardpropagation(data.input));

      assert.equal(
        data.output,
        Math.round(forwardpropagation(data.input)),
        'input: ' + data.input + ' output: ' + data.output
      );
    });
  }

  private modelData() {
    return {
      layers: [3, 1],
      thresholds: [[-0.3658, -0.0281, 0.2527], [-0.2412]],
      weights: [
        [
          [7.76849, -4.66939],
          [-4.12036, 5.06214],
          [6.66429, 7.7002]
        ],
        [[-11.90173, -12.22545, 15.43677]]
      ]
    };
  }
}
