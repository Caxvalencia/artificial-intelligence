import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

import { Perceptron } from '../perceptron';

@suite
export class PerceptronTest {
  @test
  public validatesTrainingData() {
    const perceptron = new Perceptron();

    assert.throws(() => perceptron.learn(), 'Perceptron requires at least one training sample');
    assert.throws(
      () => perceptron.addData([1, Number.NaN], 1),
      'Training data must contain only finite numbers'
    );
    assert.throws(() => perceptron.addData([1], 2), 'Perceptron output must be either 0 or 1');

    perceptron.addData([1, 0], 1);
    assert.throws(
      () => perceptron.addData([1], 1),
      'Training data dimension must be 2; received 1'
    );
  }

  @test
  public validatesProcessStateAndDimensions() {
    const perceptron = new Perceptron();

    assert.throws(
      () => perceptron.process([1]),
      'Perceptron must be trained or configured before processing data'
    );

    perceptron.setWeights([1, 1]);
    perceptron.threshold = 0;

    assert.throws(() => perceptron.process([1]), 'Process data dimension must be 2; received 1');
    assert.throws(
      () => perceptron.process([1, Number.POSITIVE_INFINITY]),
      'Process data must contain only finite numbers'
    );
  }

  @test
  public copiesTrainingData() {
    const input = [1, 0];
    const perceptron = new Perceptron().addData(input, 1);

    input[0] = 0;

    assert.deepEqual(Array.from(perceptron.dataStack[0][0]), [1, 0]);
  }

  @test
  public testAND() {
    let perceptron = new Perceptron();
    let data: any = [
      // data and output
      [[0, 0], 0],
      [[0, 1], 0],
      [[1, 0], 0],
      [[1, 1], 1]
    ];

    data.forEach((data) => {
      perceptron.addData(data[0], data[1]).learn();
    });

    data.forEach((data) => {
      assert.equal(data[1], perceptron.process(data[0]), data[0] + ' -> ' + data[1]);
    });
  }

  @test
  public testOR() {
    let perceptron = new Perceptron();
    let data: any = [
      // data and output
      [[0, 0], 0],
      [[0, 1], 1],
      [[1, 0], 1],
      [[1, 1], 1]
    ];

    data.forEach((data) => {
      perceptron.addData(data[0], data[1]).learn();
    });

    data.forEach((data) => {
      assert.equal(data[1], perceptron.process(data[0]), data[0] + ' -> ' + data[1]);
    });
  }

  @test
  public testFailXOR() {
    const perceptron = new Perceptron().setMaxEpochs(10);
    const data: any = [
      // data and output
      [[0, 0], 1],
      [[0, 1], 0],
      [[1, 0], 0],
      [[1, 1], 1]
    ];

    assert.throws(() => {
      data.forEach((_data) => {
        perceptron.addData(_data[0], _data[1]).learn();
      });
    }, 'Perceptron did not converge after 10 epochs');
    assert.equal(perceptron.trainingStats.epochs, 10);
    assert.isAbove(perceptron.trainingStats.errors, 0);
    assert.isFalse(perceptron.trainingStats.converged);
  }

  @test
  public exposesConvergedTrainingStats() {
    const perceptron = new Perceptron();

    perceptron
      .addData([0, 0] as any, 0)
      .addData([0, 1] as any, 1)
      .addData([1, 0] as any, 1)
      .addData([1, 1] as any, 1)
      .learn();

    assert.isAbove(perceptron.trainingStats.epochs, 0);
    assert.equal(perceptron.trainingStats.errors, 0);
    assert.isTrue(perceptron.trainingStats.converged);
  }

  @test
  public rejectsInvalidMaxEpochs() {
    const perceptron = new Perceptron();

    assert.throws(() => perceptron.setMaxEpochs(0), 'maxEpochs must be a positive integer');
  }
}
