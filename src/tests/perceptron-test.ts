import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

import { Perceptron } from '../perceptron';

@suite
export class PerceptronTest {
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
