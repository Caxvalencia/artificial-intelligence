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
    let perceptron = new Perceptron();
    let data: any = [
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
    });
  }
}
