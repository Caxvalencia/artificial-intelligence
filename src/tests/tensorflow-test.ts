import * as tf from '@tensorflow/tfjs-node';
import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

// Or if running with GPU:
// import '@tensorflow/tfjs-node-gpu';

@suite
export class TensorflowTest {
  @test
  public async trainXOR() {
    const model: tf.Sequential = tf.sequential();

    const hiddenLayer: tf.layers.Layer = tf.layers.dense({
      units: 4,
      inputShape: [2],
      activation: 'sigmoid',
      kernelInitializer: tf.initializers.glorotUniform({ seed: 1 })
    });

    const outputLayer: tf.layers.Layer = tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
      kernelInitializer: tf.initializers.glorotUniform({ seed: 2 })
    });

    model.add(hiddenLayer);
    model.add(outputLayer);

    model.compile({
      loss: 'meanSquaredError',
      optimizer: tf.train.sgd(0.75)
    });

    const inputData = [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1]
    ];
    const input = tf.tensor2d(inputData);
    const output = tf.tensor1d([1, 0, 0, 1]);

    await model.fit(input, output, {
      verbose: 0,
      epochs: 1500
      // callbacks: {
      //     onEpochEnd: async (epoch, log) => {
      //         // console.log(`Epoch ${epoch}: loss = ${log.loss}`);
      //     }
      // }
    });

    const predictionsArray = (model.predict(input) as tf.Tensor).dataSync();
    const outputArray = output.dataSync();

    for (let index = 0; index < predictionsArray.length; index++) {
      const prediction = Math.round(predictionsArray[index]);

      assert.equal(
        outputArray[index],
        prediction,
        'input: ' + inputData[index] + ' output: ' + outputArray[index]
      );
    }
  }
}
