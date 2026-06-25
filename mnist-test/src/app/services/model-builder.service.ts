import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

// Custom Self-Attention Layer
export class SelfAttentionLayer extends tf.layers.Layer {
  private units: number;
  private wQ: tf.LayerVariable | null = null;
  private wK: tf.LayerVariable | null = null;
  private wV: tf.LayerVariable | null = null;

  constructor(config: { units: number; [key: string]: any }) {
    super(config as any);
    this.units = config.units;
  }

  static get className() {
    return 'SelfAttentionLayer';
  }

  override build(inputShape: tf.Shape | tf.Shape[]) {
    const shape = Array.isArray(inputShape[0])
      ? (inputShape[0] as number[])
      : (inputShape as number[]);
    const lastDim = shape[shape.length - 1];

    this.wQ = this.addWeight(
      'wQ',
      [lastDim, this.units],
      'float32',
      tf.initializers.glorotUniform({}),
    );
    this.wK = this.addWeight(
      'wK',
      [lastDim, this.units],
      'float32',
      tf.initializers.glorotUniform({}),
    );
    this.wV = this.addWeight(
      'wV',
      [lastDim, this.units],
      'float32',
      tf.initializers.glorotUniform({}),
    );

    super.build(inputShape);
  }

  override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    const shape = Array.isArray(inputShape[0])
      ? (inputShape[0] as number[])
      : (inputShape as number[]);
    return [shape[0], shape[1], this.units];
  }

  override call(inputs: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor | tf.Tensor[] {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const q = tf.dot(input, this.wQ!.read());
      const k = tf.dot(input, this.wK!.read());
      const v = tf.dot(input, this.wV!.read());

      // Scaled Dot-Product Attention: Softmax( (Q * K^T) / sqrt(d) ) * V
      const scores = tf.matMul(q, k, false, true);
      const scaledScores = tf.div(scores, tf.sqrt(tf.scalar(this.units)));
      const attentionWeights = tf.softmax(scaledScores, -1);

      return tf.matMul(attentionWeights, v);
    });
  }

  override getConfig() {
    const config = super.getConfig();
    Object.assign(config, { units: this.units });
    return config;
  }
}

tf.serialization.registerClass(SelfAttentionLayer);

// Hopfield Network class
export class HopfieldNetwork {
  private size = 784; // 28x28
  private weights: tf.Tensor2D;

  constructor() {
    this.weights = tf.zeros([this.size, this.size]);
  }

  // Train with Hopfield Hebbian learning rule
  train(patterns: Float32Array[]) {
    tf.tidy(() => {
      let wSum = tf.zeros([this.size, this.size]);
      const count = patterns.length;
      if (count === 0) return;

      for (const pattern of patterns) {
        // Convert to bipolar values (-1, +1)
        const pat = tf.tensor1d(pattern).mul(2).sub(1);
        const pat2D = pat.expandDims(1);
        const outer = tf.matMul(pat2D, pat2D, false, true);
        wSum = wSum.add(outer);
      }

      // Average weights, zero out diagonal to prevent self-feedback
      const scale = tf.scalar(1 / count);
      let newWeights = wSum.mul(scale);
      const mask = tf.scalar(1).sub(tf.eye(this.size));
      newWeights = newWeights.mul(mask);

      this.weights.dispose();
      this.weights = tf.keep(newWeights) as tf.Tensor2D;
    });
  }

  // Synchronous state update reconstruction step
  reconstructStep(state: Float32Array): Float32Array {
    return tf.tidy(() => {
      // Bipolar state (-1, +1)
      const st = tf.tensor1d(state).mul(2).sub(1);
      const wState = tf.matMul(this.weights, st.expandDims(1)).squeeze();

      // Sign activation function: if >= 0 then 1 else -1
      const activated = tf.where(
        wState.greaterEqual(0),
        tf.onesLike(wState),
        tf.onesLike(wState).mul(-1),
      );

      // Map back to [0, 1]
      const finalState = activated.add(1).div(2);
      return finalState.dataSync() as Float32Array;
    });
  }
}

export interface LayerDescription {
  id: string;
  type: 'dense' | 'conv2d' | 'maxPool2d' | 'flatten' | 'reshape' | 'attention' | 'dropout';
  config: any;
  x?: number;
  y?: number;
}

@Injectable({
  providedIn: 'root',
})
export class ModelBuilderService {
  buildModel(layersConfig: LayerDescription[]): tf.Sequential {
    const model = tf.sequential();

    layersConfig.forEach((layerConf, index) => {
      const isFirst = index === 0;
      const conf = { ...layerConf.config };

      switch (layerConf.type) {
        case 'dense':
          if (isFirst) conf.inputShape = [784];
          model.add(tf.layers.dense(conf));
          break;
        case 'conv2d':
          if (isFirst) conf.inputShape = [28, 28, 1];
          model.add(tf.layers.conv2d(conf));
          break;
        case 'maxPool2d':
          model.add(tf.layers.maxPooling2d(conf));
          break;
        case 'flatten':
          model.add(tf.layers.flatten());
          break;
        case 'reshape':
          if (isFirst) conf.inputShape = [784];
          model.add(tf.layers.reshape(conf));
          break;
        case 'attention':
          if (isFirst) conf.inputShape = [28, 28]; // sequence size = 28, features = 28
          model.add(new SelfAttentionLayer(conf));
          break;
        case 'dropout':
          model.add(tf.layers.dropout(conf));
          break;
      }
    });

    return model;
  }

  getDefaultCNNConfig(): LayerDescription[] {
    return [
      { id: '1', type: 'reshape', config: { targetShape: [28, 28, 1] } },
      {
        id: '2',
        type: 'conv2d',
        config: {
          filters: 8,
          kernelSize: 3,
          strides: 1,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
        },
      },
      { id: '3', type: 'maxPool2d', config: { poolSize: [2, 2], strides: [2, 2] } },
      {
        id: '4',
        type: 'conv2d',
        config: {
          filters: 16,
          kernelSize: 3,
          strides: 1,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
        },
      },
      { id: '5', type: 'maxPool2d', config: { poolSize: [2, 2], strides: [2, 2] } },
      { id: '6', type: 'flatten', config: {} },
      {
        id: '7',
        type: 'dense',
        config: { units: 64, activation: 'relu', kernelInitializer: 'varianceScaling' },
      },
      {
        id: '8',
        type: 'dense',
        config: { units: 10, activation: 'softmax', kernelInitializer: 'varianceScaling' },
      },
    ];
  }

  getDefaultDenseConfig(): LayerDescription[] {
    return [
      {
        id: '1',
        type: 'dense',
        config: { units: 128, activation: 'relu', kernelInitializer: 'varianceScaling' },
      },
      {
        id: '2',
        type: 'dense',
        config: { units: 64, activation: 'relu', kernelInitializer: 'varianceScaling' },
      },
      {
        id: '3',
        type: 'dense',
        config: { units: 10, activation: 'softmax', kernelInitializer: 'varianceScaling' },
      },
    ];
  }

  getDefaultTransformerConfig(): LayerDescription[] {
    return [
      // Treat MNIST 28x28 as sequence of 28 steps with 28 features
      { id: '1', type: 'reshape', config: { targetShape: [28, 28] } },
      { id: '2', type: 'attention', config: { units: 32 } },
      { id: '3', type: 'flatten', config: {} },
      {
        id: '4',
        type: 'dense',
        config: { units: 64, activation: 'relu', kernelInitializer: 'varianceScaling' },
      },
      {
        id: '5',
        type: 'dense',
        config: { units: 10, activation: 'softmax', kernelInitializer: 'varianceScaling' },
      },
    ];
  }
}
