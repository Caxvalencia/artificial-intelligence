import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const TRAIN_TEST_RATIO = 5 / 6;
const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

@Injectable({
  providedIn: 'root',
})
export class MnistDataService {
  private datasetImages: Float32Array | null = null;
  private datasetLabels: Uint8Array | null = null;
  private trainIndices: Uint32Array | null = null;
  private testIndices: Uint32Array | null = null;
  private trainImages: Float32Array | null = null;
  private testImages: Float32Array | null = null;
  private trainLabels: Uint8Array | null = null;
  private testLabels: Uint8Array | null = null;

  private shuffledTrainIndex = 0;
  private shuffledTestIndex = 0;

  public isLoaded = false;
  public loadProgress = 0;

  async load(): Promise<void> {
    if (this.isLoaded) return;

    // Load images
    const img = new Image();
    img.crossOrigin = 'anonymous';

    const imgRequest = new Promise<void>((resolve, reject) => {
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        const chunkSize = 5000;
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = chunkSize;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          reject(new Error('Could not create canvas context'));
          return;
        }

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * chunkSize * IMAGE_SIZE * 4,
            chunkSize * IMAGE_SIZE,
          );

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // Read only red channel since it is grayscale, normalize to [0, 1]
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }

          this.loadProgress = Math.round((((i + 1) * chunkSize) / NUM_DATASET_ELEMENTS) * 50);
        }

        this.datasetImages = new Float32Array(datasetBytesBuffer);
        resolve();
      };
      img.onerror = (err) => reject(err);
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    // Load labels
    const labelsRequest = fetch(MNIST_LABELS_PATH).then(async (res) => {
      const buffer = await res.arrayBuffer();
      this.datasetLabels = new Uint8Array(buffer);
      this.loadProgress = 75;
    });

    await Promise.all([imgRequest, labelsRequest]);

    if (!this.datasetImages || !this.datasetLabels) {
      throw new Error('Failed to load dataset');
    }

    // Create shuffled indices
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice images and labels into train/test
    this.trainImages = this.datasetImages.subarray(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.subarray(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels = this.datasetLabels.subarray(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.subarray(NUM_CLASSES * NUM_TRAIN_ELEMENTS);

    this.isLoaded = true;
    this.loadProgress = 100;
  }

  getTrainData(batchSize?: number): { xs: tf.Tensor; ys: tf.Tensor } {
    if (!this.trainImages || !this.trainLabels || !this.trainIndices) {
      throw new Error('Data not loaded');
    }
    const size = batchSize || NUM_TRAIN_ELEMENTS;
    return this.nextBatch(size, [this.trainImages, this.trainLabels], () => {
      const index = this.trainIndices![this.shuffledTrainIndex];
      this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices!.length;
      return index;
    });
  }

  getTestData(batchSize?: number): { xs: tf.Tensor; ys: tf.Tensor } {
    if (!this.testImages || !this.testLabels || !this.testIndices) {
      throw new Error('Data not loaded');
    }
    const size = batchSize || NUM_TEST_ELEMENTS;
    return this.nextBatch(size, [this.testImages, this.testLabels], () => {
      const index = this.testIndices![this.shuffledTestIndex];
      this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices!.length;
      return index;
    });
  }

  private nextBatch(
    batchSize: number,
    data: [Float32Array, Uint8Array],
    indexFn: () => number,
  ): { xs: tf.Tensor; ys: tf.Tensor } {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = indexFn();
      const image = data[0].subarray(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label = data[1].subarray(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const ys = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, ys };
  }
}
