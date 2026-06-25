import { Injectable, inject, NgZone } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { BehaviorSubject } from 'rxjs';
import { MnistDataService } from './mnist-data.service';
import { ModelBuilderService, LayerDescription, HopfieldNetwork } from './model-builder.service';

export interface TrainingProgress {
  epoch: number;
  batch: number;
  loss: number;
  acc: number;
  valLoss: number;
  valAcc: number;
  isTraining: boolean;
  batchesPerEpoch?: number;
  isEpochEnd?: boolean;
}

@Injectable({
  providedIn: 'root',
})
export class TrainingService {
  private dataService = inject(MnistDataService);
  private builderService = inject(ModelBuilderService);
  private ngZone = inject(NgZone);

  public model: tf.LayersModel | null = null;
  public hopfieldNet: HopfieldNetwork | null = null;
  public currentBackend = 'webgl';

  // Observable streams for UI updates
  public progress$ = new BehaviorSubject<TrainingProgress>({
    epoch: 0,
    batch: 0,
    loss: 0,
    acc: 0,
    valLoss: 0,
    valAcc: 0,
    isTraining: false,
  });

  public logMessages$ = new BehaviorSubject<string[]>([]);
  public isWebGPUSupported = false;

  constructor() {
    this.detectBackendSupport();
  }

  private async detectBackendSupport() {
    try {
      // Comprobar si el backend de WebGPU está registrado en el motor de TensorFlow.js
      // Usamos engine().backendNames() porque findBackend() da null antes de inicializarse
      const isRegistered = tf.engine().backendNames().indexOf('webgpu') !== -1;
      console.log('¿WebGPU registrado en TensorFlow.js?:', isRegistered);
      console.log('¿Navigator GPU disponible?:', !!navigator.gpu);

      if (isRegistered) {
        // Intentar activar WebGPU de forma activa
        await tf.setBackend('webgpu');
        await tf.ready();
        this.isWebGPUSupported = true;
        this.currentBackend = 'webgpu';
        this.log('WebGPU detectado y configurado como backend inicial.');
      } else {
        await tf.setBackend('webgl');
        await tf.ready();
        this.isWebGPUSupported = false;
        this.currentBackend = 'webgl';
        this.log('WebGPU no disponible. Configurado WebGL.');
      }
    } catch (e: any) {
      console.warn('Error inicializando WebGPU, haciendo fallback a WebGL:', e);
      this.isWebGPUSupported = false;
      try {
        await tf.setBackend('webgl');
        await tf.ready();
      } catch (err) {
        // Fallback a CPU si WebGL falla por completo
        await tf.setBackend('cpu');
        await tf.ready();
      }
      this.currentBackend = tf.getBackend();
    }
  }

  private log(message: string) {
    const currentLogs = this.logMessages$.value;
    this.ngZone.run(() => {
      this.logMessages$.next([
        `[${new Date().toLocaleTimeString()}] ${message}`,
        ...currentLogs.slice(0, 49),
      ]);
    });
  }

  async setBackend(backend: 'webgpu' | 'webgl' | 'cpu') {
    try {
      await tf.setBackend(backend);
      await tf.ready();
      this.currentBackend = backend;
      this.log(`Backend cambiado a: ${backend}`);
      return true;
    } catch (e: any) {
      this.log(`Error al configurar backend ${backend}: ${e.message}`);
      return false;
    }
  }

  private compileQueue: Promise<void> = Promise.resolve();

  // Compile the TFJS architecture
  async compileModel(
    layers: LayerDescription[],
    optimizerName: string,
    lr: number,
    lossName: string,
  ) {
    this.compileQueue = this.compileQueue.then(async () => {
      await tf.ready();
      if (this.model) {
        this.model.dispose();
        this.model = null;
      }

      this.log('Compilando modelo dinámico...');
      try {
        this.model = this.builderService.buildModel(layers);

        let optimizer: tf.Optimizer;
        if (optimizerName === 'adam') {
          optimizer = tf.train.adam(lr);
        } else if (optimizerName === 'rmsprop') {
          optimizer = tf.train.rmsprop(lr);
        } else {
          optimizer = tf.train.sgd(lr);
        }

        const loss =
          lossName === 'categoricalCrossentropy' ? 'categoricalCrossentropy' : 'meanSquaredError';

        this.model.compile({
          optimizer,
          loss,
          metrics: ['accuracy'],
        });

        this.model.summary();
        this.log('Modelo compilado con éxito. Listo para entrenamiento.');
      } catch (e: any) {
        this.log(`Error de compilación: ${e.message}`);
        throw e;
      }
    });
    return this.compileQueue;
  }

  // Train the model
  async trainModel(epochs: number, batchSize: number, valSplit = 0.15) {
    // Esperar a que terminen las compilaciones en cola
    await this.compileQueue;

    if (!this.model) {
      this.log('Error: No se ha compilado el modelo.');
      return;
    }

    this.log('Preparando datos para entrenamiento...');

    // Fetch all training images
    const trainData = this.dataService.getTrainData();
    const testData = this.dataService.getTestData();

    const totalSamples = trainData.xs.shape[0];
    const batchesPerEpoch = Math.ceil(totalSamples / batchSize);

    this.log(
      `Iniciando entrenamiento: ${epochs} épocas, Lote: ${batchSize}. backend actual: ${tf.getBackend()}`,
    );

    this.ngZone.run(() => {
      this.progress$.next({
        epoch: 0,
        batch: 0,
        loss: 0,
        acc: 0,
        valLoss: 0,
        valAcc: 0,
        isTraining: true,
        batchesPerEpoch,
        isEpochEnd: false,
      });
    });

    try {
      await this.model.fit(trainData.xs, trainData.ys, {
        epochs,
        batchSize,
        validationData: [testData.xs, testData.ys],
        callbacks: {
          onEpochBegin: async (epoch, logs) => {
            const current = this.progress$.value;
            this.ngZone.run(() => {
              this.progress$.next({
                ...current,
                epoch: epoch + 1,
                batch: 0,
                batchesPerEpoch,
                isEpochEnd: false,
              });
            });
          },
          onBatchEnd: async (batch, logs) => {
            const current = this.progress$.value;
            this.ngZone.run(() => {
              const loss = logs
                ? logs['loss'] !== undefined
                  ? logs['loss']
                  : current.loss
                : current.loss;
              const acc = logs
                ? logs['acc'] !== undefined
                  ? logs['acc']
                  : logs['accuracy'] !== undefined
                    ? logs['accuracy']
                    : current.acc
                : current.acc;
              this.progress$.next({
                ...current,
                batch: batch + 1,
                loss,
                acc,
                batchesPerEpoch,
                isEpochEnd: false,
              });
            });
            // Ceder control al navegador para refrescar UI y logs
            await tf.nextFrame();
          },
          onEpochEnd: (epoch, logs) => {
            if (logs) {
              const loss = logs['loss'] !== undefined ? logs['loss'] : 0;
              const acc =
                logs['acc'] !== undefined
                  ? logs['acc']
                  : logs['accuracy'] !== undefined
                    ? logs['accuracy']
                    : 0;
              const valLoss =
                logs['val_loss'] !== undefined
                  ? logs['val_loss']
                  : logs['valLoss'] !== undefined
                    ? logs['valLoss']
                    : 0;
              const valAcc =
                logs['val_acc'] !== undefined
                  ? logs['val_acc']
                  : logs['val_accuracy'] !== undefined
                    ? logs['val_accuracy']
                    : logs['valAcc'] !== undefined
                      ? logs['valAcc']
                      : 0;

              this.ngZone.run(() => {
                this.progress$.next({
                  epoch: epoch + 1,
                  batch: 0,
                  loss,
                  acc,
                  valLoss,
                  valAcc,
                  isTraining: true,
                  batchesPerEpoch,
                  isEpochEnd: true,
                });
              });

              this.log(
                `Época ${epoch + 1}/${epochs} - Pérdida: ${loss.toFixed(4)} - Prec.: ${(acc * 100).toFixed(2)}% - Val Pérdida: ${valLoss.toFixed(4)} - Val Prec.: ${(valAcc * 100).toFixed(2)}%`,
              );
            }
          },
        },
      });
      this.log('Entrenamiento finalizado correctamente.');
    } catch (e: any) {
      this.log(`Error durante entrenamiento: ${e.message}`);
    } finally {
      // Clean up tensors
      trainData.xs.dispose();
      trainData.ys.dispose();
      testData.xs.dispose();
      testData.ys.dispose();

      const current = this.progress$.value;
      this.ngZone.run(() => {
        this.progress$.next({ ...current, isTraining: false, isEpochEnd: false });
      });
    }
  }

  // Train Hopfield network with 10 representative digits (one for each class 0-9)
  async trainHopfield() {
    await tf.ready();
    this.log('Entrenando Red de Hopfield con 10 patrones representativos de MNIST...');
    this.hopfieldNet = new HopfieldNetwork();

    try {
      // Find 10 representative digits (one of each label 0 to 9)
      const testData = this.dataService.getTestData(100);
      const images = testData.xs.arraySync() as number[][];
      const labels = testData.ys.arraySync() as number[][];

      const patternsToTrain: Float32Array[] = [];
      const foundClasses = new Set<number>();

      for (let i = 0; i < images.length; i++) {
        const classIdx = labels[i].indexOf(1);
        if (!foundClasses.has(classIdx)) {
          foundClasses.add(classIdx);
          patternsToTrain.push(new Float32Array(images[i]));
        }
        if (foundClasses.size === 10) break;
      }

      testData.xs.dispose();
      testData.ys.dispose();

      if (patternsToTrain.length > 0) {
        this.hopfieldNet.train(patternsToTrain);
        this.log(
          `Red de Hopfield entrenada exitosamente con ${patternsToTrain.length} dígitos representativos.`,
        );
      } else {
        this.log('No se pudieron obtener suficientes dígitos de prueba para entrenar Hopfield.');
      }
    } catch (e: any) {
      this.log(`Error al entrenar Hopfield: ${e.message}`);
    }
  }

  // Reconstruct an image with Hopfield
  reconstructWithHopfield(noiseImage: Float32Array, steps = 5): Float32Array {
    if (!this.hopfieldNet) {
      this.log('Error: La red de Hopfield no ha sido entrenada.');
      return noiseImage;
    }

    let state: any = new Float32Array(noiseImage);
    for (let s = 0; s < steps; s++) {
      state = this.hopfieldNet.reconstructStep(state);
    }
    return state;
  }

  // Prediction using TFJS model
  predict(image: Float32Array): { probabilities: number[]; prediction: number } {
    if (!this.model) {
      throw new Error('No model loaded/trained');
    }

    return tf.tidy(() => {
      // Reshape according to first layer requirements
      let tensor = tf.tensor2d(image, [1, 784]);

      const firstLayerType = this.model!.layers[0].getClassName();

      if (firstLayerType === 'Conv2D') {
        tensor = tensor.reshape([1, 28, 28, 1]);
      } else if (firstLayerType === 'SelfAttentionLayer') {
        tensor = tensor.reshape([1, 28, 28]);
      }

      const output = this.model!.predict(tensor) as tf.Tensor;
      const probabilities = Array.from(output.dataSync());
      const prediction = output.argMax(-1).dataSync()[0];

      return { probabilities, prediction };
    });
  }

  // Export TFJS Model
  async exportModel() {
    if (!this.model) {
      this.log('Error: No hay un modelo para exportar.');
      return;
    }
    try {
      this.log('Exportando modelo al navegador...');
      await this.model.save('downloads://mnist-custom-model');
      this.log('Modelo exportado en descargas (archivos .json y .weights.bin).');
    } catch (e: any) {
      this.log(`Error al exportar: ${e.message}`);
    }
  }
}
