<div align="center">
  <img src="docs/assets/artificial-intelligence-logo.svg" alt="Artificial Intelligence Logo" width="520" />
  <h1>Artificial Intelligence</h1>
  <p><b>Redes neuronales educativas en TypeScript con demos web y pruebas TensorFlow.js</b></p>

  <p>
    <a href="https://nodejs.org/" target="_blank" rel="noopener noreferrer">
      <img src="https://img.shields.io/badge/Node.js-%3E%3D22-339933?style=flat-square&logo=node.js&logoColor=white" alt="Node.js" />
    </a>
    <a href="https://www.typescriptlang.org/" target="_blank" rel="noopener noreferrer">
      <img src="https://img.shields.io/badge/TypeScript-v5.9-blue?style=flat-square&logo=typescript" alt="TypeScript" />
    </a>
    <a href="https://pnpm.io/" target="_blank" rel="noopener noreferrer">
      <img src="https://img.shields.io/badge/pnpm-v10.33-orange?style=flat-square&logo=pnpm" alt="pnpm" />
    </a>
    <a href="https://www.tensorflow.org/js" target="_blank" rel="noopener noreferrer">
      <img src="https://img.shields.io/badge/TensorFlow.js-v4.22-ff6f00?style=flat-square&logo=tensorflow" alt="TensorFlow.js" />
    </a>
    <a href="https://angular.dev/" target="_blank" rel="noopener noreferrer">
      <img src="https://img.shields.io/badge/MNIST%20demo-Angular%2021-red?style=flat-square&logo=angular" alt="Angular MNIST demo" />
    </a>
  </p>
</div>

---

**Artificial Intelligence** es una implementación educativa de algoritmos
básicos de redes neuronales en TypeScript. El proyecto incluye un perceptrón
para clasificación lineal, una red multicapa entrenada mediante
backpropagation, pruebas con TensorFlow.js y demos para el navegador.

La librería permite experimentar con compuertas lógicas, fronteras de decisión,
funciones de activación, entrenamiento determinista, importación y exportación
de modelos, además de una app Angular para validar flujos de entrenamiento con
MNIST.

> Este repositorio está orientado al aprendizaje y la experimentación. No está
> preparado como paquete publicado ni pretende reemplazar librerías de machine
> learning para producción.

---

## Características

- Perceptrón de una capa para problemas linealmente separables.
- Red neuronal multicapa con backpropagation, momentum y múltiples salidas.
- Funciones de activación binaria, sigmoidal, ReLU y tangente hiperbólica.
- Importación y exportación de modelos entrenados.
- Bundle UMD para ejecutar las demos en el navegador.
- Suite de pruebas en TypeScript y ejemplo equivalente con TensorFlow.js.
- App Angular de experimentación con MNIST, TensorFlow.js, WebGPU/WebGL/CPU y
  arquitecturas visuales configurables.

## Requisitos

- Node.js 22 o superior.
- pnpm 10.33.0 o compatible.

Comprueba las versiones instaladas:

```bash
node --version
pnpm --version
```

## Primeros pasos

Instala las dependencias:

```bash
pnpm install
```

Ejecuta las pruebas:

```bash
pnpm test
```

Genera los bundles de producción:

```bash
pnpm production
```

Los archivos compilados se generan en:

- `dist/bundle.js`: bundle principal.
- `demo/dist/bundle.js`: bundle utilizado por las demos.

Para ejecutar la app visual de MNIST:

```bash
cd mnist-test
pnpm install
pnpm start
```

Después abre `http://localhost:4200/`.

## Uso rápido

### Perceptrón

El perceptrón aprende problemas linealmente separables, como las compuertas
lógicas AND y OR. XOR no puede resolverse con una sola capa.

```ts
import { Perceptron } from './src';

const and = new Perceptron();

and
  .addData(new Float64Array([0, 0]), 0)
  .addData(new Float64Array([0, 1]), 0)
  .addData(new Float64Array([1, 0]), 0)
  .addData(new Float64Array([1, 1]), 1)
  .learn();

console.log(and.process(new Float64Array([1, 1]))); // 1
console.log(and.process(new Float64Array([0, 1]))); // 0
```

El entrenamiento intenta como máximo 8000 épocas de forma predeterminada. El
límite puede configurarse y las estadísticas de la última ejecución quedan
disponibles en `trainingStats`:

```ts
and.setMaxEpochs(2000).learn();

console.log(and.trainingStats);
// { epochs: number, errors: number, converged: boolean }
```

Si no converge, lanza un error descriptivo. Esto ocurre, por ejemplo, al
intentar enseñar XOR a un perceptrón simple.

### Red multicapa con backpropagation

Para resolver problemas no lineales se pueden encadenar capas. El siguiente
ejemplo entrena una red para XNOR:

```ts
import { Backpropagation, type TrainingSample } from './src';

const dataset: TrainingSample[] = [
  { input: [0, 0], output: 1 },
  { input: [0, 1], output: 0 },
  { input: [1, 0], output: 0 },
  { input: [1, 1], output: 1 }
];

const network = new Backpropagation({
  epochs: 15000,
  learningRate: 0.3,
  momentum: 0.77,
  seed: 42,
  shuffle: true,
  targetLoss: 0.002,
  verbose: false
});

network.addLayer(3).addLayer(1).learn(dataset);

for (const sample of dataset) {
  const prediction = network.process(sample.input)[0];
  console.log(sample.input, Math.round(prediction));
}
```

La primera llamada a `addLayer` crea la primera capa entrenable. El tamaño de la
entrada se infiere a partir de cada muestra del dataset; no debe añadirse una
capa específica para la entrada.

## App visual MNIST

La carpeta `mnist-test/` contiene una aplicación Angular independiente para
probar TensorFlow.js en el navegador con el dataset MNIST. No usa directamente
la clase `Backpropagation` del paquete raíz; funciona como laboratorio visual
para construir modelos `tf.LayersModel`.

Incluye:

- Lienzo de arquitectura con nodos conectables y zoom/pan.
- Presets `Flow CNN`, `Flow MLP`, `Flow Attention`, `Flow Híbrido Óptimo` y
  `Flow ViT Parches`.
- Capas `dense`, `conv2d`, `maxPool2d`, `flatten`, `reshape`, `attention`,
  `dropout`, `concatenate` y `add`.
- Configuración de optimizador (`adam`, `rmsprop`, `sgd`), learning rate,
  función de pérdida, épocas y batch size.
- Selección de backend TensorFlow.js entre WebGPU, WebGL y CPU, con fallback
  automático cuando WebGPU no está disponible.
- Carga automática de MNIST desde los assets públicos de `learnjs-data`.
- Gráficas de aprendizaje, consola de entrenamiento, exportación del modelo a
  descargas y panel para dibujar dígitos y ver probabilidades.
- Red de Hopfield entrenada con patrones representativos de MNIST para añadir
  ruido y reconstruir dígitos.

Consulta [`mnist-test/README.md`](mnist-test/README.md) para los comandos y la
estructura específica de esta app.

## API pública

El punto de entrada `src/index.ts` exporta `Perceptron`, `Backpropagation`,
`ActivationFunctionType` y los tipos públicos de configuración, muestras,
historial y serialización.

### `Perceptron`

| Método                  | Descripción                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| `addData(data, output)` | Añade una muestra de entrenamiento y devuelve la instancia.                  |
| `learn()`               | Entrena hasta converger o alcanzar el límite de errores.                     |
| `process(data)`         | Clasifica una entrada y devuelve `0` o `1` con la activación predeterminada. |
| `setWeights(weights)`   | Reemplaza manualmente los pesos del perceptrón.                              |
| `setMaxEpochs(epochs)`  | Configura el límite de épocas y devuelve la instancia.                       |
| `trainingStats`         | Estadísticas de la última ejecución de entrenamiento.                        |

Los datos de entrada deben tener siempre la misma longitud.

El perceptrón valida que las entradas contengan números finitos, que las salidas
sean `0` o `1` y que la predicción use la dimensión con la que fue entrenado.

### `Backpropagation`

```ts
const network = new Backpropagation();
```

| Opción               | Valor predeterminado | Descripción                                             |
| -------------------- | -------------------- | ------------------------------------------------------- |
| `epochs`             | `1000`               | Máximo de épocas de entrenamiento.                      |
| `activationFunction` | `SIGMOIDAL`          | Función aplicada por las neuronas.                      |
| `learningRate`       | `0.3`                | Factor utilizado para actualizar pesos y biases.        |
| `momentum`           | `0.77`               | Contribución de la actualización anterior, entre 0 y 1. |
| `seed`               | Sin definir          | Semilla entera para resultados reproducibles.           |
| `shuffle`            | `false`              | Mezcla una copia del dataset en cada época.             |
| `targetLoss`         | Sin definir          | Detiene el entrenamiento al alcanzar esta pérdida.      |
| `patience`           | Sin definir          | Detiene el entrenamiento tras épocas sin mejora.        |
| `verbose`            | `false`              | Muestra información periódica.                          |

Todas las opciones son opcionales:

```ts
const network = new Backpropagation({
  epochs: 5000,
  learningRate: 0.3,
  verbose: false
});
```

| Método o propiedad        | Descripción                                            |
| ------------------------- | ------------------------------------------------------ |
| `addLayer(numberNeurons)` | Añade una capa y devuelve la instancia.                |
| `learn(dataset)`          | Entrena la red y devuelve la instancia.                |
| `process(data)`           | Devuelve un arreglo con las salidas de la última capa. |
| `exportModel()`           | Devuelve capas, umbrales y pesos serializables.        |
| `importModel(model)`      | Reconstruye una red desde un modelo exportado.         |
| `error`                   | Pérdida promedio calculada en la última época.         |
| `history.loss`            | Pérdida promedio guardada para cada época ejecutada.   |
| `history.epochs`          | Número real de épocas ejecutadas.                      |
| `history.stoppedEarly`    | Indica si finalizó por pérdida objetivo o paciencia.   |

Backpropagation requiere al menos una capa y un dataset no vacío. Todas las
muestras deben tener la misma dimensión y contener números finitos. El
entrenamiento crea copias internas, por lo que no modifica el dataset recibido.
Para una única salida puede usarse un objetivo escalar. Si la última capa tiene
varias neuronas, cada muestra debe proporcionar un vector de igual dimensión:

```ts
const dataset: TrainingSample[] = [
  { input: [0, 1], output: [0, 1] },
  { input: [1, 0], output: [1, 0] }
];

new Backpropagation({ epochs: 2000, seed: 7 }).addLayer(3).addLayer(2).learn(dataset);
```

Las activaciones y actualizaciones también se comprueban durante el
entrenamiento. Si una suma, pérdida, peso o umbral deja de ser finito, el proceso
falla explícitamente en lugar de continuar con valores `NaN` o infinitos.

Notas de configuración:

- `epochs` acepta `0` para inicializar historial sin ejecutar entrenamiento.
- `learningRate` debe ser un número finito positivo.
- `momentum` debe estar en el rango `[0, 1)`.
- `seed`, cuando se define, debe ser un entero y controla inicialización y
  mezcla para resultados reproducibles.
- `patience` debe ser un entero positivo y se evalúa contra mejoras estrictas de
  pérdida.
- `targetLoss` debe ser un número finito no negativo.

### Funciones de activación

La API pública expone los siguientes valores de `ActivationFunctionType`:

- `BINARY`
- `RELU`
- `SIGMOIDAL`
- `HYPERBOLIC_TANGENT`

```ts
import { ActivationFunctionType } from './src';
```

La activación binaria está destinada al perceptrón simple. No puede utilizarse
para backpropagation porque la función escalón no es diferenciable.

## Guardar y cargar modelos

`exportModel()` produce un objeto JSON serializable:

```ts
const exported = network.exportModel();
const json = JSON.stringify(exported);

const restored = new Backpropagation();
restored.importModel(JSON.parse(json));

console.log(restored.process([0, 0]));
```

Formato del modelo:

```ts
interface Model {
  version: 1;
  config: {
    activationFunction: ActivationFunctionType;
    learningRate: number;
    momentum: number;
  };
  layers: number[];
  biases: number[][];
  weights: number[][][];
}
```

La importación también acepta modelos históricos con `thresholds`. Toda nueva
exportación utiliza el formato versionado con `biases`.

## Scripts disponibles

| Comando                       | Descripción                                               |
| ----------------------------- | --------------------------------------------------------- |
| `pnpm test`                   | Ejecuta todas las pruebas y muestra la cobertura con NYC. |
| `pnpm test-file -- <archivo>` | Ejecuta un archivo de prueba específico.                  |
| `pnpm dev`                    | Genera los bundles en modo desarrollo.                    |
| `pnpm watch`                  | Regenera los bundles al detectar cambios.                 |
| `pnpm prod`                   | Alias de la compilación de producción.                    |
| `pnpm production`             | Genera bundles minificados para producción y demos.       |
| `pnpm demo`                   | Sirve el directorio `demo/` en `http://localhost:6767`.   |
| `pnpm format`                 | Formatea los archivos mantenidos con Prettier.            |
| `pnpm format:check`           | Comprueba el formato sin modificar archivos.              |

Ejemplo para ejecutar una prueba específica:

```bash
pnpm test-file -- src/tests/perceptron-test.ts
```

## Demos en el navegador

Primero genera el bundle y levanta el servidor:

```bash
pnpm production
pnpm demo
```

Después abre una de estas rutas:

- `http://localhost:6767/decision-border/`: clasificación visual y frontera de decisión.

El bundle UMD expone `Perceptron` y `Backpropagation` como variables globales:

```html
<script src="/dist/bundle.js"></script>
<script>
  const perceptron = new Perceptron();
</script>
```

Las carpetas `demo/perceptron-graficas-de-datos/` y `demo/perceptron-red/`
contienen experimentos históricos que usan partes de una API anterior. Se
conservan como referencia, pero no forman parte de las demos mantenidas.

## App Angular MNIST

La app vive en `mnist-test/` y tiene sus propias dependencias, lockfile y
scripts:

| Comando                              | Descripción                                     |
| ------------------------------------ | ----------------------------------------------- |
| `cd mnist-test && pnpm start`        | Sirve la app en `http://localhost:4200/`.       |
| `cd mnist-test && pnpm build`        | Compila la aplicación Angular.                  |
| `cd mnist-test && pnpm test`         | Ejecuta las pruebas configuradas por Angular.   |
| `cd mnist-test && pnpm format`       | Formatea `src/**/*.{ts,html,css}` con Prettier. |
| `cd mnist-test && pnpm format:check` | Comprueba formato sin modificar archivos.       |

La app descarga MNIST desde Google Cloud Storage al arrancar. Necesita conexión
de red y un navegador compatible con Canvas, WebGL y, opcionalmente, WebGPU.

## Estructura del proyecto

```text
.
├── docs/
│   └── assets/                   # Imagen y recursos del README
├── demo/                         # Demos para navegador
├── mnist-test/                   # App Angular + TensorFlow.js para MNIST
│   ├── src/app/components/       # Header, workspace, paneles y gráficas
│   ├── src/app/services/         # Datos MNIST, builder TFJS y entrenamiento
│   ├── angular.json
│   └── package.json
├── src/
│   ├── activation-functions/     # Funciones de activación y derivadas
│   ├── tests/                    # Pruebas unitarias y ejemplo TensorFlow.js
│   ├── backpropagation.ts        # Red multicapa y entrenamiento
│   ├── layer.ts                  # Forward, gradientes y actualización por capa
│   ├── neuron.ts                 # Estado independiente de cada neurona
│   ├── perceptron.ts             # Perceptrón simple
│   ├── synaptic-processor.ts     # Cálculo de sinapsis y actualización
│   ├── types.ts                  # Contratos públicos
│   └── index.ts                  # API pública
├── package.json
├── patches/                      # Patch local de @tensorflow/tfjs-node
├── pnpm-lock.yaml
├── PRUEBAS.md                    # Documentación detallada de la suite
├── tsconfig.json
└── webpack.config.js
```

## Cómo funciona

### Perceptrón simple

Cada muestra se procesa calculando la suma ponderada de sus entradas más el
umbral. Cuando la salida no coincide con el valor esperado, se recalculan los
pesos usando el error y el factor de aprendizaje.

### Backpropagation

La red realiza una propagación hacia delante para obtener una predicción,
calcula todos los deltas de salida, propaga los gradientes hacia las capas
anteriores y después actualiza pesos y biases. Momentum y orden del dataset son
configurables. Una semilla permite reproducir inicialización, mezcla y resultado.

## Desarrollo

Flujo recomendado antes de enviar cambios:

```bash
pnpm install --frozen-lockfile
pnpm exec tsc --noEmit
pnpm format:check
pnpm test
pnpm production
pnpm audit
```

Para validar también la app Angular:

```bash
cd mnist-test
pnpm install --frozen-lockfile
pnpm format:check
pnpm test
pnpm build
```

Las pruebas principales cubren:

- Compuertas AND, OR y el fallo esperado de XOR en el perceptrón.
- XNOR y objetivos de múltiples salidas con backpropagation.
- Configuración, validaciones, entrenamiento determinista y early stopping.
- Importación legacy, exportación versionada y round trip de modelos.
- Entrenamiento comparativo de XNOR con TensorFlow.js.

Consulta [`PRUEBAS.md`](PRUEBAS.md) para ver los comandos, el alcance y la
descripción de cada caso de prueba.

## Limitaciones conocidas

- El proyecto todavía no publica declaraciones compiladas ni una entrada
  preparada para instalarse como dependencia externa.
- Los datasets se procesan completos en memoria y sin mini-batches.
- Todas las capas de una red usan la misma función de activación.
- El estado del optimizador no se serializa; al reanudar entrenamiento, momentum
  comienza desde cero.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta `LICENSE` para más
información.
