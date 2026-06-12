# Artificial Intelligence

Implementación educativa de algoritmos básicos de redes neuronales en TypeScript.
El proyecto incluye un perceptrón para clasificación lineal, una red multicapa
entrenada mediante backpropagation, pruebas con TensorFlow.js y demos para el
navegador.

> Este repositorio está orientado al aprendizaje y la experimentación. No está
> preparado como paquete publicado ni pretende reemplazar librerías de machine
> learning para producción.

## Características

- Perceptrón de una capa para problemas linealmente separables.
- Red neuronal multicapa con backpropagation y momentum.
- Funciones de activación binaria, sigmoidal, ReLU y tangente hiperbólica.
- Importación y exportación de modelos entrenados.
- Bundle UMD para ejecutar las demos en el navegador.
- Suite de pruebas en TypeScript y ejemplo equivalente con TensorFlow.js.

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

El entrenamiento lanza `Maximum error limit reached` cuando no logra converger
después de 8000 iteraciones con errores. Esto ocurre, por ejemplo, al intentar
enseñar XOR a un perceptrón simple.

### Red multicapa con backpropagation

Para resolver problemas no lineales se pueden encadenar capas. El siguiente
ejemplo entrena una red para XNOR:

```ts
import { Backpropagation } from './src';

const dataset = [
  { input: [0, 0], output: 1 },
  { input: [0, 1], output: 0 },
  { input: [1, 0], output: 0 },
  { input: [1, 1], output: 1 }
];

const network = new Backpropagation({
  epochs: 15000,
  learningRate: 0.3,
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

## API pública

El punto de entrada `src/index.ts` exporta las clases `Perceptron` y
`Backpropagation`.

### `Perceptron`

| Método                  | Descripción                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| `addData(data, output)` | Añade una muestra de entrenamiento y devuelve la instancia.                  |
| `learn()`               | Entrena hasta converger o alcanzar el límite de errores.                     |
| `process(data)`         | Clasifica una entrada y devuelve `0` o `1` con la activación predeterminada. |
| `setWeights(weights)`   | Reemplaza manualmente los pesos del perceptrón.                              |

Los datos de entrada deben tener siempre la misma longitud.

### `Backpropagation`

```ts
const network = new Backpropagation();
```

| Opción               | Tipo                     | Valor predeterminado | Descripción                                        |
| -------------------- | ------------------------ | -------------------- | -------------------------------------------------- |
| `epochs`             | `number`                 | `1000`               | Cantidad de épocas de entrenamiento.               |
| `activationFunction` | `ActivationFunctionType` | Sigmoidal            | Función aplicada por las neuronas.                 |
| `learningRate`       | `number`                 | `0.3`                | Factor utilizado para actualizar los pesos.        |
| `verbose`            | `boolean`                | `false`              | Muestra información de entrenamiento y predicción. |

Los valores predeterminados se aplican al crear la instancia sin argumentos.
Cuando se entrega un objeto de configuración, `epochs` es obligatorio:

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
| `error`                   | Error calculado en la última muestra procesada.        |
| `history.loss`            | Historial del error guardado durante el entrenamiento. |

### Funciones de activación

Internamente existen los siguientes valores de `ActivationFunctionType`:

- `BINARY`
- `RELU`
- `SIGMOIDAL`
- `HYPERBOLIC_TANGENT`

Actualmente el enum no se exporta desde el punto de entrada público. Para
experimentar dentro del repositorio puede importarse directamente:

```ts
import { ActivationFunctionType } from './src/activation-functions/activation-function';
```

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
  layers: number[];
  thresholds: number[][];
  weights: number[][][];
}
```

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

## Estructura del proyecto

```text
.
├── demo/                         # Demos para navegador
├── src/
│   ├── activation-functions/     # Funciones de activación y derivadas
│   ├── tests/                    # Pruebas unitarias y ejemplo TensorFlow.js
│   ├── backpropagation.ts        # Red multicapa y entrenamiento
│   ├── layer.ts                  # Administración de capas
│   ├── neuron.ts                 # Neurona usada por backpropagation
│   ├── perceptron.ts             # Perceptrón simple
│   ├── synaptic-processor.ts     # Cálculo de sinapsis y actualización
│   └── index.ts                  # API pública
├── package.json
├── pnpm-lock.yaml
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
calcula el error desde la capa de salida hacia las capas anteriores y actualiza
pesos y umbrales. La implementación usa un factor de momentum fijo de `0.77`.

Los pesos y umbrales iniciales son aleatorios, por lo que el resultado exacto y
la velocidad de convergencia pueden variar entre ejecuciones.

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

Las pruebas principales cubren:

- Compuertas AND, OR y el fallo esperado de XOR en el perceptrón.
- OR, AND, XOR y XNOR con backpropagation.
- Importación y exportación de modelos.
- Entrenamiento de XNOR con TensorFlow.js.

## Limitaciones conocidas

- El proyecto todavía no expone declaraciones de tipos ni una entrada preparada
  para instalarse como dependencia externa.
- `Backpropagation` espera una única salida objetivo por muestra.
- Los datasets se procesan completos en memoria y sin mini-batches.
- No existen validaciones exhaustivas para dimensiones incompatibles o modelos
  importados incorrectos.
- Algunas partes de la API interna conservan nombres y decisiones históricas.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta `LICENSE` para más
información.
