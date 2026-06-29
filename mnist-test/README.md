# MNIST Test

Aplicación Angular para construir, entrenar y probar modelos de clasificación
MNIST con TensorFlow.js desde una interfaz visual.

La app es independiente del paquete TypeScript de la raíz. Sirve como laboratorio
web para experimentar con `tf.LayersModel`, backends del navegador y arquitecturas
visuales basadas en nodos.

## Requisitos

- Node.js 22 o superior.
- pnpm 10.33.0 o compatible.
- Navegador con soporte para Canvas y WebGL.
- WebGPU opcional para aceleración cuando el navegador lo permita.
- Conexión de red para descargar MNIST desde `learnjs-data`.

## Comandos

Instala dependencias:

```bash
pnpm install
```

Levanta el servidor de desarrollo:

```bash
pnpm start
```

Abre `http://localhost:4200/`.

Compila la app:

```bash
pnpm build
```

Ejecuta pruebas:

```bash
pnpm test
```

Comprueba formato:

```bash
pnpm format:check
```

## Funcionalidades

- Lienzo visual para componer arquitecturas con nodos.
- Conexiones manuales entre capas, incluyendo ramas con `concatenate` y `add`.
- Zoom, pan y restablecimiento de vista.
- Presets de arquitectura:
  - `Flow CNN`: red convolucional base para imágenes 28x28.
  - `Flow MLP`: perceptrón multicapa denso.
  - `Flow Attention`: reshape secuencial con autoatención.
  - `Flow Híbrido Óptimo`: convoluciones locales más autoatención.
  - `Flow ViT Parches`: enfoque tipo Vision Transformer compacto por parches.
- Capas disponibles: `dense`, `conv2d`, `maxPool2d`, `flatten`, `reshape`,
  `attention`, `dropout`, `concatenate` y `add`.
- Panel de hiperparámetros con optimizador, learning rate, pérdida, épocas y
  batch size.
- Backend TensorFlow.js seleccionable entre WebGPU, WebGL y CPU.
- Métricas de entrenamiento en tiempo real y gráficas de aprendizaje.
- Exportación del modelo entrenado mediante `downloads://mnist-custom-model`.
- Panel de pruebas para dibujar dígitos y ver probabilidades por clase.
- Red de Hopfield para añadir ruido y reconstruir patrones 28x28.

## Datos

`MnistDataService` descarga:

- `mnist_images.png`
- `mnist_labels_uint8`

desde `https://storage.googleapis.com/learnjs-data/model-builder/`.

El dataset contiene 65 000 ejemplos. La app usa una división aproximada de
5/6 para entrenamiento y 1/6 para prueba, normaliza píxeles a `[0, 1]` y entrega
labels one-hot de 10 clases.

## Backends

`TrainingService` intenta inicializar WebGPU si está registrado y disponible en
`navigator.gpu`. Si falla, usa WebGL; si WebGL tampoco inicializa, cae a CPU.

Desde la UI se puede cambiar entre:

- `webgpu`
- `webgl`
- `cpu`

## Estructura

```text
src/app/
├── components/
│   ├── header/             # Estado de datos y selector de backend
│   ├── learning-charts/    # Gráficas de métricas
│   ├── testing-panel/      # Canvas de dibujo y Hopfield
│   ├── training-panel/     # Parámetros, acciones y logs
│   └── workspace/          # Lienzo de arquitectura
├── services/
│   ├── mnist-data.service.ts      # Descarga y batching del dataset
│   ├── model-builder.service.ts   # Construcción funcional de modelos TFJS
│   └── training.service.ts        # Backend, entrenamiento, predicción y exportación
├── app.html
├── app.css
└── app.ts
```

## Notas de uso

- El modelo se recompila cuando cambian capas o parámetros estructurales.
- `conv2d` y `maxPool2d` necesitan recibir tensores 3D; normalmente deben ir
  después de una capa `reshape` con forma `[28, 28, 1]`.
- Las capas `concatenate` y `add` aceptan múltiples entradas; el resto reemplaza
  la conexión previa por la nueva conexión.
- Para clasificación MNIST, la salida final debería tener 10 unidades con
  activación `softmax` cuando se usa `categoricalCrossentropy`.
