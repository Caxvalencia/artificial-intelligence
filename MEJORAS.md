# Plan de mejoras

Este documento recoge mejoras detectadas durante la revisión técnica del
perceptrón, la red multicapa y el entrenamiento por backpropagation.

Las tareas están ordenadas para poder implementarlas una a una. Cada cambio debe
mantener las pruebas existentes, añadir cobertura específica y pasar:

```bash
pnpm format:check
pnpm exec tsc --noEmit
pnpm test
pnpm production
```

## Estado

| ID   | Prioridad | Mejora                                            | Estado     |
| ---- | --------- | ------------------------------------------------- | ---------- |
| M-01 | Alta      | Corregir derivadas de activación                  | Completada |
| M-02 | Alta      | Reemplazar entrenamiento recursivo del perceptrón | Completada |
| M-03 | Media     | Corregir conteo de épocas y pérdida               | Completada |
| M-04 | Alta      | Persistir configuración completa del modelo       | Pendiente  |
| M-05 | Media     | Validar entradas, datasets y modelos              | Pendiente  |
| M-06 | Media     | Mejorar estabilidad numérica                      | Pendiente  |
| M-07 | Alta      | Soportar múltiples neuronas de salida             | Pendiente  |
| M-08 | Baja      | Hacer configurable el entrenamiento               | Pendiente  |
| M-09 | Baja      | Reducir acoplamiento interno                      | Pendiente  |
| M-10 | Baja      | Fortalecer tipos y API pública                    | Pendiente  |

## M-01: Corregir derivadas de activación

**Problema**

- La derivada de ReLU devuelve el valor de la sinapsis en lugar de `0` o `1`.
- La activación binaria no implementa una derivada y rompe el entrenamiento por
  backpropagation con un `TypeError`.

**Archivos relacionados**

- `src/activation-functions/relu.function.ts`
- `src/activation-functions/binary.function.ts`
- `src/activation-functions/activation-function.ts`

**Implementación propuesta**

1. Corregir la derivada de ReLU:

   ```ts
   return synapse > 0 ? 1 : 0;
   ```

2. Decidir el contrato de la activación binaria:
   - Rechazarla explícitamente para backpropagation con un error descriptivo; o
   - Definir una aproximación diferenciable documentada.
3. Validar que toda función seleccionable para backpropagation tenga derivada.

**Decisión implementada**

- ReLU devuelve `1` para sinapsis positivas y `0` para cero o negativas.
- La activación binaria permanece disponible para el perceptrón.
- Intentar usar su derivada en backpropagation genera un error descriptivo,
  porque la función escalón no es diferenciable.

**Criterios de aceptación**

- ReLU devuelve derivadas correctas para valores negativos, cero y positivos.
- Backpropagation con activación binaria no termina en un `TypeError`.
- Existen pruebas unitarias para cada función de activación y su derivada.

## M-02: Reemplazar entrenamiento recursivo del perceptrón

**Problema**

`Perceptron.learn()` ejecuta épocas mediante llamadas recursivas. Un dataset que
no converge, como XOR, puede producir `Maximum call stack size exceeded` antes
de alcanzar el límite configurado.

**Archivo relacionado**

- `src/perceptron.ts`

**Implementación propuesta**

1. Reemplazar `runEpochs()` recursivo por un bucle iterativo.
2. Renombrar `LIMIT_ERRORS` a un concepto preciso, por ejemplo `maxEpochs`.
3. Incluir el número de épocas y el error final en el mensaje de fallo.
4. Devolver estadísticas básicas del entrenamiento o exponerlas en propiedades.

**Criterios de aceptación**

- XOR termina con un error descriptivo y nunca desborda la pila.
- AND y OR continúan convergiendo.
- El número máximo de épocas es configurable.
- Existe una prueba que valida el límite de entrenamiento.

**Decisión implementada**

- El entrenamiento utiliza un bucle iterativo.
- `maxEpochs` tiene un valor predeterminado de `8000` y puede configurarse con
  `setMaxEpochs()`.
- `trainingStats` expone las épocas ejecutadas, los errores de la última época y
  si el entrenamiento convergió.

## M-03: Corregir conteo de épocas y pérdida

**Problema**

- Backpropagation ejecuta una época antes del bucle principal.
- Configurar `epochs: 0` sigue entrenando una vez.
- `history.loss` guarda el error de la última muestra, no la pérdida de toda la
  época.
- `error` también representa únicamente la última muestra procesada.

**Archivo relacionado**

- `src/backpropagation.ts`

**Implementación propuesta**

1. Ejecutar exactamente el número de épocas configurado.
2. Hacer que `runEpoch()` devuelva la pérdida promedio del dataset.
3. Guardar una entrada en `history.loss` por cada época.
4. Definir si `error` representa la pérdida de la última época o eliminarla en
   favor de una propiedad con nombre más preciso.

**Criterios de aceptación**

- `epochs: 0` no modifica pesos ni umbrales.
- `history.loss.length === epochs`.
- Cada valor de `history.loss` representa el promedio del dataset completo.
- Existen pruebas para cero, una y varias épocas.

**Decisión implementada**

- Backpropagation ejecuta exactamente el número de épocas configurado.
- `runEpoch()` devuelve la pérdida promedio de las muestras procesadas.
- `error` representa la pérdida promedio de la última época.
- `history.loss` se reinicia al comenzar cada entrenamiento y guarda una entrada
  por época.

## M-04: Persistir configuración completa del modelo

**Problema**

El modelo exportado solo contiene capas, pesos y umbrales. Al importarlo se
pierden la función de activación, learning rate, momentum y otros parámetros.
Esto puede cambiar las predicciones del modelo restaurado.

**Archivos relacionados**

- `src/backpropagation.ts`
- `src/neuron.ts`

**Implementación propuesta**

1. Versionar el formato del modelo.
2. Incluir como mínimo:
   - Versión del formato.
   - Función de activación.
   - Arquitectura.
   - Pesos.
   - Umbrales o biases.
   - Learning rate.
   - Momentum.
3. Mantener compatibilidad con modelos antiguos o fallar con un mensaje claro.
4. Validar el modelo antes de modificar la red actual.

**Formato sugerido**

```ts
interface SerializedModel {
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

**Criterios de aceptación**

- Una red importada genera las mismas predicciones que la red exportada.
- La prueba cubre al menos sigmoidal y tangente hiperbólica.
- Un modelo inválido genera un error descriptivo.

## M-05: Validar entradas, datasets y modelos

**Problema**

Actualmente varias entradas inválidas terminan en errores internos poco claros,
resultados silenciosamente incorrectos o valores `NaN`.

**Casos que deben validarse**

- Entrenamiento sin datos.
- Predicción antes de entrenar o importar pesos.
- Muestras con dimensiones diferentes.
- Entrada de predicción con dimensión incorrecta.
- Dataset vacío.
- Número de capas o neuronas inválido.
- Pesos, umbrales y arquitectura incompatibles al importar.
- Valores no finitos como `NaN` o `Infinity`.

**Archivos relacionados**

- `src/perceptron.ts`
- `src/backpropagation.ts`
- `src/layer.ts`
- `src/synaptic-processor.ts`

**Implementación propuesta**

1. Crear tipos reutilizables para muestras y datasets.
2. Validar datos en los límites públicos de la API.
3. No modificar el dataset recibido por el consumidor.
4. Utilizar errores descriptivos con contexto de la dimensión esperada y real.

**Criterios de aceptación**

- Todos los casos inválidos anteriores tienen una prueba.
- Ninguna entrada inválida produce errores internos genéricos.
- `Backpropagation.learn()` no muta el dataset recibido.

## M-06: Mejorar estabilidad numérica

**Problema**

La derivada sigmoidal actual puede producir `NaN` para valores grandes porque
calcula exponenciales directamente.

**Archivos relacionados**

- `src/activation-functions/sigmoidal.function.ts`
- `src/activation-functions/hyperbolic-tangent.function.ts`

**Implementación propuesta**

1. Calcular la derivada sigmoidal usando la salida:

   ```ts
   const output = activation(synapse);
   return output * (1 - output);
   ```

2. Añadir pruebas con valores extremos positivos y negativos.
3. Considerar gradient clipping configurable para backpropagation.
4. Detectar y rechazar pesos, errores o pérdidas no finitas.

**Criterios de aceptación**

- Activaciones y derivadas nunca devuelven `NaN` para entradas finitas.
- Existen pruebas con valores extremos.
- El entrenamiento falla explícitamente si aparecen valores no finitos.

## M-07: Soportar múltiples neuronas de salida

**Problema**

Backpropagation recibe un único objetivo escalar y lo aplica a todas las neuronas
de salida. Además, los errores de capas ocultas pueden sobrescribirse al procesar
cada salida.

**Archivos relacionados**

- `src/backpropagation.ts`
- `src/neuron.ts`
- `src/layer.ts`

**Implementación propuesta**

1. Cambiar el dataset para aceptar objetivos vectoriales:

   ```ts
   interface TrainingSample {
     input: ArrayLike<number>;
     output: ArrayLike<number>;
   }
   ```

2. Validar que la dimensión del objetivo coincida con la última capa.
3. Calcular todos los errores de salida antes de propagar hacia atrás.
4. Acumular gradientes por capa antes de actualizar pesos.
5. Mantener temporalmente compatibilidad con objetivos escalares para redes con
   una sola salida.

**Criterios de aceptación**

- Una red con dos salidas aprende un problema de prueba.
- Los gradientes ocultos incluyen la contribución de todas las salidas.
- Un objetivo con dimensión incorrecta genera un error descriptivo.

## M-08: Hacer configurable el entrenamiento

**Problema**

Algunas decisiones importantes están codificadas directamente:

- Momentum fijo en `0.77`.
- Inicialización aleatoria sin semilla.
- Orden fijo del dataset.
- Sin early stopping.
- Sin tolerancia o objetivo de pérdida.

**Archivos relacionados**

- `src/backpropagation.ts`
- `src/neuron.ts`
- `src/perceptron.ts`

**Implementación propuesta**

Agregar opciones como:

```ts
interface TrainingConfig {
  epochs: number;
  learningRate: number;
  momentum: number;
  seed?: number;
  shuffle?: boolean;
  targetLoss?: number;
  patience?: number;
}
```

También conviene utilizar una estrategia de inicialización acorde con la función
de activación, como Xavier/Glorot o He.

**Criterios de aceptación**

- Dos entrenamientos con la misma semilla producen el mismo resultado.
- Momentum y learning rate pueden configurarse.
- Early stopping puede finalizar antes del máximo de épocas.
- El comportamiento predeterminado queda documentado.

## M-09: Reducir acoplamiento interno

**Problema**

- Todas las neuronas comparten un `SynapticProcessor` mutable.
- `Neuron` hereda de `Perceptron`, aunque representan conceptos diferentes.
- La propagación, el almacenamiento de estado y la actualización de pesos están
  distribuidos entre varias clases con responsabilidades solapadas.

**Archivos relacionados**

- `src/neuron.ts`
- `src/perceptron.ts`
- `src/layer.ts`
- `src/synaptic-processor.ts`
- `src/backpropagation.ts`

**Implementación propuesta**

1. Reemplazar la herencia `Neuron extends Perceptron` por composición.
2. Hacer que cada neurona gestione sus propios pesos, bias y caché de forward.
3. Evitar un procesador mutable compartido entre neuronas.
4. Separar claramente:
   - Forward pass.
   - Backward pass.
   - Actualización del optimizador.
   - Serialización.
5. Considerar representar pesos por matrices y biases por vectores a nivel de
   capa.

**Criterios de aceptación**

- No existe estado mutable compartido accidentalmente entre neuronas.
- Cada clase tiene una responsabilidad clara.
- Las pruebas de comportamiento actuales siguen pasando.
- El nuevo diseño facilita implementar múltiples salidas y mini-batches.

## M-10: Fortalecer tipos y API pública

**Problema**

La implementación usa varios `any`, propiedades sin tipos precisos y contratos
que no se exportan desde la API pública.

**Archivos relacionados**

- `src/index.ts`
- `src/perceptron.ts`
- `src/backpropagation.ts`
- `src/synaptic-processor.ts`
- `tsconfig.json`

**Implementación propuesta**

1. Exportar tipos públicos:
   - `ActivationFunctionType`
   - `TrainingSample`
   - `BackpropagationConfig`
   - `SerializedModel`
2. Reemplazar `any` por tipos concretos.
3. Activar gradualmente opciones estrictas de TypeScript.
4. Añadir tipos de retorno a métodos públicos.
5. Normalizar nombres:
   - `threshold` por `bias`, si se adopta la convención habitual.
   - Variables locales en camelCase.
   - Nombres de pruebas que describan el comportamiento real.

**Criterios de aceptación**

- La API pública puede utilizarse sin imports internos.
- No quedan `any` evitables en el núcleo.
- TypeScript detecta dimensiones y configuraciones inválidas cuando sea posible.

## Pruebas adicionales recomendadas

- Pruebas unitarias aisladas para cada activación y derivada.
- Gradient checking numérico para una red pequeña.
- Round trip de serialización para cada activación.
- Entrenamiento determinista con semilla.
- Dataset vacío y dimensiones inválidas.
- Una red con múltiples salidas.
- Estabilidad con entradas grandes y pequeñas.
- Conteo exacto de épocas e historial de pérdida.
- Pruebas que confirmen que los datasets no se mutan.

## Orden recomendado de implementación

1. M-01: corregir derivadas.
2. M-02: eliminar recursividad.
3. M-03: corregir épocas y pérdida.
4. M-05: añadir validaciones.
5. M-06: estabilidad numérica.
6. M-04: versionar persistencia.
7. M-08: configurar entrenamiento.
8. M-07: múltiples salidas.
9. M-09: reducir acoplamiento.
10. M-10: cerrar tipos y API pública.

Las mejoras M-07 y M-09 tienen mayor impacto arquitectónico. Conviene abordarlas
después de asegurar las correcciones matemáticas, validaciones y pruebas de
regresión.
