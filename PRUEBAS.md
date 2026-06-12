# Documentación de pruebas

Este documento describe la suite automatizada del proyecto, cómo ejecutarla y
qué comportamiento protege cada caso.

## Herramientas

- **Mocha** ejecuta los casos de prueba.
- **@testdeck/mocha** permite declarar suites y pruebas mediante decoradores.
- **Chai** proporciona las aserciones.
- **ts-mocha** ejecuta directamente los archivos TypeScript.
- **NYC** recopila y muestra la cobertura.
- **TensorFlow.js para Node.js** se utiliza en una prueba comparativa de
  entrenamiento.

Las pruebas están en `src/tests/` y siguen el patrón `*-test.ts`.

## Ejecutar las pruebas

Instala primero las dependencias:

```bash
pnpm install
```

Ejecuta la suite completa:

```bash
pnpm test
```

Ejecuta un archivo específico:

```bash
pnpm test-file -- src/tests/backpropagation-test.ts
```

Verifica además tipos, formato y bundle:

```bash
pnpm exec tsc --noEmit
pnpm format:check
pnpm production
```

`pnpm test` ejecuta todos los archivos `src/tests/*-test.ts` sin límite de
tiempo y muestra la cobertura en la terminal.

## Resumen de la suite

| Archivo                                 |  Casos | Alcance principal                                 |
| --------------------------------------- | -----: | ------------------------------------------------- |
| `activation-function-test.ts`           |      4 | Activaciones, derivadas y estabilidad numérica.   |
| `perceptron-test.ts`                    |      9 | Validaciones, convergencia y límites.             |
| `backpropagation-test.ts`               |      7 | Entrenamiento, configuración y múltiples salidas. |
| `backpropagation-import-export-test.ts` |      4 | Validación y persistencia de modelos.             |
| `tensorflow-test.ts`                    |      1 | Referencia de entrenamiento con TensorFlow.js.    |
| **Total**                               | **25** |                                                   |

## Funciones de activación

Archivo: `src/tests/activation-function-test.ts`

| Prueba                                     | Comportamiento protegido                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `binaryActivation`                         | La activación binaria clasifica correctamente y rechaza el cálculo de su derivada para backpropagation. |
| `reluActivationAndDerivative`              | ReLU y su derivada producen los valores esperados para entradas negativas, cero y positivas.            |
| `sigmoidalActivationAndDerivative`         | La sigmoidal y su derivada son correctas y permanecen finitas para valores extremos.                    |
| `hyperbolicTangentActivationAndDerivative` | Tanh y su derivada son correctas para cero y valores extremos.                                          |

## Perceptrón

Archivo: `src/tests/perceptron-test.ts`

| Prueba                                 | Comportamiento protegido                                                                                          |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `validatesTrainingData`                | Rechaza entrenamiento sin muestras, valores no finitos, salidas fuera de `0` y `1`, y dimensiones inconsistentes. |
| `validatesProcessStateAndDimensions`   | Rechaza predicciones antes de configurar el modelo, dimensiones incorrectas y valores no finitos.                 |
| `rejectsNonFiniteSynapticCalculations` | Detiene cálculos sinápticos que producen valores no finitos.                                                      |
| `copiesTrainingData`                   | El perceptrón conserva una copia interna y no depende de mutaciones posteriores de la entrada original.           |
| `testAND`                              | Aprende y clasifica correctamente la compuerta lógica AND.                                                        |
| `testOR`                               | Aprende y clasifica correctamente la compuerta lógica OR.                                                         |
| `testFailXOR`                          | XOR no converge con un perceptrón simple y termina con un error descriptivo al alcanzar `maxEpochs`.              |
| `exposesConvergedTrainingStats`        | Expone épocas, errores finales y estado de convergencia después del entrenamiento.                                |
| `rejectsInvalidMaxEpochs`              | Rechaza límites de épocas no positivos.                                                                           |

## Backpropagation

Archivo: `src/tests/backpropagation-test.ts`

| Prueba                                                | Comportamiento protegido                                                                                               |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `validatesConfigurationAndLayers`                     | Rechaza épocas, learning rate, momentum, semilla, pérdida objetivo, paciencia, activación y tamaños de capa inválidos. |
| `validatesDatasetWithoutMutatingIt`                   | Valida datasets, dimensiones y valores finitos sin modificar las entradas recibidas.                                   |
| `validatesTrainingAndProcessState`                    | Rechaza entrenamiento sin capas y predicciones antes de entrenar o con dimensiones incorrectas.                        |
| `runsExactlyConfiguredEpochsAndSupportsEarlyStopping` | Respeta `epochs: 0`, registra el historial y finaliza temprano al alcanzar la pérdida objetivo.                        |
| `producesDeterministicModelsWithTheSameSeed`          | La misma semilla, configuración y dataset producen modelos serializados idénticos.                                     |
| `learnsTwoOutputs`                                    | Una red con dos neuronas de salida aprende objetivos vectoriales.                                                      |
| `learnsNonLinearProblemsWithScalarTargets`            | Una red multicapa aprende XNOR manteniendo compatibilidad con objetivos escalares.                                     |

Las pruebas de aprendizaje configuran una semilla cuando necesitan resultados
reproducibles. Los casos comparan la salida redondeada porque el modelo produce
probabilidades o valores continuos.

## Importación y exportación

Archivo: `src/tests/backpropagation-import-export-test.ts`

| Prueba                                               | Comportamiento protegido                                                                                        |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `validatesModelsBeforeImportingOrExporting`          | Rechaza modelos vacíos, no entrenados, con versión incompatible o valores no finitos antes de modificar la red. |
| `importsLegacyModelsAndExportsVersionedModels`       | Acepta modelos históricos con `thresholds` y los exporta en el formato versionado con `biases`.                 |
| `preservesSigmoidalAndTanhPredictionsAfterRoundTrip` | Serializar, convertir a JSON e importar conserva exactamente las predicciones sigmoidal y tanh.                 |
| `persistsTrainingConfiguration`                      | Conserva activación, learning rate y momentum al restaurar el modelo.                                           |

## Referencia con TensorFlow.js

Archivo: `src/tests/tensorflow-test.ts`

`trainXOR` construye y entrena una red equivalente con TensorFlow.js para
resolver XNOR. Este caso sirve como ejemplo comparativo y como comprobación de
que la dependencia nativa de TensorFlow.js funciona correctamente.

Durante esta prueba puede aparecer una advertencia de deprecación originada por
una dependencia de TensorFlow.js. La advertencia no indica un fallo de la suite.

## Cobertura

NYC incluye todos los archivos TypeScript y excluye declaraciones y archivos
JavaScript. Al ejecutar `pnpm test`, la terminal muestra cobertura de:

- Sentencias.
- Ramas.
- Funciones.
- Líneas.

La cobertura ayuda a localizar código no ejercitado, pero no sustituye las
aserciones de comportamiento. Una nueva funcionalidad debe incluir casos
positivos, validaciones relevantes y regresiones conocidas.

## Añadir una prueba

1. Añade el caso al archivo correspondiente o crea un archivo con sufijo
   `-test.ts` dentro de `src/tests/`.
2. Declara la clase con `@suite` y el método con `@test`.
3. Usa una semilla cuando el entrenamiento dependa de inicialización aleatoria.
4. Mantén datasets pequeños y expectativas centradas en comportamiento público.
5. Ejecuta la prueba aislada y después la suite completa.

Ejemplo:

```ts
import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

@suite
export class ExampleTest {
  @test
  public describesExpectedBehavior(): void {
    assert.equal(1 + 1, 2);
  }
}
```

Antes de finalizar un cambio:

```bash
pnpm format:check
pnpm exec tsc --noEmit
pnpm test
pnpm production
```
