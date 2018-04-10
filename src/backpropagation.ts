import { Neuron } from './neuron';

'use strict';
export class Backpropagation {
    LIMIT_ERRORS: number;
    counterErrors: number;
    funcionActivacion: string;
    error: number;
    factorAprendizaje: number;
    layers: Neuron[][];

    /**
     * @construtor
     */
    constructor() {
        this.layers = [];
        this.factorAprendizaje = 0.25;
        this.error = 0;
        this.funcionActivacion = 'sigmoidal';

        this.counterErrors = 0;
        this.LIMIT_ERRORS = 100000;
    }

    forwardPropagationData(data) {
        let outputs = [];
        let values = data.valor;

        this.layers.forEach((layer: Neuron[]) => {
            if (outputs.length > 0) {
                values = outputs;
                outputs = [];
            }

            layer.forEach((neuron: Neuron) => {
                neuron.setData(values);

                if (!neuron.weights) {
                    neuron.assignWeights();
                }

                neuron.calculateSynapses();

                if (neuron.outputNeurons.length > 0) {
                    outputs.push(neuron.output());
                } else {
                    neuron.calculateError(data.salida);
                }
            });
        });

        return this;
    }

    learn(datas) {
        let learnCallback = () => {
            const indexLastLayer = this.layers.length - 1;
            let sumErrors = 0;

            datas.forEach(data => {
                // Forwardpropagation
                this.forwardPropagationData(data);

                this.layers[indexLastLayer].forEach((neurona: Neuron) => {
                    //Solo con una neurona tenemos acceso a todas las otras
                    neurona.backpropagation();

                    return false;
                });

                //Reajustar pesos
                this.layers.forEach(capa => {
                    capa.forEach((neurona: Neuron) => {
                        neurona.reajustarPesos();
                        sumErrors += Math.pow(neurona.error, 2);
                    });
                });

                sumErrors *= 0.5;
            });

            this.setError(sumErrors);
        };

        learnCallback();

        while (this.error > 0.0001) {
            this.counterErrors++;

            if (this.counterErrors >= this.LIMIT_ERRORS) {
                // this.counterErrors = 0;
                return this;
            }

            learnCallback();
        }

        this.counterErrors = 0;

        return this;
    }

    addLayer(cantNeuronas) {
        let capa: Neuron[] = [];
        let tieneCapas: boolean = this.layers.length > 0;

        for (let i = 0; i < cantNeuronas; i++) {
            capa[i] = new Neuron(tieneCapas);
            capa[i].setLearningFactor(this.factorAprendizaje);
        }

        let indiceNuevaCapa = this.layers.push(capa) - 1;
        let capaAnterior = this.layers[indiceNuevaCapa - 1];

        // Verificar si existe capa anterior
        if (capaAnterior) {
            // Apuntar con cada Neuron de la nueva capa a la anterior
            capa.forEach(neurona => {
                neurona.inputNeurons = capaAnterior;
            });

            // Apuntar con cada neurona de la capa anterior a la nueva capa
            capaAnterior.forEach(neurona => {
                neurona.neuronasSalida = capa;
            });
        }

        return this;
    }

    addNeurona(capa, posicion) {
        let neurona = new Neuron();
        neurona.setLearningFactor(this.factorAprendizaje);

        if (posicion) {
            this.layers[capa][posicion] = neurona;

            return;
        }

        this.layers[capa].push(neurona);
    }

    process(data) {
        let outputs = [];

        this.layers.forEach(layer => {
            if (outputs.length > 0) {
                data = outputs;
                outputs = [];
            }

            layer.forEach((neuron: Neuron) => {
                outputs.push(neuron.process(data));
            });
        });

        return outputs.map(output => {
            return Math.round(output);
        });
    }

    setError(error) {
        this.error = error;

        return this;
    }
}