export namespace HyperbolicTangent {
  export function activation(synapse) {
    return Math.tanh(synapse);
  }

  export function prime(synapse) {
    const output = activation(synapse);

    return 1.0 - output * output;
  }
}
