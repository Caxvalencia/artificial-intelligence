export namespace HyperbolicTangent {
  export function activation(synapse: number): number {
    return Math.tanh(synapse);
  }

  export function prime(synapse: number): number {
    const output = activation(synapse);

    return 1.0 - output * output;
  }
}
