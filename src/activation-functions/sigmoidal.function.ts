export namespace Sigmoidal {
  export function activation(synapse: number): number {
    if (synapse >= 0) {
      const exponential = Math.exp(-synapse);

      return 1 / (1 + exponential);
    }

    const exponential = Math.exp(synapse);

    return exponential / (1 + exponential);
  }

  export function prime(synapse: number): number {
    const output = activation(synapse);

    return output * (1 - output);
  }
}
