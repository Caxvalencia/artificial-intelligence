export namespace Sigmoidal {
  export function activation(synapse) {
    return 1 / (1 + Math.E ** -synapse);
  }

  export function prime(synapse) {
    const ex = Math.E ** synapse;

    return ex / (ex + 1) ** 2;
  }
}
