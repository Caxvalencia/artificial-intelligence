/**
 * Rectified Linear Unit (ReLU)
 */
export namespace ReLU {
  export function activation(synapse) {
    return synapse > 0 ? synapse : 0;
  }

  export function prime(synapse) {
    return synapse > 0 ? 1 : 0;
  }
}
