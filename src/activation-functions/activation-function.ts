import { Binary } from './binary.function';
import { HyperbolicTangent } from './hyperbolic-tangent.function';
import { Sigmoidal } from './sigmoidal.function';
import { ReLU } from './relu.function';

export enum ActivationFunctionType {
  BINARY = 'BINARY',
  RELU = 'RELU',
  SIGMOIDAL = 'SIGMOIDAL',
  HYPERBOLIC_TANGENT = 'HYPERBOLIC_TANGENT'
}

const callback: Record<
  ActivationFunctionType,
  { activation: (synapse: number) => number; prime: (synapse: number) => number }
> = {
  [ActivationFunctionType.BINARY]: Binary,
  [ActivationFunctionType.RELU]: ReLU,
  [ActivationFunctionType.SIGMOIDAL]: Sigmoidal,
  [ActivationFunctionType.HYPERBOLIC_TANGENT]: HyperbolicTangent
};

export class ActivationFunction {
  protected default: ActivationFunctionType;
  private callback: (synapse: number) => number;
  private callbackPrime: (synapse: number) => number;

  constructor(functionName: ActivationFunctionType = ActivationFunctionType.BINARY) {
    this.default = functionName;
    this.setCallback();
    this.setCallbackPrime();
  }

  /**
   * @static
   * @param {string} functionName
   * @returns
   */
  static init(functionName: ActivationFunctionType) {
    return new ActivationFunction(functionName);
  }

  /**
   * @param {number} synapse
   * @returns
   */
  activation(synapse: number) {
    return this.callback(synapse);
  }

  /**
   * @param {number} synapse
   * @returns
   */
  prime(synapse: number) {
    return this.callbackPrime(synapse);
  }

  primeFromOutput(output: number): number {
    if (this.default === ActivationFunctionType.SIGMOIDAL) {
      return output * (1 - output);
    }

    if (this.default === ActivationFunctionType.HYPERBOLIC_TANGENT) {
      return 1 - output * output;
    }

    return this.prime(output);
  }

  private setCallback() {
    this.callback = callback[this.default].activation;
  }

  private setCallbackPrime() {
    this.callbackPrime = callback[this.default].prime;
  }
}
