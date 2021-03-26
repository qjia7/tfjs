/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class MatMulVectorProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  elementsPerThread: [number, number, number] = [4, 1, 1];
  aShape: [number, number, number];
  addBias: boolean;
  activation: string;
  hasPreluActivationWeights: boolean;
  sharedMemorySize: number;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, addBias = false, activation: string = null,
      hasPreluActivationWeights = false) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.elementsPerThread);

    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.aShape = aShape;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.sharedMemorySize = Math.ceil(this.aShape[2] / 4);
    this.shaderKey = `matMulVector_${this.elementsPerThread}_${activation}_${
        this.sharedMemorySize}`;
  }

  getUserCode(): string {
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivationWeights) {
        activationSnippet = `float activation(float a, ivec3 outCoord) {
              float b = getPreluActivationWeightsAtOutCoords(outCoord);
              ${this.activation}
            }`;
      } else {
        activationSnippet = `
              float activation(float a, ivec3 outCoord) {
                ${this.activation}
              }
            `;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}

      int dimAOuter = ${this.aShape[1]};
      int dimInner = ${this.aShape[2]};
      int dimBOuter = ${this.outputShape[2]};

        vec4 mm_readA(int row, int col) {
          if (row < dimAOuter)
          {
              int index = row * dimInner + col;
              vec4 result = vec4(A[index],
                A[index + 1],
                A[index + 2],
                A[index + 3]);
              return result;
          }
          else {
              return vec4(0, 0, 0, 0);
          }
      }

      vec4 mm_readB(int row, int col) {
        int index = row * dimBOuter + col;
        vec4 result = vec4(B[index],
            B[index + 1],
            B[index + 2],
            B[index + 3]);
        return result;
    }

      void mm_write(int row, int col, float value) {
        if (row < dimAOuter && col < dimBOuter)
        {
          ivec3 outCoord = ivec3(0, row, col);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          int index = row * dimBOuter + col;
          setOutput(index, value);
        }
      }

      const int WORK_PER_THREAD_X = ${this.elementsPerThread[0]};
      const int WORK_PER_THREAD_Y = ${this.elementsPerThread[1]};
      shared vec4 mm_Asub[${this.sharedMemorySize}];
      void main()
      {
        int globalRow = int(gl_GlobalInvocationID.y) * WORK_PER_THREAD_Y;
        int globalCol = int(gl_GlobalInvocationID.x) * WORK_PER_THREAD_X;
        int RowPerThread = WORK_PER_THREAD_Y;
        vec4 acc[WORK_PER_THREAD_Y];
        vec4 ACached;
        vec4 BCached[4];

        // Without this initialization strange values show up in acc.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            acc[innerRow] = vec4(0.0, 0.0, 0.0, 0.0);
        }

        int localIndex = int(gl_LocalInvocationID.x);
        while (localIndex < ${this.sharedMemorySize})
        {
            mm_Asub[localIndex] = mm_readA(globalRow, localIndex * 4);
            localIndex += int(gl_WorkGroupSize.x);
        }
        barrier();

        int sharedDimNearestVec4 = dimInner / 4;
        int sharedDimVec4Remainder = dimInner % 4;
        for (int k = 0; k < sharedDimNearestVec4; k++) {
            BCached[0] = mm_readB(k * 4, globalCol);
            BCached[1] = mm_readB(k * 4 + 1, globalCol);
            BCached[2] = mm_readB(k * 4 + 2, globalCol);
            BCached[3] = mm_readB(k * 4 + 3, globalCol);

            for (int i = 0; i < RowPerThread; i++) {
                ACached = mm_Asub[k];
                acc[i] = BCached[0] * ACached.x + acc[i];
                acc[i] = BCached[1] * ACached.y + acc[i];
                acc[i] = BCached[2] * ACached.z + acc[i];
                acc[i] = BCached[3] * ACached.w + acc[i];
            }
        }

        if (sharedDimVec4Remainder == 1) {
            BCached[0] = mm_readB(dimInner - 1, globalCol);
            for (int i = 0; i < RowPerThread; i++) {
                ACached = mm_Asub[sharedDimNearestVec4];
                acc[i] = BCached[0] * ACached.x + acc[i];
            }
        }
        else if (sharedDimVec4Remainder == 2) {
            BCached[0] = mm_readB(dimInner - 2, globalCol);
            BCached[1] = mm_readB(dimInner - 1, globalCol);
            for (int i = 0; i < RowPerThread; i++) {
                ACached = mm_Asub[sharedDimNearestVec4];
                acc[i] = BCached[0] * ACached.x + acc[i];
                acc[i] = BCached[1] * ACached.y + acc[i];
            }
        }
        else if (sharedDimVec4Remainder == 3) {
            BCached[0] = mm_readB(dimInner - 3, globalCol);
            BCached[1] = mm_readB(dimInner - 2, globalCol);
            BCached[2] = mm_readB(dimInner - 1, globalCol);
            for (int i = 0; i < RowPerThread; i++) {
                ACached = mm_Asub[sharedDimNearestVec4];
                acc[i] = BCached[0] * ACached.x + acc[i];
                acc[i] = BCached[1] * ACached.y + acc[i];
                acc[i] = BCached[2] * ACached.z + acc[i];
            }
        }

        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          mm_write(globalRow + innerRow, globalCol, acc[innerRow].x);
          mm_write(globalRow + innerRow, globalCol + 1, acc[innerRow].y);
          mm_write(globalRow + innerRow, globalCol + 2, acc[innerRow].z);
          mm_write(globalRow + innerRow, globalCol + 3, acc[innerRow].w);
        }
      }
    `;
    return userCode;
  }
}
