/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {util} from '@tensorflow/tfjs-core';
import {computeDispatch, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {matMulHeader} from './matmul_webgpu';
import {WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedSource(workPerThread: number[]): string {
  return `
    ${matMulHeader}

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${workPerThread[0]};
    const int TILE_M = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TILE_N = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TILE_K = TILE_M;
    const int VEC_SIZE = 1;

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int width0 = dimInner;
      int group_x = int(gl_WorkGroupID.x);
      int group_y = int(gl_WorkGroupID.y);
      int local_x = int(gl_LocalInvocationID.x);
      int local_y = int(gl_LocalInvocationID.y);
      float dot[16];
      for (int i = 0; i < 16; i++)
      {
        dot[i] = 0.0;
      }

      int globalRow = group_y * TILE_M + 16 * local_y;
      int colA = local_x;
      int globalCol = local_x + (group_x * (TILE_N / VEC_SIZE));
      int rowB = 0;
      int w = 0;
      do
      {
        float brow0 = mm_readB(rowB, globalCol); rowB++;
        float brow1 = mm_readB(rowB, globalCol); rowB++;
        float brow2 = mm_readB(rowB, globalCol); rowB++;
        float brow3 = mm_readB(rowB, globalCol); rowB++;
        float brow4 = mm_readB(rowB, globalCol); rowB++;
        float brow5 = mm_readB(rowB, globalCol); rowB++;
        float brow6 = mm_readB(rowB, globalCol); rowB++;
        float brow7 = mm_readB(rowB, globalCol); rowB++;
        float brow8 = mm_readB(rowB, globalCol); rowB++;
        float brow9 = mm_readB(rowB, globalCol); rowB++;
        float browa = mm_readB(rowB, globalCol); rowB++;
        float browb = mm_readB(rowB, globalCol); rowB++;
        float browc = mm_readB(rowB, globalCol); rowB++;
        float browd = mm_readB(rowB, globalCol); rowB++;
        float browe = mm_readB(rowB, globalCol); rowB++;
        float browf = mm_readB(rowB, globalCol); rowB++;

        float arow;
#define MM_DOT_PRODUCT(_row, _dot)  \
        arow = mm_readA(globalRow + _row, colA);              \
        _dot = fma(subgroupShuffle(arow, 0u), brow0, _dot);   \
        _dot = fma(subgroupShuffle(arow, 1u), brow1, _dot);   \
        _dot = fma(subgroupShuffle(arow, 2u), brow2, _dot);   \
        _dot = fma(subgroupShuffle(arow, 3u), brow3, _dot);   \
        _dot = fma(subgroupShuffle(arow, 4u), brow4, _dot);   \
        _dot = fma(subgroupShuffle(arow, 5u), brow5, _dot);   \
        _dot = fma(subgroupShuffle(arow, 6u), brow6, _dot);   \
        _dot = fma(subgroupShuffle(arow, 7u), brow7, _dot);   \
        _dot = fma(subgroupShuffle(arow, 8u), brow8, _dot);   \
        _dot = fma(subgroupShuffle(arow, 9u), brow9, _dot);   \
        _dot = fma(subgroupShuffle(arow, 10u), browa, _dot);  \
        _dot = fma(subgroupShuffle(arow, 11u), browb, _dot);  \
        _dot = fma(subgroupShuffle(arow, 12u), browc, _dot);  \
        _dot = fma(subgroupShuffle(arow, 13u), browd, _dot);  \
        _dot = fma(subgroupShuffle(arow, 14u), browe, _dot);  \
        _dot = fma(subgroupShuffle(arow, 15u), browf, _dot);

        MM_DOT_PRODUCT(0x0, dot[0]);
        MM_DOT_PRODUCT(0x1, dot[1]);
        MM_DOT_PRODUCT(0x2, dot[2]);
        MM_DOT_PRODUCT(0x3, dot[3]);
        MM_DOT_PRODUCT(0x4, dot[4]);
        MM_DOT_PRODUCT(0x5, dot[5]);
        MM_DOT_PRODUCT(0x6, dot[6]);
        MM_DOT_PRODUCT(0x7, dot[7]);
        MM_DOT_PRODUCT(0x8, dot[8]);
        MM_DOT_PRODUCT(0x9, dot[9]);
        MM_DOT_PRODUCT(0xa, dot[10]);
        MM_DOT_PRODUCT(0xb, dot[11]);
        MM_DOT_PRODUCT(0xc, dot[12]);
        MM_DOT_PRODUCT(0xd, dot[13]);
        MM_DOT_PRODUCT(0xe, dot[14]);
        MM_DOT_PRODUCT(0xf, dot[15]);
#undef MM_DOT_PRODUCT

        colA += (TILE_K / VEC_SIZE);
        w += (TILE_K / VEC_SIZE);
      } while (w < width0);

      mm_write(globalRow, globalCol, dot[0]);
      mm_write(globalRow + 1, globalCol, dot[1]);
      mm_write(globalRow + 2, globalCol, dot[2]);
      mm_write(globalRow + 3, globalCol, dot[3]);
      mm_write(globalRow + 4, globalCol, dot[4]);
      mm_write(globalRow + 5, globalCol, dot[5]);
      mm_write(globalRow + 6, globalCol, dot[6]);
      mm_write(globalRow + 7, globalCol, dot[7]);
      mm_write(globalRow + 8, globalCol, dot[8]);
      mm_write(globalRow + 9, globalCol, dot[9]);
      mm_write(globalRow + 10, globalCol, dot[10]);
      mm_write(globalRow + 11, globalCol, dot[11]);
      mm_write(globalRow + 12, globalCol, dot[12]);
      mm_write(globalRow + 13, globalCol, dot[13]);
      mm_write(globalRow + 14, globalCol, dot[14]);
      mm_write(globalRow + 15, globalCol, dot[15]);
    }
  `;
}

export class MatMulSIMD16Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 1, 1];
  extensions = ['GL_KHR_shader_subgroup_shuffle'];

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, transposeA = false, transposeB = false) {
    const dimInner = transposeA ? aShape[1] : aShape[2];
    const dimBOuter = outputShape[2];
    const bShape = transposeB ? [outputShape[0], dimBOuter, dimInner] :
                                [outputShape[0], dimInner, dimBOuter];
    this.outputShape = outputShape;
    const tileAOuter = this.workGroupSize[1] * 16;
    const tileBOuter = this.workGroupSize[0] * 1;
    const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    util.assert(
        tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0,
        () =>
            'tileInner must be multiple of workgroupsize.x and workgroupsize.y');
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, aShape.slice(1));
    let sampleA;
    if (transposeA === false) {
      sampleA = fitA ?
          `A[row * dimInner + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[row * dimInner + col] : 0`;
    } else {
      sampleA = fitA ?
          `A[col * dimAOuter + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[col * dimAOuter + row] : 0`;
    }

    const fitB = tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1));
    let sampleB;
    if (transposeB === false) {
      sampleB = fitB ?
          `B[row * dimBOuter + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[row * dimBOuter + col] : 0`;
    } else {
      sampleB = fitB ?
          `B[col * dimInner + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[col * dimInner + row] : 0`;
    }

    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [1, 16, 1]);
    this.userCode = `
      int dimAOuter = ${transposeA === true ? `aShape[2]` : `aShape[1]`};
      int dimInner = ${transposeA === true ? `aShape[1]` : `aShape[2]`};
      int dimBOuter = ${transposeB === true ? `bShape[1]` : `bShape[2]`};

      ${makeMatMulPackedSource([
      1, 16, 1
    ])}
      float mm_readA(int row, int col) {
        return ${sampleA};
      }

      float mm_readB(int row, int col) {
        return ${sampleB};
      }

      void mm_write(int row, int col, float value) {
        if (coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimBOuter)))
        {
          setOutput(row * dimBOuter + col, value);
        }
      }

      void main() {
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
    this.shaderKey = `matmulSIMD${this.workPerThread}${fitA}${fitB}${
        transposeA}${transposeB}`;
  }
}
