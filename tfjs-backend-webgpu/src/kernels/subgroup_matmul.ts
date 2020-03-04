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

// SIMD_16x2_1x8
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
      vec2 dot[16];
      for (int i = 0; i < 16; i++)
      {
        dot[i] = vec2(0.0, 0.0);
      }

      int globalRow = group_y * TILE_M + 16 * local_y;
      int colA = local_x * 2;
      int globalCol = local_x * 2 + (group_x * (TILE_N / VEC_SIZE));
      int rowB = 0;
      int w = 0;
      do
      {
        float brow0x = mm_readB(rowB, globalCol);
        float brow0y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow1x = mm_readB(rowB, globalCol);
        float brow1y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow2x = mm_readB(rowB, globalCol);
        float brow2y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow3x = mm_readB(rowB, globalCol);
        float brow3y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow4x = mm_readB(rowB, globalCol);
        float brow4y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow5x = mm_readB(rowB, globalCol);
        float brow5y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow6x = mm_readB(rowB, globalCol);
        float brow6y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow7x = mm_readB(rowB, globalCol);
        float brow7y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow8x = mm_readB(rowB, globalCol);
        float brow8y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float brow9x = mm_readB(rowB, globalCol);
        float brow9y = mm_readB(rowB, globalCol + 1);
        rowB++;
        float browax = mm_readB(rowB, globalCol);
        float broway = mm_readB(rowB, globalCol + 1);
        rowB++;
        float browbx = mm_readB(rowB, globalCol);
        float browby = mm_readB(rowB, globalCol + 1);
        rowB++;
        float browcx = mm_readB(rowB, globalCol);
        float browcy = mm_readB(rowB, globalCol + 1);
        rowB++;
        float browdx = mm_readB(rowB, globalCol);
        float browdy = mm_readB(rowB, globalCol + 1);
        rowB++;
        float browex = mm_readB(rowB, globalCol);
        float browey = mm_readB(rowB, globalCol + 1);
        rowB++;
        float browfx = mm_readB(rowB, globalCol);
        float browfy = mm_readB(rowB, globalCol + 1);
        rowB++;

        vec2 arow;
        for (int i = 0; i < 16; i++)
        {
          arow.x = mm_readA(globalRow + i, colA);
          arow.y = mm_readA(globalRow + i, colA + 1);
          dot[i].x += subgroupBroadcast(arow, 0u).x * brow0x;
          dot[i].x += subgroupBroadcast(arow, 0u).y * brow1x;
          dot[i].x += subgroupBroadcast(arow, 1u).x * brow2x;
          dot[i].x += subgroupBroadcast(arow, 1u).y * brow3x;
          dot[i].x += subgroupBroadcast(arow, 2u).x * brow4x;
          dot[i].x += subgroupBroadcast(arow, 2u).y * brow5x;
          dot[i].x += subgroupBroadcast(arow, 3u).x * brow6x;
          dot[i].x += subgroupBroadcast(arow, 3u).y * brow7x;
          dot[i].x += subgroupBroadcast(arow, 4u).x * brow8x;
          dot[i].x += subgroupBroadcast(arow, 4u).y * brow9x;
          dot[i].x += subgroupBroadcast(arow, 5u).x * browax;
          dot[i].x += subgroupBroadcast(arow, 5u).y * browbx;
          dot[i].x += subgroupBroadcast(arow, 6u).x * browcx;
          dot[i].x += subgroupBroadcast(arow, 6u).y * browdx;
          dot[i].x += subgroupBroadcast(arow, 7u).x * browex;
          dot[i].x += subgroupBroadcast(arow, 7u).y * browfx;

          dot[i].y += subgroupBroadcast(arow, 0u).x * brow0y;
          dot[i].y += subgroupBroadcast(arow, 0u).y * brow1y;
          dot[i].y += subgroupBroadcast(arow, 1u).x * brow2y;
          dot[i].y += subgroupBroadcast(arow, 1u).y * brow3y;
          dot[i].y += subgroupBroadcast(arow, 2u).x * brow4y;
          dot[i].y += subgroupBroadcast(arow, 2u).y * brow5y;
          dot[i].y += subgroupBroadcast(arow, 3u).x * brow6y;
          dot[i].y += subgroupBroadcast(arow, 3u).y * brow7y;
          dot[i].y += subgroupBroadcast(arow, 4u).x * brow8y;
          dot[i].y += subgroupBroadcast(arow, 4u).y * brow9y;
          dot[i].y += subgroupBroadcast(arow, 5u).x * broway;
          dot[i].y += subgroupBroadcast(arow, 5u).y * browby;
          dot[i].y += subgroupBroadcast(arow, 6u).x * browcy;
          dot[i].y += subgroupBroadcast(arow, 6u).y * browdy;
          dot[i].y += subgroupBroadcast(arow, 7u).x * browey;
          dot[i].y += subgroupBroadcast(arow, 7u).y * browfy;
        }

        colA += (TILE_K / VEC_SIZE);
        w += (TILE_K / VEC_SIZE);
      } while (w < width0);

      mm_write(globalRow, globalCol, dot[0].x);
      mm_write(globalRow, globalCol + 1, dot[0].y);
      mm_write(globalRow + 1, globalCol, dot[1].x);
      mm_write(globalRow + 1, globalCol + 1, dot[1].y);
      mm_write(globalRow + 2, globalCol, dot[2].x);
      mm_write(globalRow + 2, globalCol + 1, dot[2].y);
      mm_write(globalRow + 3, globalCol, dot[3].x);
      mm_write(globalRow + 3, globalCol + 1, dot[3].y);
      mm_write(globalRow + 4, globalCol, dot[4].x);
      mm_write(globalRow + 4, globalCol + 1, dot[4].y);
      mm_write(globalRow + 5, globalCol, dot[5].x);
      mm_write(globalRow + 5, globalCol + 1, dot[5].y);
      mm_write(globalRow + 6, globalCol, dot[6].x);
      mm_write(globalRow + 6, globalCol + 1, dot[6].y);
      mm_write(globalRow + 7, globalCol, dot[7].x);
      mm_write(globalRow + 7, globalCol + 1, dot[7].y);
      mm_write(globalRow + 8, globalCol, dot[8].x);
      mm_write(globalRow + 8, globalCol + 1, dot[8].y);
      mm_write(globalRow + 9, globalCol, dot[9].x);
      mm_write(globalRow + 9, globalCol + 1, dot[9].y);
      mm_write(globalRow + 10, globalCol, dot[10].x);
      mm_write(globalRow + 10, globalCol + 1, dot[10].y);
      mm_write(globalRow + 11, globalCol, dot[11].x);
      mm_write(globalRow + 11, globalCol + 1, dot[11].y);
      mm_write(globalRow + 12, globalCol, dot[12].x);
      mm_write(globalRow + 12, globalCol + 1, dot[12].y);
      mm_write(globalRow + 13, globalCol, dot[13].x);
      mm_write(globalRow + 13, globalCol + 1, dot[13].y);
      mm_write(globalRow + 14, globalCol, dot[14].x);
      mm_write(globalRow + 14, globalCol + 1, dot[14].y);
      mm_write(globalRow + 15, globalCol, dot[15].x);
      mm_write(globalRow + 15, globalCol + 1, dot[15].y);
    }
  `;
}

export class MatMulSIMDProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [8, 1, 1];
  extensions = ['GL_KHR_shader_subgroup_ballot'];

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, transposeA = false, transposeB = false) {
    const dimInner = transposeA ? aShape[1] : aShape[2];
    const dimBOuter = outputShape[2];
    const bShape = transposeB ? [outputShape[0], dimBOuter, dimInner] :
                                [outputShape[0], dimInner, dimBOuter];
    this.outputShape = outputShape;
    const tileAOuter = this.workGroupSize[1] * 16;
    const tileBOuter = this.workGroupSize[0] * 2;
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
        this.dispatchLayout, this.outputShape, this.workGroupSize, [2, 16, 1]);
    this.userCode = `
      int dimAOuter = ${transposeA === true ? `aShape[2]` : `aShape[1]`};
      int dimInner = ${transposeA === true ? `aShape[1]` : `aShape[2]`};
      int dimBOuter = ${transposeB === true ? `bShape[1]` : `bShape[2]`};

      ${makeMatMulPackedSource([
      2, 16, 1
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
