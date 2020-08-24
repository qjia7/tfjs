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

import * as tf from '@tensorflow/tfjs-core';
import {describeWebGPU} from './test_util';

describeWebGPU('Ops resnet', () => {
  jasmine.DEFAULT_TIMEOUT_INTERVAL = 200000;
  // Performs `trials` trials, of `reps` repetitions each. At the end of each
  // trial, endTrial() is run (and included in the benchmark time). This
  // allows the cost of endTrial() to be amortized across the many iterations.
  // This is needed in particular because WebGPU readbacks are asynchronous
  // and therefore always incur latency. (Plus, in Chrome right now, readbacks
  // are very inefficient, making the problem way worse.) Readbacks could be
  // avoided by using fences, but we don't have a common abstraction over
  // WebGL and WebGPU fences at the moment.
  async function time(
      doRep: (r: number) => tf.Tensor[] | tf.Tensor,
      endTrial?: () => Promise<void>, disposeAfterEachTrial = false,
      trials = 50, reps = 50) {
    const times = [];

    let toDispose: tf.Tensor[] = [];
    const dispose = () => {
      for (const t of toDispose) {
        t.dispose();
      }
      toDispose = [];
    };

    const trial = async () => {
      let result;
      for (let r = 0; r < reps; ++r) {
        result = doRep(r);

        toDispose = toDispose.concat(Array.isArray(result) ? result : [result]);
      }

      if (endTrial != null) {
        await endTrial();
      } else {
        await (Array.isArray(result) ? result[0] : result).data();
      }
    };

    // Warm-up. Specifically, this pre-allocates enough memory for an entire
    // trial, ensuring that no allocations happen when timing a trial (if the
    // backend reuses allocations).
    await trial();
    dispose();

    for (let t = 0; t < trials; ++t) {
      const start = tf.util.now();
      await trial();
      times.push(tf.util.now() - start);
      if (disposeAfterEachTrial) {
        dispose();
      }
    }

    const mean = times.reduce((a, b) => a + b, 0) / trials;
    //const min = Math.min(...times);
    const fmt = (n: number) => n.toFixed(3);
    console.log(`${fmt(mean / reps)}`);
    //console.log(`Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
    //console.log(`Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
  }
  const nRep = 50;
  const nTrail = 50;

  it('pad', async () => {
    const doTest = async (xShape: Array<number>, pShape: Array<[number, number]>) => {
      const arr = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arr.map((x) => tf.randomNormal(xShape));
      const b:Array<[number, number]> = pShape;
      const c = 0;
      await time(
        (r) => {
          res[r] = tf.pad(a[r], b, c)
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep);
      a.forEach(t => t.dispose());
    };
    await doTest([1, 15, 15, 256], [[0, 0], [1, 1], [1, 1], [0, 0]]);
    await doTest([1, 29, 29, 128], [[0, 0], [1, 1], [1, 1], [0, 0]]);
    await doTest([1, 57, 57, 64], [[0, 0], [1, 1], [1, 1], [0, 0]]);
    await doTest([1, 113, 113, 64], [[0, 0], [1, 1], [1, 1], [0, 0]]);
    await doTest([1, 225, 225, 3], [[0, 0], [3, 3], [3, 3], [0, 0]]);
    await doTest([500, 600, 3], [[50, 50], [0, 0], [0, 0]]);
  });

  it('conv2d', async () => {
    const doTest = async (
      xShape: [number, number, number, number], fShape: [number, number, number, number],
      stride: [number, number], pad: 'valid'|'same'|number, format: 'NHWC'|'NCHW') => {
      const arrX = new Array(nRep).fill(0);
      const arrF = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const x = arrX.map((x) => tf.randomNormal<tf.Rank.R4>(xShape));
      const f = arrF.map((x) => tf.randomNormal<tf.Rank.R4>(fShape));
      await time(
        (r) => {
          res[r] = tf.conv2d(x[r], f[r], stride, pad, format);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep);
      x.forEach(t => t.dispose());
      f.forEach(t => t.dispose());
    };

    await doTest([1,8,8,256],[1,1,256,1024],[1,1],'same', 'NHWC');
    await doTest([1,8,8,512],[1,1,512,2048],[1,1],'same', 'NHWC');
    await doTest([1,8,8,1024],[1,1,1024,2048],[1,1],'same', 'NHWC');
    await doTest([1,8,8,1024],[1,1,1024,512],[1,1],'same', 'NHWC');
    await doTest([1,8,8,2048],[1,1,2048,17],[1,1],'same', 'NHWC');
    await doTest([1,8,8,2048],[1,1,2048,32],[1,1],'same', 'NHWC');
    await doTest([1,8,8,2048],[1,1,2048,34],[1,1],'same', 'NHWC');
    await doTest([1,8,8,2048],[1,1,2048,512],[1,1],'same', 'NHWC');
    await doTest([1,15,15,128],[1,1,128,512],[1,1],'same', 'NHWC');
    await doTest([1,15,15,256],[1,1,256,1024],[1,1],'same', 'NHWC');
    await doTest([1,15,15,512],[1,1,512,256],[1,1],'same', 'NHWC');
    await doTest([1,15,15,512],[1,1,512,1024],[1,1],'same', 'NHWC');
    await doTest([1,15,15,1024],[1,1,1024,256],[1,1],'same', 'NHWC');
    await doTest([1,29,29,64],[1,1,64,256],[1,1],'same', 'NHWC');
    await doTest([1,29,29,128],[1,1,128,512],[1,1],'same', 'NHWC');
    await doTest([1,29,29,256],[1,1,256,128],[1,1],'same', 'NHWC');
    await doTest([1,29,29,256],[1,1,256,512],[1,1],'same', 'NHWC');
    await doTest([1,29,29,512],[1,1,512,128],[1,1],'same', 'NHWC');
    await doTest([1,57,57,64],[1,1,64,64],[1,1],'same', 'NHWC');
    await doTest([1,57,57,64],[1,1,64,256],[1,1],'same', 'NHWC');
    await doTest([1,57,57,256],[1,1,256,64],[1,1],'same', 'NHWC');

    await doTest([1,8,8,512],[3,3,512,512],[1,1],'same', 'NHWC');
    await doTest([1,15,15,256],[3,3,256,256],[1,1],'same', 'NHWC');
    await doTest([1,17,17,256],[3,3,256,256],[2,2],'valid', 'NHWC');
    await doTest([1,29,29,128],[3,3,128,128],[1,1],'same', 'NHWC');
    await doTest([1,31,31,128],[3,3,128,128],[2,2],'valid', 'NHWC');
    await doTest([1,57,57,64],[3,3,64,64],[1,1],'same', 'NHWC');
    await doTest([1,59,59,64],[3,3,64,64],[2,2],'valid', 'NHWC');
    await doTest([1,231,231,3],[7,7,3,64],[2,2],'valid', 'NHWC');
  });

  it('relu', async () => {
    const doTest = async (xShape: Array<number>) => {
      const arr = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arr.map((x) => tf.randomNormal(xShape));
      await time(
        (r) => {
          res[r] = tf.relu(a[r]);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep);
      a.forEach(t => t.dispose());
    };
    await doTest([1,15,15,128]);
    await doTest([1,15,15,256]);
    await doTest([1,15,15,512]);
    await doTest([1,15,15,1024]);
    await doTest([1,29,29,64]);
    await doTest([1,29,29,128]);
    await doTest([1,29,29,256]);
    await doTest([1,29,29,512]);
    await doTest([1,57,57,64]);
    await doTest([1,57,57,256]);
    await doTest([1,8,8,256]);
    await doTest([1,8,8,512]);
    await doTest([1,8,8,1024]);
    await doTest([1,8,8,2048]);
    await doTest([1,113,113,64]);
  });

  it('add', async () => {
    const doTest = async (aShape: Array<number>, bShape: Array<number>) => {
      const arrA = new Array(nRep).fill(0);
      const arrB = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const a = arrA.map((x) => tf.randomNormal(aShape));
      const b = arrB.map((x) => tf.randomNormal(bShape));
      await time(
        (r) => {
          res[r] = tf.add(a[r], b[r]);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep);
      a.forEach(t => t.dispose());
      b.forEach(t => t.dispose());
    };
    await doTest([1,8,8,17],[17]);
    await doTest([1,8,8,32],[32]);
    await doTest([1,8,8,34],[34]);
    await doTest([1,8,8,256],[256]);
    await doTest([1,8,8,512],[512]);
    await doTest([1,8,8,1024],[1,8,8,1024]);
    await doTest([1,8,8,1024],[1024]);
    await doTest([1,8,8,2048],[1,8,8,2048]);
    await doTest([1,8,8,2048],[2048]);
    await doTest([1,15,15,128],[128]);
    await doTest([1,15,15,256],[256]);
    await doTest([1,15,15,512],[1,15,15,512]);
    await doTest([1,15,15,512],[512]);
    await doTest([1,15,15,1024],[1,15,15,1024]);
    await doTest([1,15,15,1024],[1024]);
    await doTest([1,29,29,64],[64]);
    await doTest([1,29,29,128],[128]);
    await doTest([1,29,29,256],[1,29,29,256]);
    await doTest([1,29,29,256],[256]);
    await doTest([1,29,29,512],[1,29,29,512]);
    await doTest([1,29,29,512],[512]);
    await doTest([1,57,57,64],[64]);
    await doTest([1,57,57,256],[1,57,57,256]);
    await doTest([1,57,57,256],[256]);
    await doTest([1,113,13,64],[64]);
    await doTest([225,225,3],[3]);
  });

  it('maxPool', async () => {
    const doTest = async (
      xShape: [number, number, number, number], filter: [number, number],
      stride: [number, number], pad: 'valid'|'same'|number) => {
      const arrX = new Array(nRep).fill(0);
      const res = new Array(nRep);
      const x = arrX.map((x) => tf.randomNormal<tf.Rank.R4>(xShape));
      await time(
        (r) => {
          res[r] = tf.maxPool(x[r], filter, stride, pad);
          return [];
        },
        async () => {
          await res[res.length - 1].data();
          for (const t of res) {
            t.dispose();
          }
        },
        false, nTrail, nRep);
      x.forEach(t => t.dispose());
    };
    await doTest([1,15,15,1024], [1, 1], [2, 2], 'same');
    await doTest([1,29,29,512], [1, 1], [2, 2], 'same');
    await doTest([1,57,57,256], [1, 1], [2, 2], 'same');
    await doTest([1,115,115,64], [3, 3], [2, 2], 'valid');
  });
});
