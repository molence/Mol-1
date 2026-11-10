# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gated input projection of triangle multiplication as one CuTe DSL kernel (Blackwell, BF16).

    ab[r, m] = sigmoid(G[m, r]) * P[m, r] * mask[m],   [G | P] = X W^T

with W's rows interleaved in blocks of 128 ([g(128); p(128); g; p; ...], see interleave_128),
written as [2C, M] (positions contiguous), the layout the triangular einsum reads.

Persistent CTAs, one 128-position tile at a time: the tile of X (128 x C) is loaded once into
shared memory (one stage per 64-wide K block, each released after its last use) and the W
blocks of 256 rows stream past it; each 256-row block is one tcgen05 GEMM into one of two
256-column TMEM slots, so the epilogue of one block overlaps the GEMM of the next. Warps 0-15
are the epilogue (four per TMEM lane quadrant, a quarter of the gate/projection columns each,
one 16-bit store per output: 32 consecutive positions per warp instruction), warp 16 issues
the MMAs, warp 17 the TMA loads.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

BLOCK = 128  # gate / projection rows per W block


class TriMulProjectionSm100:
    def __init__(self, c: int, bk: int = 64, ec: int = 8, epi_warps: int = 16):
        self.c = c
        self.bm = 128
        self.bn = 2 * BLOCK
        self.n_items = (2 * c) // BLOCK
        assert (2 * c) % BLOCK == 0 and c % bk == 0
        self.bk = bk
        self.k_tiles = c // bk
        self.ec = ec
        self.acc_dtype = Float32
        self.ab_dtype = cutlass.BFloat16
        self.slots = 2
        self.n_epi_warps = epi_warps
        self.ws = epi_warps // 4
        self.mma_warp = epi_warps
        self.tma_warp = epi_warps + 1
        self.threads = 32 * (epi_warps + 2)

    def _setup(self):
        k_major = cute.nvgpu.OperandMajorMode.K
        self.mma = utils.sm100.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, k_major, k_major, self.acc_dtype, tcgen05.CtaGroup.ONE, (self.bm, self.bn))
        self.mma_tiler = (self.bm, self.bn, self.bk)
        a_one = utils.sm100.make_smem_layout_a(self.mma, self.mma_tiler, self.ab_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(self.mma, self.mma_tiler, self.ab_dtype, 1)
        self.a_bytes = cute.size_in_bytes(self.ab_dtype, a_one)
        self.b_bytes = cute.size_in_bytes(self.ab_dtype, b_one)
        cap = utils.get_smem_capacity_in_bytes()
        self.num_b_stage = (cap - 2048 - self.k_tiles * self.a_bytes) // self.b_bytes
        assert self.num_b_stage >= 2
        self.a_layout = utils.sm100.make_smem_layout_a(self.mma, self.mma_tiler, self.ab_dtype, self.k_tiles)
        self.b_layout = utils.sm100.make_smem_layout_b(self.mma, self.mma_tiler, self.ab_dtype, self.num_b_stage)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mMask: cute.Tensor, mOut: cute.Tensor,
                 num_ctas: Int32, stream):
        """mX: (M, C, 1) K-major; mW: (4C, C, 1); mMask: (M,); mOut: (M, 2C), positions contiguous."""
        self._setup()
        op = cpasync.CopyBulkTensorTileG2SOp()
        cl = (1, 1, 1, 1)
        tma_a, tma_t_a = cute.nvgpu.make_tiled_tma_atom_A(
            op, mX, cute.slice_(self.a_layout, (None, None, None, 0)), self.mma_tiler, self.mma, cl)
        tma_b, tma_t_b = cute.nvgpu.make_tiled_tma_atom_B(
            op, mW, cute.slice_(self.b_layout, (None, None, None, 0)), self.mma_tiler, self.mma, cl)
        self.kernel(self.mma, tma_a, tma_t_a, tma_b, tma_t_b, mMask, mOut, self.a_layout, self.b_layout,
                    ).launch(grid=[num_ctas, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mma: cute.TiledMma, tma_a: cute.CopyAtom, mA: cute.Tensor, tma_b: cute.CopyAtom,
               mB: cute.Tensor, mMask: cute.Tensor, mOut: cute.Tensor,
               a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        M = mMask.shape[0]
        num_tiles = (M + self.bm - 1) // self.bm

        if warp_idx == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.k_tiles * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.slots * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        cl_vmnk = cute.make_layout((1, 1, 1, 1))
        # one stage per K block of the resident X tile
        a_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=self.k_tiles,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.a_bytes, cta_layout_vmnk=cl_vmnk, defer_sync=True)
        b_producer, b_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(), num_stages=self.num_b_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.b_bytes, cta_layout_vmnk=cl_vmnk, defer_sync=True).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=self.slots,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.n_epi_warps),
            cta_layout_vmnk=cl_vmnk, defer_sync=True)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * (self.n_epi_warps + 1))
        epi_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.n_epi_warps)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0, is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr)
        pipeline_init_arrive(cluster_shape_mn=cl_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(self.ab_dtype, a_layout.outer, 1024, a_layout.inner)
        sB = smem.allocate_tensor(self.ab_dtype, b_layout.outer, 1024, b_layout.inner)
        gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        thr = mma.get_slice(0)
        one = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(
            tma_a, 0, one, cute.group_modes(sA, 0, 3), cute.group_modes(thr.partition_A(gA), 0, 3))
        tBsB, tBgB = cpasync.tma_partition(
            tma_b, 0, one, cute.group_modes(sB, 0, 3), cute.group_modes(thr.partition_B(gB), 0, 3))
        tCrA = mma.make_fragment_A(sA)
        tCrB = mma.make_fragment_B(sB)
        tAcc_fake = mma.make_fragment_C(mma.partition_shape_C(self.mma_tiler[:2]))

        pipeline_init_wait(cluster_shape_mn=cl_vmnk)

        # ---------------- TMA producer ----------------
        if warp_idx == self.tma_warp:
            a_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.k_tiles)
            tile = bid
            while tile < num_tiles:
                for i in cutlass.range(self.n_items, unroll=1):
                    b_producer.reset()
                    peek = b_producer.try_acquire()
                    for k in cutlass.range(0, self.k_tiles, 1, unroll=1):
                        if i == 0:
                            a_pipe.producer_acquire(a_state)
                            cute.copy(tma_a, tAgA[(None, tile, k, 0)], tAsA[(None, a_state.index)],
                                      tma_bar_ptr=a_pipe.producer_get_barrier(a_state))
                            a_state.advance()
                        h = b_producer.acquire_and_advance(peek)
                        cute.copy(tma_b, tBgB[(None, i, h.count, 0)], tBsB[(None, h.index)], tma_bar_ptr=h.barrier)
                        peek = cutlass.Boolean(1)
                        if h.count + 1 < self.k_tiles:
                            peek = b_producer.try_acquire()
                tile += nctas
            b_producer.tail()
            a_pipe.producer_tail(a_state)

        # ---------------- MMA ----------------
        if warp_idx == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.slots)
            a_phase = Int32(0)
            tile = bid
            while tile < num_tiles:
                for i in cutlass.range(self.n_items, unroll=1):
                    acc = cute.make_tensor(tmem_ptr + acc_state.index * self.bn, tAcc_fake.layout)
                    b_consumer.reset()
                    peek = b_consumer.try_wait()
                    acc_pipeline.producer_acquire(acc_state)
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range(self.k_tiles, unroll=1):
                        a_st = pipeline.PipelineState(self.k_tiles, Int32(0), k, a_phase)
                        if i == 0:
                            a_pipe.consumer_wait(a_st)
                        h = b_consumer.wait_and_advance(peek)
                        for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                            cute.gemm(mma, acc, tCrA[(None, None, kb, k)], tCrB[(None, None, kb, h.index)], acc)
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                        h.release()
                        if i == self.n_items - 1:
                            a_pipe.consumer_release(a_st)  # X block k no longer needed by this tile
                        peek = cutlass.Boolean(1)
                        if h.count + 1 < self.k_tiles:
                            peek = b_consumer.try_wait()
                    acc_pipeline.producer_commit(acc_state)
                    acc_state.advance()
                a_phase = a_phase ^ 1
                tile += nctas
            acc_pipeline.producer_tail(acc_state)

        # ---------------- Epilogue ----------------
        if warp_idx < self.n_epi_warps:
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            hsel = warp_idx // 4  # this warp's slice of the block's gate columns
            acc_c = cute.flat_divide(
                cute.make_tensor(tmem_ptr, tAcc_fake.layout)[((None, None), 0, 0)], (self.bm, self.ec))
            tiled_t2r = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.ec)), self.acc_dtype),
                acc_c[(None, None, 0, 0)])
            thr_t2r = tiled_t2r.get_slice(tidx % 128)
            cId = thr_t2r.partition_D(cute.make_identity_tensor((self.bm, self.ec)))
            row_in_tile = cId[0][0]
            nck = (BLOCK // self.ws) // self.ec  # chunks per warp per block
            rG = cute.make_rmem_tensor(cId.shape, self.acc_dtype)
            rP = cute.make_rmem_tensor(cId.shape, self.acc_dtype)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.slots)
            tile = bid
            while tile < num_tiles:
                row = tile * self.bm + row_in_tile
                row_ok = row < M
                row_c = cutlass.min(row, M - 1)
                mk = mMask[row_c].to(Float32)
                for i in cutlass.range(self.n_items, unroll=1):
                    acc_pipeline.consumer_wait(acc_state)
                    base = tmem_ptr + acc_state.index * self.bn
                    for ci in cutlass.range_constexpr(nck):
                        cg = hsel * nck + ci          # gate chunk (columns 0..127)
                        cp = cg + BLOCK // self.ec    # projection chunk (columns 128..255)
                        cute.copy(tiled_t2r, thr_t2r.partition_S(cute.make_tensor(base, acc_c.layout)[(None, None, 0, cg)]), rG)
                        cute.copy(tiled_t2r, thr_t2r.partition_S(cute.make_tensor(base, acc_c.layout)[(None, None, 0, cp)]), rP)
                        cute.arch.fence_view_async_tmem_load()
                        if const_expr(ci + 1 == nck):
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_state)
                        ch0 = i * BLOCK + cg * self.ec
                        for e in cutlass.range(0, self.ec, 2, unroll_full=True):
                            h0, h1 = cute.arch.mul_packed_f32x2((0.5, 0.5), (rG[e], rG[e + 1]))
                            t0 = cute.math.tanh(h0, fastmath=True)
                            t1 = cute.math.tanh(h1, fastmath=True)
                            s0, s1 = cute.arch.fma_packed_f32x2((0.5, 0.5), (t0, t1), (0.5, 0.5))  # sigmoid(G)
                            q0, q1 = cute.arch.mul_packed_f32x2((s0, s1), (rP[e] * mk, rP[e + 1] * mk))
                            if row_ok:
                                mOut[(row_c, ch0 + e)] = q0.to(self.ab_dtype)
                                mOut[(row_c, ch0 + e + 1)] = q1.to(self.ab_dtype)
                    acc_state.advance()
                tile += nctas
            epi_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
