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

Persistent CTA pairs (cluster of 2, tcgen05 cta_group::2): one 256-position tile per pair,
each CTA keeping its 128 positions of X (128 x C) in shared memory (one stage per 64-wide K
block, double buffered across tiles when it fits) while the 256-row blocks of W stream past,
each CTA loading half of every block. Each block is one 256 x 256 x C MMA into one of two
256-column TMEM slots, so the epilogue of one block overlaps the MMA of the next. Halving the
W traffic per SM is what lets the MMAs run near their peak (2.05 PF with an empty epilogue).
Warps 0-7 are the epilogue in two groups of four (one per TMEM lane quadrant); each group
stages 16-channel chunks in shared memory, positions contiguous, and stores them with TMA.
Warp 8 issues the MMAs (leader CTA), warp 9 the TMA loads.
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
    def __init__(self, c: int, bk: int = 64, ec: int = 16, epi_warps: int = 8, o_stages: int = 2):
        self.c = c
        self.cta_m = 128
        self.bm = 256          # MMA M across the CTA pair
        self.bn = 2 * BLOCK    # W rows per block
        self.n_items = (2 * c) // BLOCK
        assert (2 * c) % BLOCK == 0 and c % bk == 0
        self.bk = bk
        self.k_tiles = c // bk
        self.ec = ec
        self.acc_dtype = Float32
        self.ab_dtype = cutlass.BFloat16
        self.slots = 2
        self.n_epi_warps = epi_warps
        self.n_groups = epi_warps // 4
        assert (BLOCK // ec) % self.n_groups == 0
        self.o_stages = o_stages
        self.mma_warp = epi_warps
        self.tma_warp = epi_warps + 1
        self.threads = 32 * (epi_warps + 2)

    def _setup(self):
        k_major = cute.nvgpu.OperandMajorMode.K
        self.mma = utils.sm100.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, k_major, k_major, self.acc_dtype, tcgen05.CtaGroup.TWO, (self.bm, self.bn))
        self.mma_tiler = (self.bm, self.bn, self.bk)
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (self.mma.thr_id.shape,))
        a_one = utils.sm100.make_smem_layout_a(self.mma, self.mma_tiler, self.ab_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(self.mma, self.mma_tiler, self.ab_dtype, 1)
        self.a_bytes = cute.size_in_bytes(self.ab_dtype, a_one)
        self.b_bytes = cute.size_in_bytes(self.ab_dtype, b_one)
        cap = utils.get_smem_capacity_in_bytes() - 2048 - self.n_groups * self.o_stages * self.cta_m * self.ec * 2
        self.a_bufs = 2 if cap - 2 * self.k_tiles * self.a_bytes >= 4 * self.b_bytes else 1
        self.a_stages = self.a_bufs * self.k_tiles
        self.num_b_stage = (cap - self.a_stages * self.a_bytes) // self.b_bytes
        assert self.num_b_stage >= 2
        self.a_layout = utils.sm100.make_smem_layout_a(self.mma, self.mma_tiler, self.ab_dtype, self.a_stages)
        self.b_layout = utils.sm100.make_smem_layout_b(self.mma, self.mma_tiler, self.ab_dtype, self.num_b_stage)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mMask: cute.Tensor, mOut: cute.Tensor,
                 num_ctas: Int32, stream):
        """mX: (M, C, 1) K-major; mW: (4C, C, 1); mMask: (M,); mOut: (M, 2C), positions contiguous.
        num_ctas must be even."""
        self._setup()
        a_op = utils.sm100.cluster_shape_to_tma_atom_A((2, 1), self.mma.thr_id)
        b_op = utils.sm100.cluster_shape_to_tma_atom_B((2, 1), self.mma.thr_id)
        tma_a, tma_t_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mX, cute.slice_(self.a_layout, (None, None, None, 0)), self.mma_tiler, self.mma,
            self.cluster_layout_vmnk.shape)
        tma_b, tma_t_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mW, cute.slice_(self.b_layout, (None, None, None, 0)), self.mma_tiler, self.mma,
            self.cluster_layout_vmnk.shape)
        o_one = cute.make_layout((self.cta_m, self.ec), stride=(1, self.cta_m))
        tma_o, tma_t_o = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mOut, o_one, (self.cta_m, self.ec))
        self.kernel(self.mma, tma_a, tma_t_a, tma_b, tma_t_b, tma_o, tma_t_o, mMask, self.cluster_layout_vmnk,
                    self.a_layout, self.b_layout,
                    ).launch(grid=[num_ctas, 1, 1], block=[self.threads, 1, 1], cluster=(2, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mma: cute.TiledMma, tma_a: cute.CopyAtom, mA: cute.Tensor, tma_b: cute.CopyAtom,
               mB: cute.Tensor, tma_o: cute.CopyAtom, mO: cute.Tensor, mMask: cute.Tensor, cl_vmnk: cute.Layout,
               a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        v = bid % 2  # CTA within the pair (0: leader, issues the MMAs)
        is_leader = v == 0
        pair = bid // 2
        npairs = nctas // 2
        coord_vmnk = cl_vmnk.get_flat_coord(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()))
        M = mMask.shape[0]
        num_tiles = (M + self.bm - 1) // self.bm

        if warp_idx == self.tma_warp:
            for a in (tma_a, tma_b, tma_o):
                cpasync.prefetch_descriptor(a)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.a_stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.slots * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        a_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=self.a_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.a_bytes * 2, cta_layout_vmnk=cl_vmnk, defer_sync=True)
        b_producer, b_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(), num_stages=self.num_b_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.b_bytes * 2, cta_layout_vmnk=cl_vmnk, defer_sync=True).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=self.slots,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.n_epi_warps * 2),
            cta_layout_vmnk=cl_vmnk, defer_sync=True)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * (self.n_epi_warps + 1))
        epi_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.n_epi_warps)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0, is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr)
        pipeline_init_arrive(cluster_shape_mn=cl_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(self.ab_dtype, a_layout.outer, 1024, a_layout.inner)
        sB = smem.allocate_tensor(self.ab_dtype, b_layout.outer, 1024, b_layout.inner)
        # output staging: per epilogue group, o_stages chunks of [EC channels][128 positions]
        sO = smem.allocate_tensor(
            self.ab_dtype,
            cute.make_layout((self.cta_m, self.ec, self.o_stages * self.n_groups),
                             stride=(1, self.cta_m, self.cta_m * self.ec)), 128)
        a_mask = cpasync.create_tma_multicast_mask(cl_vmnk, coord_vmnk, mcast_mode=2)
        b_mask = cpasync.create_tma_multicast_mask(cl_vmnk, coord_vmnk, mcast_mode=1)
        gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        thr = mma.get_slice(v)
        tAsA, tAgA = cpasync.tma_partition(
            tma_a, coord_vmnk[2], cute.make_layout(cute.slice_(cl_vmnk, (0, 0, None, 0)).shape),
            cute.group_modes(sA, 0, 3), cute.group_modes(thr.partition_A(gA), 0, 3))
        tBsB, tBgB = cpasync.tma_partition(
            tma_b, coord_vmnk[1], cute.make_layout(cute.slice_(cl_vmnk, (0, None, 0, 0)).shape),
            cute.group_modes(sB, 0, 3), cute.group_modes(thr.partition_B(gB), 0, 3))
        tOsO, tOgO = cpasync.tma_partition(
            tma_o, 0, cute.make_layout(1), cute.group_modes(sO, 0, 2),
            cute.group_modes(cute.local_tile(mO, (self.cta_m, self.ec), (None, None)), 0, 2))
        tCrA = mma.make_fragment_A(sA)
        tCrB = mma.make_fragment_B(sB)
        tAcc_fake = mma.make_fragment_C(mma.partition_shape_C(self.mma_tiler[:2]))

        pipeline_init_wait(cluster_shape_mn=cl_vmnk)

        # ---------------- TMA producer ----------------
        if warp_idx == self.tma_warp:
            a_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.a_stages)
            tile = pair
            while tile < num_tiles:
                for i in cutlass.range(self.n_items, unroll=1):
                    b_producer.reset()
                    peek = b_producer.try_acquire()
                    for k in cutlass.range(0, self.k_tiles, 1, unroll=1):
                        if i == 0:
                            a_pipe.producer_acquire(a_state)
                            cute.copy(tma_a, tAgA[(None, tile, k, 0)], tAsA[(None, a_state.index)],
                                      tma_bar_ptr=a_pipe.producer_get_barrier(a_state), mcast_mask=a_mask)
                            a_state.advance()
                        h = b_producer.acquire_and_advance(peek)
                        cute.copy(tma_b, tBgB[(None, i, h.count, 0)], tBsB[(None, h.index)], tma_bar_ptr=h.barrier,
                                  mcast_mask=b_mask)
                        peek = cutlass.Boolean(1)
                        if h.count + 1 < self.k_tiles:
                            peek = b_producer.try_acquire()
                tile += npairs
            b_producer.tail()
            a_pipe.producer_tail(a_state)

        # ---------------- MMA (issued by the leader CTA of the pair) ----------------
        if warp_idx == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.slots)
            it = Int32(0)
            tile = pair
            while tile < num_tiles:
                abase = (it % self.a_bufs) * self.k_tiles
                aphase = (it // self.a_bufs) % 2
                for i in cutlass.range(self.n_items, unroll=1):
                    acc = cute.make_tensor(tmem_ptr + acc_state.index * self.bn, tAcc_fake.layout)
                    b_consumer.reset()
                    peek = cutlass.Boolean(1)
                    if is_leader:
                        peek = b_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_state)
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range(self.k_tiles, unroll=1):
                        if is_leader:
                            a_st = pipeline.PipelineState(self.a_stages, Int32(0), abase + k, aphase)
                            if i == 0:
                                a_pipe.consumer_wait(a_st)
                            h = b_consumer.wait_and_advance(peek)
                            for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                                cute.gemm(mma, acc, tCrA[(None, None, kb, abase + k)], tCrB[(None, None, kb, h.index)], acc)
                                mma.set(tcgen05.Field.ACCUMULATE, True)
                            h.release()
                            if i == self.n_items - 1:
                                a_pipe.consumer_release(a_st)  # X block k no longer needed by this tile
                            peek = cutlass.Boolean(1)
                            if h.count + 1 < self.k_tiles:
                                peek = b_consumer.try_wait()
                    if is_leader:
                        acc_pipeline.producer_commit(acc_state)
                    acc_state.advance()
                it += 1
                tile += npairs
            acc_pipeline.producer_tail(acc_state)

        # ---------------- Epilogue ----------------
        if warp_idx < self.n_epi_warps:
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            grp = warp_idx // 4  # epilogue group: four warps covering the 128 TMEM lanes
            grp_bar = pipeline.NamedBarrier(barrier_id=4 + grp, num_threads=128)
            is_issuer = (warp_idx % 4) == 0
            acc_c = cute.flat_divide(
                cute.make_tensor(tmem_ptr, tAcc_fake.layout)[((None, None), 0, 0)], (self.cta_m, self.ec))
            tiled_t2r = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.ec)), self.acc_dtype),
                acc_c[(None, None, 0, 0)])
            thr_t2r = tiled_t2r.get_slice(tidx % 128)
            cId = thr_t2r.partition_D(cute.make_identity_tensor((self.cta_m, self.ec)))
            lane_row = cId[0][0]
            nck = (BLOCK // self.ec) // self.n_groups  # chunks per group per block
            rG = cute.make_rmem_tensor(cId.shape, self.acc_dtype)
            rP = cute.make_rmem_tensor(cId.shape, self.acc_dtype)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.slots)
            ostage = Int32(0)
            tile = pair
            while tile < num_tiles:
                mt = tile * 2 + v  # this CTA's 128-position output tile
                mk = mMask[cutlass.min(mt * self.cta_m + lane_row, M - 1)].to(Float32)
                for i in cutlass.range(self.n_items, unroll=1):
                    acc_pipeline.consumer_wait(acc_state)
                    base = tmem_ptr + acc_state.index * self.bn
                    for ci in cutlass.range_constexpr(nck):
                        cg = grp * nck + ci            # gate chunk (columns 0..127)
                        cp = cg + BLOCK // self.ec     # projection chunk (columns 128..255)
                        cute.copy(tiled_t2r, thr_t2r.partition_S(cute.make_tensor(base, acc_c.layout)[(None, None, 0, cg)]), rG)
                        cute.copy(tiled_t2r, thr_t2r.partition_S(cute.make_tensor(base, acc_c.layout)[(None, None, 0, cp)]), rP)
                        cute.arch.fence_view_async_tmem_load()
                        if const_expr(ci + 1 == nck):
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_state)
                        # the staging chunk is free once the TMA store issued o_stages chunks ago has read it
                        if is_issuer:
                            cute.arch.cp_async_bulk_wait_group(self.o_stages - 1, read=True)
                        grp_bar.arrive_and_wait()
                        sidx = grp * self.o_stages + ostage
                        for e in cutlass.range(0, self.ec, 2, unroll_full=True):
                            h0, h1 = cute.arch.mul_packed_f32x2((0.5, 0.5), (rG[e], rG[e + 1]))
                            t0 = cute.math.tanh(h0, fastmath=True)
                            t1 = cute.math.tanh(h1, fastmath=True)
                            s0, s1 = cute.arch.fma_packed_f32x2((0.5, 0.5), (t0, t1), (0.5, 0.5))  # sigmoid(G)
                            q0, q1 = cute.arch.mul_packed_f32x2((s0, s1), (rP[e] * mk, rP[e + 1] * mk))
                            sO[(lane_row, e, sidx)] = q0.to(self.ab_dtype)
                            sO[(lane_row, e + 1, sidx)] = q1.to(self.ab_dtype)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        grp_bar.arrive_and_wait()
                        if is_issuer:
                            cute.copy(tma_o, tOsO[(None, sidx)], tOgO[(None, mt, i * (BLOCK // self.ec) + cg)])
                            cute.arch.cp_async_bulk_commit_group()
                        ostage = (ostage + 1) % self.o_stages
                    acc_state.advance()
                tile += npairs
            if is_issuer:
                cute.arch.cp_async_bulk_wait_group(0, read=False)
            epi_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
