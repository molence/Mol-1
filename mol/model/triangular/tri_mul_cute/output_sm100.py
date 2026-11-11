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

"""Output side of triangle multiplication as one CuTe DSL kernel (Blackwell, BF16).

    out[m, o] = sigmoid(G[m, o]) * (rstd[m] * (P[m, o] - mean[m] * u[o]) + v[o]) (+ res[m, o])
    P = T^T W1^T   with T [C, M] the triangular einsum output (positions contiguous),
                   W1 = W_p * gamma (the output LayerNorm's weight folded into the projection)
    G = X W2^T     with X [M, C] the normalized input and W2 the output gate weight
    u = rowsum(W1), v = W_p beta, mean/rstd: per-position LayerNorm statistics of T

so LayerNorm(T)^T W_p^T is never materialized and T is read in its einsum layout (an MN-major
operand), without the transpose.

Persistent CTA pairs (cluster of 2, tcgen05 cta_group::2): one 256-position tile per pair,
split in NS output-channel subtiles of BN = C / NS columns; each subtile runs G and then P as
two GEMMs streaming only their own A operand, each CTA loading its 128 positions and half of
the weight rows. Warps 0-7 are the epilogue (two per TMEM lane quadrant), warp 8 issues
tcgen05.mma (leader CTA), warp 9 issues TMA loads, warps 10-11 only hand their registers to
the epilogue. TMEM holds 512 // BN accumulator slots, so the next subtile's GEMMs proceed
while the epilogue drains one. The epilogue prefetches the next subtile's residual (256-bit
loads) and statistics one subtile ahead and writes full 32-byte sectors.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


class TriMulOutSm100:
    def __init__(self, c: int, has_res: bool = True, ns: int = 2, bk: int = 64, ec: int = 16):
        self.c = c
        self.has_res = has_res
        self.ns = ns
        self.bn = c // ns
        self.ec = ec
        self.nch = self.bn // ec
        self.cta_m = 128
        self.bm = 256  # MMA M across the CTA pair
        self.bk = bk
        self.k_tiles = c // bk
        assert c % bk == 0 and c % ns == 0 and self.bn % ec == 0 and self.nch % 2 == 0 and self.bn % 16 == 0
        self.slots = min(512 // self.bn, 8)
        assert self.slots >= 2
        self.acc_dtype = Float32
        self.ab_dtype = cutlass.BFloat16
        self.n_epi_warps = 8
        self.mma_warp = 8
        self.tma_warp = 9
        self.threads = 32 * 12
        self.regs_epi = 224
        self.regs_other = 56

    def _setup(self):
        mn_major = cute.nvgpu.OperandMajorMode.MN
        k_major = cute.nvgpu.OperandMajorMode.K
        one = tcgen05.CtaGroup.TWO
        self.mma1 = utils.sm100.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, mn_major, k_major, self.acc_dtype, one, (self.bm, self.bn))
        self.mma2 = utils.sm100.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, k_major, k_major, self.acc_dtype, one, (self.bm, self.bn))
        self.mma_tiler = (self.bm, self.bn, self.bk)
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (self.mma1.thr_id.shape,))
        a_one = utils.sm100.make_smem_layout_a(self.mma1, self.mma_tiler, self.ab_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(self.mma1, self.mma_tiler, self.ab_dtype, 1)
        self.stage_bytes = cute.size_in_bytes(self.ab_dtype, a_one) + cute.size_in_bytes(self.ab_dtype, b_one)
        cap = utils.get_smem_capacity_in_bytes()
        self.num_ab_stage = (cap - 3072 - 8 * self.c) // self.stage_bytes
        self.a1_layout = utils.sm100.make_smem_layout_a(self.mma1, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.a2_layout = utils.sm100.make_smem_layout_a(self.mma2, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.b_layout = utils.sm100.make_smem_layout_b(self.mma1, self.mma_tiler, self.ab_dtype, self.num_ab_stage)

    @cute.jit
    def __call__(self, mT: cute.Tensor, mX: cute.Tensor, mW1: cute.Tensor, mW2: cute.Tensor,
                 mRes: cute.Tensor, mOut: cute.Tensor, mMean: cute.Tensor, mRstd: cute.Tensor,
                 mU: cute.Tensor, mV: cute.Tensor, num_ctas: Int32, stream):
        """mT: (M, C, 1) M-major; mX, mRes, mOut: (M, C, 1) C-major; mW1, mW2: (C, C, 1);
        mMean, mRstd: (M,); mU, mV: (C,). mRes is ignored without has_res."""
        self._setup()
        a_op = utils.sm100.cluster_shape_to_tma_atom_A((2, 1), self.mma1.thr_id)
        b_op = utils.sm100.cluster_shape_to_tma_atom_B((2, 1), self.mma1.thr_id)
        cl = self.cluster_layout_vmnk.shape
        tma_a1, tma_t_a1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mT, cute.slice_(self.a1_layout, (None, None, None, 0)), self.mma_tiler, self.mma1, cl)
        tma_a2, tma_t_a2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mX, cute.slice_(self.a2_layout, (None, None, None, 0)), self.mma_tiler, self.mma2, cl)
        tma_b1, tma_t_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mW1, cute.slice_(self.b_layout, (None, None, None, 0)), self.mma_tiler, self.mma1, cl)
        tma_b2, tma_t_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mW2, cute.slice_(self.b_layout, (None, None, None, 0)), self.mma_tiler, self.mma2, cl)
        self.kernel(
            self.mma1, self.mma2, tma_a1, tma_t_a1, tma_a2, tma_t_a2, tma_b1, tma_t_b1, tma_b2, tma_t_b2,
            mRes, mOut, mMean, mRstd, mU, mV, self.cluster_layout_vmnk, self.a1_layout, self.a2_layout, self.b_layout,
        ).launch(grid=[num_ctas, 1, 1], block=[self.threads, 1, 1], cluster=(2, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mma1: cute.TiledMma, mma2: cute.TiledMma,
               tma_a1: cute.CopyAtom, mA1: cute.Tensor, tma_a2: cute.CopyAtom, mA2: cute.Tensor,
               tma_b1: cute.CopyAtom, mB1: cute.Tensor, tma_b2: cute.CopyAtom, mB2: cute.Tensor,
               mRes: cute.Tensor, mOut: cute.Tensor, mMean: cute.Tensor, mRstd: cute.Tensor,
               mU: cute.Tensor, mV: cute.Tensor, cl_vmnk: cute.Layout,
               a1_layout: cute.ComposedLayout, a2_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()
        v = bid % 2  # CTA within the pair (0: leader, issues the MMAs)
        is_leader = v == 0
        pair = bid // 2
        npairs = nctas // 2
        coord_vmnk = cl_vmnk.get_flat_coord(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()))
        M = mMean.shape[0]
        num_tiles = (M + self.bm - 1) // self.bm

        if warp_idx == self.tma_warp:
            for a in (tma_a1, tma_a2, tma_b1, tma_b2):
                cpasync.prefetch_descriptor(a)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.slots * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.stage_bytes * 2,
            cta_layout_vmnk=cl_vmnk,
            defer_sync=True,
        ).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.slots,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.n_epi_warps * 2),
            cta_layout_vmnk=cl_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * (self.n_epi_warps + 1))
        epi_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.n_epi_warps)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0, is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=cl_vmnk, is_relaxed=True)

        # A stages hold either operand (T tiles MN-major, X tiles K-major): two views of one buffer
        sA1 = smem.allocate_tensor(self.ab_dtype, a1_layout.outer, 1024, a1_layout.inner)
        sA2 = cute.make_tensor(cute.recast_ptr(sA1.iterator, a2_layout.inner), a2_layout.outer)
        sB = smem.allocate_tensor(self.ab_dtype, b_layout.outer, 1024, b_layout.inner)
        sU = smem.allocate_tensor(Float32, cute.make_layout(self.c), 16)
        sV = smem.allocate_tensor(Float32, cute.make_layout(self.c), 16)

        # (bM, bK, RestM, RestK, RestL) / (bN, bK, RestN, RestK, RestL)
        gA1 = cute.local_tile(mA1, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gA2 = cute.local_tile(mA2, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB1 = cute.local_tile(mB1, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gB2 = cute.local_tile(mB2, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        thr1 = mma1.get_slice(v)
        thr2 = mma2.get_slice(v)
        a_mask = cpasync.create_tma_multicast_mask(cl_vmnk, coord_vmnk, mcast_mode=2)
        b_mask = cpasync.create_tma_multicast_mask(cl_vmnk, coord_vmnk, mcast_mode=1)
        a_cl = cute.make_layout(cute.slice_(cl_vmnk, (0, 0, None, 0)).shape)
        b_cl = cute.make_layout(cute.slice_(cl_vmnk, (0, None, 0, 0)).shape)
        tAsA1, tAgA1 = cpasync.tma_partition(
            tma_a1, coord_vmnk[2], a_cl, cute.group_modes(sA1, 0, 3), cute.group_modes(thr1.partition_A(gA1), 0, 3))
        tAsA2, tAgA2 = cpasync.tma_partition(
            tma_a2, coord_vmnk[2], a_cl, cute.group_modes(sA2, 0, 3), cute.group_modes(thr2.partition_A(gA2), 0, 3))
        tBsB1, tBgB1 = cpasync.tma_partition(
            tma_b1, coord_vmnk[1], b_cl, cute.group_modes(sB, 0, 3), cute.group_modes(thr1.partition_B(gB1), 0, 3))
        tBsB2, tBgB2 = cpasync.tma_partition(
            tma_b2, coord_vmnk[1], b_cl, cute.group_modes(sB, 0, 3), cute.group_modes(thr2.partition_B(gB2), 0, 3))
        tCrA1 = mma1.make_fragment_A(sA1)
        tCrA2 = mma2.make_fragment_A(sA2)
        tCrB = mma1.make_fragment_B(sB)
        tAcc_fake = mma1.make_fragment_C(mma1.partition_shape_C(self.mma_tiler[:2]))

        pipeline_init_wait(cluster_shape_mn=cl_vmnk)

        if warp_idx >= self.mma_warp:
            cute.arch.setmaxregister_decrease(self.regs_other)
        else:
            cute.arch.setmaxregister_increase(self.regs_epi)

        # ---------------- TMA producer ----------------
        if warp_idx == self.tma_warp:
            tile = pair
            while tile < num_tiles:
                for j in cutlass.range(self.ns, unroll=1):
                    for g in cutlass.range_constexpr(2):  # 0: G = X W2^T, 1: P = T W1^T
                        ab_producer.reset()
                        peek = ab_producer.try_acquire()
                        for k in cutlass.range(0, self.k_tiles, 1, unroll=1):
                            h = ab_producer.acquire_and_advance(peek)
                            if const_expr(g == 0):
                                cute.copy(tma_a2, tAgA2[(None, tile, h.count, 0)], tAsA2[(None, h.index)], tma_bar_ptr=h.barrier, mcast_mask=a_mask)
                                cute.copy(tma_b2, tBgB2[(None, j, h.count, 0)], tBsB2[(None, h.index)], tma_bar_ptr=h.barrier, mcast_mask=b_mask)
                            else:
                                cute.copy(tma_a1, tAgA1[(None, tile, h.count, 0)], tAsA1[(None, h.index)], tma_bar_ptr=h.barrier, mcast_mask=a_mask)
                                cute.copy(tma_b1, tBgB1[(None, j, h.count, 0)], tBsB1[(None, h.index)], tma_bar_ptr=h.barrier, mcast_mask=b_mask)
                            peek = cutlass.Boolean(1)
                            if h.count + 1 < self.k_tiles:
                                peek = ab_producer.try_acquire()
                tile += npairs
            ab_producer.tail()

        # ---------------- MMA ----------------
        if warp_idx == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.slots)
            tile = pair
            while tile < num_tiles:
                for j in cutlass.range(self.ns, unroll=1):
                    for g in cutlass.range_constexpr(2):
                        mma = mma2 if const_expr(g == 0) else mma1
                        tCrA = tCrA2 if const_expr(g == 0) else tCrA1
                        acc = cute.make_tensor(tmem_ptr + acc_state.index * self.bn, tAcc_fake.layout)
                        ab_consumer.reset()
                        peek = cutlass.Boolean(1)
                        if is_leader:
                            peek = ab_consumer.try_wait()
                            acc_pipeline.producer_acquire(acc_state)
                        mma.set(tcgen05.Field.ACCUMULATE, False)
                        for k in cutlass.range(self.k_tiles, unroll=1):
                            if is_leader:
                                h = ab_consumer.wait_and_advance(peek)
                                for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                                    crd = (None, None, kb, h.index)
                                    cute.gemm(mma, acc, tCrA[crd], tCrB[crd], acc)
                                    mma.set(tcgen05.Field.ACCUMULATE, True)
                                h.release()
                                peek = cutlass.Boolean(1)
                                if h.count + 1 < self.k_tiles:
                                    peek = ab_consumer.try_wait()
                        if is_leader:
                            acc_pipeline.producer_commit(acc_state)
                        acc_state.advance()
                tile += npairs
            acc_pipeline.producer_tail(acc_state)

        # ---------------- Epilogue ----------------
        if warp_idx < self.n_epi_warps:
            tmem.allocate(512)
            for i in cutlass.range(tidx, self.c, 32 * self.n_epi_warps, unroll=1):
                sU[i] = mU[i]
                sV[i] = mV[i]
            epi_barrier.arrive_and_wait()
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            hsel = warp_idx // 4  # this warp takes column chunks hsel, hsel + 2, ... of its lane quadrant
            acc_c = cute.flat_divide(
                cute.make_tensor(tmem_ptr, tAcc_fake.layout)[((None, None), 0, 0)], (self.cta_m, self.ec))
            tiled_t2r = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.ec)), self.acc_dtype),
                acc_c[(None, None, 0, 0)])
            thr_t2r = tiled_t2r.get_slice(tidx % 128)
            cId = thr_t2r.partition_D(cute.make_identity_tensor((self.cta_m, self.ec)))
            row_in_tile = v * self.cta_m + cId[0][0]  # this CTA's half of the pair's 256 positions
            n_mine = self.nch // 2
            pk = self.ec // 2  # residual kept as packed bf16 pairs until used
            vbits = self.ec * 16
            ld_atom = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), Int32, num_bits_per_copy=vbits,
                                          l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)
            st_atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), self.ab_dtype, num_bits_per_copy=vbits,
                                          l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)
            mRes32 = cute.recast_tensor(mRes, Int32)
            rres = cute.make_rmem_tensor((pk, n_mine), Int32)
            rnxt = cute.make_rmem_tensor((pk, n_mine), Int32)
            snxt = cute.make_rmem_tensor((2,), Float32)
            rG = cute.make_rmem_tensor(cId.shape, self.acc_dtype)
            rP = cute.make_rmem_tensor(cId.shape, self.acc_dtype)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.slots)
            # first work item's residual and statistics
            r0 = cutlass.min(pair * self.bm + row_in_tile, M - 1)
            snxt[0] = mMean[r0]
            snxt[1] = mRstd[r0]
            if const_expr(self.has_res):
                for ci in cutlass.range_constexpr(n_mine):
                    cute.copy(ld_atom, cute.local_tile(mRes32[(r0, None, 0)], (pk,), (hsel + 2 * ci,)), rnxt[(None, ci)])
            tile = pair
            while tile < num_tiles:
                row = tile * self.bm + row_in_tile
                row_ok = row < M
                row_c = cutlass.min(row, M - 1)
                for j in cutlass.range(self.ns, unroll=1):
                    if const_expr(self.has_res):
                        rres.store(rnxt.load())
                    mean = snxt[0]
                    rstd = snxt[1]
                    nrm = -mean * rstd
                    # next work item's residual and statistics (rows clamped past the end)
                    ntile = tile + ((j + 1) // self.ns) * npairs
                    nj = (j + 1) % self.ns
                    rn = cutlass.min(ntile * self.bm + row_in_tile, M - 1)
                    snxt[0] = mMean[rn]
                    snxt[1] = mRstd[rn]
                    if const_expr(self.has_res):
                        for ci in cutlass.range_constexpr(n_mine):
                            cute.copy(ld_atom, cute.local_tile(mRes32[(rn, None, 0)], (pk,), (nj * self.nch + hsel + 2 * ci,)),
                                      rnxt[(None, ci)])
                    acc_pipeline.consumer_wait(acc_state)
                    g_state = acc_state.clone()
                    acc_state.advance()
                    acc_pipeline.consumer_wait(acc_state)
                    p_state = acc_state.clone()
                    acc_state.advance()
                    gbase = tmem_ptr + g_state.index * self.bn
                    pbase = tmem_ptr + p_state.index * self.bn
                    for ci in cutlass.range_constexpr(n_mine):
                        c = hsel + 2 * ci
                        cute.copy(tiled_t2r, thr_t2r.partition_S(cute.make_tensor(gbase, acc_c.layout)[(None, None, 0, c)]), rG)
                        cute.copy(tiled_t2r, thr_t2r.partition_S(cute.make_tensor(pbase, acc_c.layout)[(None, None, 0, c)]), rP)
                        cute.arch.fence_view_async_tmem_load()
                        if const_expr(ci + 1 == n_mine):
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(g_state)
                                acc_pipeline.consumer_release(p_state)
                        cblk = j * self.nch + c
                        ru = cute.make_rmem_tensor((self.ec,), Float32)
                        rv = cute.make_rmem_tensor((self.ec,), Float32)
                        cute.autovec_copy(cute.local_tile(sU, (self.ec,), (cblk,)), ru)
                        cute.autovec_copy(cute.local_tile(sV, (self.ec,), (cblk,)), rv)
                        resf = cute.make_rmem_tensor((self.ec,), Float32)
                        if const_expr(self.has_res):
                            resf.store(cute.recast_tensor(rres[(None, ci)], self.ab_dtype).load().to(Float32))
                        else:
                            resf.fill(0.0)
                        ro = cute.make_rmem_tensor((self.ec,), Float32)
                        for i in cutlass.range(0, self.ec, 2, unroll_full=True):
                            w0, w1 = cute.arch.fma_packed_f32x2((nrm, nrm), (ru[i], ru[i + 1]), (rv[i], rv[i + 1]))
                            p0, p1 = cute.arch.fma_packed_f32x2((rstd, rstd), (rP[i], rP[i + 1]), (w0, w1))
                            h0, h1 = cute.arch.mul_packed_f32x2((0.5, 0.5), (rG[i], rG[i + 1]))
                            t0 = cute.math.tanh(h0, fastmath=True)
                            t1 = cute.math.tanh(h1, fastmath=True)
                            s0, s1 = cute.arch.fma_packed_f32x2((0.5, 0.5), (t0, t1), (0.5, 0.5))  # sigmoid(G)
                            ro[i], ro[i + 1] = cute.arch.fma_packed_f32x2((s0, s1), (p0, p1), (resf[i], resf[i + 1]))
                        rob = cute.make_rmem_tensor((self.ec,), self.ab_dtype)
                        rob.store(ro.load().to(self.ab_dtype))
                        if row_ok:
                            cute.copy(st_atom, rob, cute.local_tile(mOut[(row_c, None, 0)], (self.ec,), (cblk,)))
                tile += npairs
            epi_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
