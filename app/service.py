"""核心调度状态机：N 张 GPU 的分区领用/归还、停用改配与驱动回报。

状态机（单张卡）：

    ACTIVE ──发起改配且卡上有占用──▶ DRAINING ──分区全部归还──▶ RECONFIGURING
      │                                                              │
      └──────────发起改配且卡已空闲──────────────────────────────▶ RECONFIGURING
                                                                     │
                              驱动 success（批次一致，带实际布局） ──┴──▶ ACTIVE
                              驱动 failed（批次一致） ─────────────────▶ DISABLED

    DISABLED ──本次批次补报 success（带实际布局）──▶ ACTIVE
             ──本次批次补报 failed ──────────────▶ 继续 DISABLED

可领空位的数据结构（卡池扩到数千张时不再扫全池）
--------------------------------------------------
每张卡最多两个分区，全部槽位构成固定宇宙：卡 ``i`` 的 p0/p1 分别占槽位
``2i`` / ``2i+1``，槽位顺序恰好等于「卡初始化顺序 + 卡内分区编号」。
每种规格（large/small）维护一棵 0/1 线段树：

* 叶子为 1  ⇔  该槽位是一张 ACTIVE 卡上的同规格空闲分区（即可领空位）；
* ``acquire`` 取树上最左的 1：O(log M)，无空位时根节点为 0，O(1) 判定 503；
* 单卡状态变化（停接、归还、驱动确认布局、停用）只改写该卡至多两个槽位，
  每次 ``update`` 沿一条根路径回放：O(log M)；
* 两棵树的后备数组各长 2*next_pow2(M)，整体额外空间 O(M)=O(N)。

关键规则：
* DRAINING / RECONFIGURING / DISABLED 的卡不参与新任务分配；其他卡照常。
* 卡一停接，空位立即退出候选；DRAINING 期间归还的分区也不重回候选。
* 调度只按驱动 success 回报确认的实际布局重建分区并恢复分配。
* 回报的 generation 与卡上当前批次不一致时一律拒收（旧回报不能改变当前状态）。
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .models import GPUState, LayoutType, PartitionSize


class SchedulerError(Exception):
    """可预期的调度错误，由 API 层映射为 4xx。"""

    status_code = 400

    def __init__(self, detail: str, status_code: Optional[int] = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(SchedulerError):
    status_code = 404


class ConflictError(SchedulerError):
    status_code = 409


@dataclass
class Partition:
    id: str
    size: PartitionSize
    slot: int
    occupied_by: Optional[str] = None


@dataclass
class GPU:
    id: str
    index: int
    state: GPUState = GPUState.ACTIVE
    layout: LayoutType = LayoutType.LARGE
    generation: int = 0
    pending_target: Optional[LayoutType] = None
    partitions: list[Partition] = field(default_factory=list)

    def is_idle(self) -> bool:
        return all(p.occupied_by is None for p in self.partitions)


def _build_partitions(gpu_index: int, gpu_id: str, layout: LayoutType) -> list[Partition]:
    """按驱动确认的布局构建分区，槽位与（卡序号、分区编号）固定对应。"""
    base = gpu_index * 2
    if layout is LayoutType.LARGE:
        return [Partition(id=f"{gpu_id}-p0", size=PartitionSize.LARGE, slot=base)]
    return [
        Partition(id=f"{gpu_id}-p0", size=PartitionSize.SMALL, slot=base),
        Partition(id=f"{gpu_id}-p1", size=PartitionSize.SMALL, slot=base + 1),
    ]


class _AvailabilityTree:
    """固定槽位宇宙上的 0/1 求和线段树，维护某一规格的可领空位。

    * ``update(pos, available)``：置叶子并沿一条根路径回放，O(log M)；
    * ``first_available()``：返回最左为 1 的槽位，O(log M)；整树为 0 时 O(1)；
    * 后备数组长度为 2 * next_pow2(M)，空间 O(M)。

    ``update_calls`` / ``nodes_written`` / ``descent_steps`` 仅为复杂度自测暴露。
    """

    def __init__(self, size: int) -> None:
        n = 1
        while n < size:
            n <<= 1
        self._n = n
        self._tree = [0] * (2 * n)
        self.update_calls = 0
        self.nodes_written = 0
        self.descent_steps = 0

    @classmethod
    def from_leaves(cls, flags: list[bool]) -> "_AvailabilityTree":
        """O(M) 批量建表（初始化/重置时使用）。"""
        tree = cls(len(flags))
        base = tree._n
        for pos, flag in enumerate(flags):
            tree._tree[base + pos] = 1 if flag else 0
        for i in range(base - 1, 0, -1):
            tree._tree[i] = tree._tree[i << 1] + tree._tree[i << 1 | 1]
        return tree

    @property
    def depth(self) -> int:
        """叶到根的层数（log2(next_pow2(M))），单次 update/下降的步数上界。"""
        return self._n.bit_length() - 1

    def update(self, pos: int, available: bool) -> None:
        self.update_calls += 1
        i = pos + self._n
        val = 1 if available else 0
        if self._tree[i] == val:
            return
        self._tree[i] = val
        self.nodes_written += 1
        i >>= 1
        while i:
            self._tree[i] = self._tree[i << 1] + self._tree[i << 1 | 1]
            self.nodes_written += 1
            i >>= 1

    def first_available(self) -> Optional[int]:
        """最左为 1 的槽位；全 0 返回 None（不遍历任何卡/分区）。"""
        if self._tree[1] == 0:
            return None
        i = 1
        while i < self._n:
            self.descent_steps += 1
            left = i << 1
            i = left if self._tree[left] else left | 1
        return i - self._n

    def leaf(self, pos: int) -> int:
        return self._tree[self._n + pos]


class SchedulerService:
    """内存态调度器。所有公开方法都在同一把异步锁内串行执行，

    卡状态与两棵候选线段树在同一临界区内一起变更，领用/归还/改配并发交错时
    不会出现状态与候选集不一致。
    """

    def __init__(
        self,
        gpu_count: int = 2,
        gpu_ids: Optional[Iterable[str]] = None,
    ) -> None:
        if gpu_ids is not None:
            ids = tuple(gpu_ids)
        else:
            if gpu_count < 1:
                raise ValueError("模拟卡数 gpu_count 必须 >= 1")
            ids = tuple(f"gpu-{i}" for i in range(gpu_count))
        self._lock = asyncio.Lock()
        self._slot_count = 2 * len(ids)
        self._gpu_list: list[GPU] = []
        self._gpus: dict[str, GPU] = {}
        for index, gid in enumerate(ids):
            gpu = GPU(
                id=gid,
                index=index,
                state=GPUState.ACTIVE,
                layout=LayoutType.LARGE,
                generation=0,
                pending_target=None,
                partitions=_build_partitions(index, gid, LayoutType.LARGE),
            )
            self._gpu_list.append(gpu)
            self._gpus[gid] = gpu
        # task_id -> (gpu_id, 全局槽位号)
        self._tasks: dict[str, tuple[str, int]] = {}
        self._trees = self._build_initial_trees()

    @property
    def gpu_count(self) -> int:
        return len(self._gpu_list)

    # ---------- 内部工具 ----------

    def _build_initial_trees(self) -> dict[PartitionSize, _AvailabilityTree]:
        """初始布局：每卡一个 large 分区占偶数槽位，small 树全空。"""
        large_flags = [(pos % 2 == 0) for pos in range(self._slot_count)]
        small_flags = [False] * self._slot_count
        return {
            PartitionSize.LARGE: _AvailabilityTree.from_leaves(large_flags),
            PartitionSize.SMALL: _AvailabilityTree.from_leaves(small_flags),
        }

    def _get_gpu(self, gpu_id: str) -> GPU:
        gpu = self._gpus.get(gpu_id)
        if gpu is None:
            raise NotFoundError(f"未知 GPU: {gpu_id}")
        return gpu

    def _mark(self, part: Partition, available: bool) -> None:
        self._trees[part.size].update(part.slot, available)

    def _remove_gpu_candidates(self, gpu: GPU) -> None:
        """卡停接：其空闲分区全部退出候选（占用中的槽位本就不在树内）。"""
        for part in gpu.partitions:
            self._mark(part, False)

    def _add_gpu_candidates(self, gpu: GPU) -> None:
        """按驱动确认的实际布局恢复候选：仅 ACTIVE 卡的空闲分区入树。"""
        if gpu.state is not GPUState.ACTIVE:
            return
        for part in gpu.partitions:
            self._mark(part, part.occupied_by is None)

    @staticmethod
    def _pump(gpu: GPU) -> None:
        """停用中的卡分区全部归还后，交给模拟驱动（进入 RECONFIGURING）。"""
        if gpu.state is GPUState.DRAINING and gpu.is_idle():
            gpu.state = GPUState.RECONFIGURING

    def _apply_confirmed_layout(self, gpu: GPU, layout: LayoutType) -> None:
        """驱动 success：只认回报里的实际布局重建分区并恢复候选。"""
        gpu.layout = layout
        gpu.partitions = _build_partitions(gpu.index, gpu.id, layout)
        gpu.state = GPUState.ACTIVE
        gpu.pending_target = None
        self._add_gpu_candidates(gpu)

    # ---------- 查询 ----------

    async def snapshot(self) -> list[GPU]:
        async with self._lock:
            return list(self._gpu_list)

    async def get_gpu(self, gpu_id: str) -> GPU:
        async with self._lock:
            return self._get_gpu(gpu_id)

    def candidate_slots(self, size: PartitionSize) -> list[int]:
        """导出某规格当前候选槽位（测试/自检用，O(N)，不在热路径）。"""
        tree = self._trees[size]
        return [pos for pos in range(self._slot_count) if tree.leaf(pos) == 1]

    # ---------- 任务领用 / 归还 ----------

    async def acquire(
        self,
        size: PartitionSize,
        task_id: Optional[str],
    ) -> tuple[str, GPU, Partition]:
        async with self._lock:
            if task_id is not None and task_id in self._tasks:
                raise ConflictError(f"任务 {task_id} 已持有分区，需先归还")
            tid = task_id or f"task-{uuid.uuid4().hex[:12]}"

            # 线段树直接定位最左空位：卡序 + 分区编号序，O(log N)；
            # 根节点为 0 即全池无同规格空位，O(1) 判定 503，不扫卡池。
            tree = self._trees[size]
            slot = tree.first_available()
            if slot is None:
                raise ConflictError(
                    f"当前没有空闲的 {size.value} 分区（停用中的卡不参与分配）",
                    status_code=503,
                )

            gpu = self._gpu_list[slot >> 1]
            part = gpu.partitions[slot & 1]
            # 不变量：树内候选必属于 ACTIVE 卡的同规格空闲分区
            assert gpu.state is GPUState.ACTIVE
            assert part.size is size and part.occupied_by is None
            part.occupied_by = tid
            tree.update(slot, False)
            self._tasks[tid] = (gpu.id, slot)
            return tid, gpu, part

    async def release(self, task_id: str) -> tuple[GPU, Partition]:
        async with self._lock:
            location = self._tasks.pop(task_id, None)
            if location is None:
                raise NotFoundError(f"未知任务或分区已归还: {task_id}")
            gpu_id, slot = location
            gpu = self._get_gpu(gpu_id)
            part = gpu.partitions[slot & 1]
            if part.occupied_by != task_id:
                # 理论上不会出现：任务表与分区占用保持一致
                raise ConflictError("任务占用记录与分区状态不一致")
            part.occupied_by = None

            if gpu.state is GPUState.ACTIVE:
                # 正常调度中的卡：归还分区重回候选，O(log N)
                self._mark(part, True)
            # DRAINING：排空期间归还的分区保持不可领，不回树；
            # 若这是最后一个分区，_pump 把卡推进 RECONFIGURING。
            self._pump(gpu)
            return gpu, part

    # ---------- 改配 ----------

    async def request_reconfig(self, gpu_id: str, target: LayoutType) -> GPU:
        async with self._lock:
            gpu = self._get_gpu(gpu_id)
            if gpu.state is not GPUState.ACTIVE:
                raise ConflictError(
                    f"{gpu_id} 处于 {gpu.state.value} 状态，已有进行中或待补报的改配批次 "
                    f"#{gpu.generation}，不能重复发起改配"
                )
            if target is gpu.layout:
                raise ConflictError(
                    f"目标布局 {target.value} 与当前已确认布局相同，无需改配"
                )
            gpu.generation += 1
            gpu.pending_target = target
            # 卡一停接，空位立即退出候选（至多两个槽位，O(log N)）
            self._remove_gpu_candidates(gpu)
            if gpu.is_idle():
                # 卡上没有在跑任务，立即交给模拟驱动
                gpu.state = GPUState.RECONFIGURING
            else:
                # 停接新任务，等分区全部归还
                gpu.state = GPUState.DRAINING
            return gpu

    # ---------- 模拟驱动回报 ----------

    async def driver_report(
        self,
        gpu_id: str,
        generation: int,
        status: str,
        actual_layout: Optional[LayoutType],
    ) -> GPU:
        async with self._lock:
            gpu = self._get_gpu(gpu_id)

            # 旧批次回报：无论成功失败都不能改变当前分配状态
            if generation != gpu.generation:
                raise ConflictError(
                    f"回报批次 #{generation} 与当前批次 #{gpu.generation} 不一致，"
                    "旧改配回报一律忽略"
                )

            if gpu.state is GPUState.RECONFIGURING:
                if status == "success":
                    if actual_layout is None:
                        raise ConflictError(
                            "success 回报必须携带驱动确认的 actual_layout"
                        )
                    # 调度只按驱动确认的实际布局重建分区并恢复分配
                    self._apply_confirmed_layout(gpu, actual_layout)
                    return gpu
                # failed：布局未知，停用等补报；该卡槽位此前已全部退出候选
                gpu.state = GPUState.DISABLED
                gpu.partitions = []
                return gpu

            if gpu.state is GPUState.DISABLED:
                if status == "success":
                    if actual_layout is None:
                        raise ConflictError(
                            "success 补报必须携带驱动确认的 actual_layout"
                        )
                    # 本次批次的有效成功回报：按实际布局恢复候选
                    self._apply_confirmed_layout(gpu, actual_layout)
                    return gpu
                # 补报仍然失败：布局依旧未知，继续停用等待下一次补报
                return gpu

            raise ConflictError(
                f"{gpu_id} 当前为 {gpu.state.value} 状态，没有等待驱动回报的改配"
            )

    # ---------- 测试辅助 ----------

    async def reset(self) -> None:
        """重置为初始状态：全部卡 large/ACTIVE，候选集恢复为初始大分区。"""
        async with self._lock:
            for gpu in self._gpu_list:
                gpu.state = GPUState.ACTIVE
                gpu.layout = LayoutType.LARGE
                gpu.generation = 0
                gpu.pending_target = None
                gpu.partitions = _build_partitions(gpu.index, gpu.id, LayoutType.LARGE)
            self._tasks.clear()
            self._trees = self._build_initial_trees()
