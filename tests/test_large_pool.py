"""大卡池扩展测试：候选线段树的顺序、对数复杂度、并发一致性与启动配置。

覆盖需求：
* 卡池扩到数千张：领用以卡初始化顺序 + 卡内分区编号取第一个同规格空位；
* 无匹配空位（503）与单卡变更的候选维护最坏 O(log N)，额外空间 O(N)；
* 卡一停接空位退出候选；排空期间归还的分区不可领；
* 失败后继续停用，本批次有效 success 才按实际布局恢复候选；
* 领用/归还/改配并发下卡状态与候选集同步；
* 重置恢复初始候选；GPU_COUNT / GPU_IDS 启动配置生效。
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from app.models import GPUState, LayoutType, PartitionSize
from app.service import SchedulerService


def _candidate_ids(svc: SchedulerService, size: PartitionSize) -> list[str]:
    return [
        svc._gpu_list[slot >> 1].partitions[slot & 1].id
        for slot in svc.candidate_slots(size)
    ]


def _assert_invariants(svc: SchedulerService) -> None:
    """树内候选必须恰好等于 ACTIVE 卡的同规格空闲分区。"""
    for size in PartitionSize:
        expected: set[int] = set()
        for gpu in svc._gpu_list:
            if gpu.state is not GPUState.ACTIVE:
                continue
            for part in gpu.partitions:
                if part.size is size and part.occupied_by is None:
                    expected.add(part.slot)
        assert set(svc.candidate_slots(size)) == expected


@pytest.fixture()
def svc() -> SchedulerService:
    return SchedulerService(gpu_count=2048)


# ---------- 顺序与基础候选 ----------

def test_initial_candidates_are_all_large_partitions_in_order(svc: SchedulerService) -> None:
    large = _candidate_ids(svc, PartitionSize.LARGE)
    assert len(large) == 2048
    # 严格按卡初始化顺序，且每卡只暴露 p0（large 布局没有 p1）
    assert large == [f"gpu-{i}-p0" for i in range(2048)]
    assert svc.candidate_slots(PartitionSize.SMALL) == []
    _assert_invariants(svc)


async def test_acquire_always_picks_first_free_slot_in_global_order(svc: SchedulerService) -> None:
    # 占掉 gpu-0 后，下一个必须是 gpu-1-p0
    tid0, gpu0, part0 = await svc.acquire(PartitionSize.LARGE, "t0")
    assert (gpu0.id, part0.id) == ("gpu-0", "gpu-0-p0")
    tid1, gpu1, part1 = await svc.acquire(PartitionSize.LARGE, None)
    assert (gpu1.id, part1.id) == ("gpu-1", "gpu-1-p0")
    # 释放 gpu-0 后，最左空位回到 gpu-0-p0
    await svc.release("t0")
    tid2, gpu2, part2 = await svc.acquire(PartitionSize.LARGE, "t2")
    assert (gpu2.id, part2.id) == ("gpu-0", "gpu-0-p0")
    _assert_invariants(svc)


async def test_split_layout_recovers_two_small_slots_in_partition_order(svc: SchedulerService) -> None:
    await svc.request_reconfig("gpu-2", LayoutType.SPLIT)
    await svc.driver_report("gpu-2", 1, "success", LayoutType.SPLIT)
    small = _candidate_ids(svc, PartitionSize.SMALL)
    # 第一张 split 卡的两个分区按 p0、p1 顺序排在最前
    assert small[:2] == ["gpu-2-p0", "gpu-2-p1"]
    _assert_invariants(svc)


# ---------- 对数复杂度（含 503） ----------

async def test_no_match_503_is_logarithmic_and_does_not_scan_pool(svc: SchedulerService) -> None:
    small_tree = svc._trees[PartitionSize.SMALL]
    small_tree.descent_steps = 0
    # 初始全是 large 布局：small 全池无空位，根节点为 0 直接判定
    with pytest.raises(Exception) as exc:
        await svc.acquire(PartitionSize.SMALL, "no-such-size")
    assert exc.value.status_code == 503
    # 根节点为 0：零次下降、不触碰任何卡
    assert small_tree.descent_steps == 0
    assert small_tree.first_available() is None

    # 把全部 large 分区也占满后再申请：下降步数严格受 log2(槽位宇宙) 上界约束
    large_tree = svc._trees[PartitionSize.LARGE]
    for i in range(svc.gpu_count):
        await svc.acquire(PartitionSize.LARGE, f"fill-{i}")
    assert large_tree.first_available() is None
    large_tree.descent_steps = 0
    with pytest.raises(Exception) as exc:
        await svc.acquire(PartitionSize.LARGE, "overflow")
    assert exc.value.status_code == 503
    assert large_tree.descent_steps <= large_tree.depth  # <= log2(next_pow2(2N))
    _assert_invariants(svc)


async def test_single_card_candidate_maintenance_is_logarithmic(svc: SchedulerService) -> None:
    tree = svc._trees[PartitionSize.LARGE]
    depth = tree.depth

    # 单卡停接：至多两个槽位更新，每个只沿一条根路径写 O(depth) 个节点
    tree.nodes_written = 0
    await svc.request_reconfig("gpu-100", LayoutType.SPLIT)
    assert tree.nodes_written <= 2 * depth

    # 排空期间归还：不回树（零写入），分区保持不可领
    await svc.acquire(PartitionSize.LARGE, "drain-job")  # 落到 gpu-0
    await svc.request_reconfig("gpu-0", LayoutType.SPLIT)
    tree.nodes_written = 0
    await svc.release("drain-job")
    assert tree.nodes_written == 0
    g0 = svc._gpus["gpu-0"]
    assert g0.state is GPUState.RECONFIGURING
    assert g0.partitions[0].slot not in svc.candidate_slots(PartitionSize.LARGE)

    # 驱动成功按实际布局恢复：只写该卡自己的槽位
    tree.nodes_written = 0
    await svc.driver_report("gpu-100", 1, "success", LayoutType.SPLIT)
    assert tree.nodes_written <= 2 * depth
    _assert_invariants(svc)


def test_tree_backing_storage_is_linear(svc: SchedulerService) -> None:
    for tree in svc._trees.values():
        assert len(tree._tree) <= 4 * svc.gpu_count  # 2 * next_pow2(2N) = O(N)


# ---------- 停接 / 排空 / 失败停用 / 恢复 ----------

async def test_stop_accepting_removes_free_slots_immediately(svc: SchedulerService) -> None:
    await svc.request_reconfig("gpu-0", LayoutType.SPLIT)  # 空闲卡直接 reconfiguring
    assert svc.candidate_slots(PartitionSize.LARGE)[0] == 2  # gpu-1-p0 的槽位
    # 有占用的卡停接：空闲槽位同样立即退出
    await svc.acquire(PartitionSize.LARGE, "job-on-g1")
    await svc.request_reconfig("gpu-1", LayoutType.SPLIT)
    assert svc.candidate_slots(PartitionSize.LARGE)[0] == 4
    _assert_invariants(svc)


async def test_failed_report_keeps_disabled_until_valid_success(svc: SchedulerService) -> None:
    await svc.request_reconfig("gpu-3", LayoutType.SPLIT)
    await svc.driver_report("gpu-3", 1, "failed", None)
    g3 = svc._gpus["gpu-3"]
    assert g3.state is GPUState.DISABLED
    assert g3.partitions == []
    assert svc.candidate_slots(PartitionSize.LARGE)[0] == 0  # gpu-0 仍在候选
    assert 6 not in svc.candidate_slots(PartitionSize.LARGE)

    # 同批次再次 failed：继续停用
    await svc.driver_report("gpu-3", 1, "failed", None)
    assert g3.state is GPUState.DISABLED

    # 本批次有效 success：按驱动确认的实际布局恢复候选
    await svc.driver_report("gpu-3", 1, "success", LayoutType.SPLIT)
    assert g3.state is GPUState.ACTIVE
    assert _candidate_ids(svc, PartitionSize.SMALL)[:2] == ["gpu-3-p0", "gpu-3-p1"]
    _assert_invariants(svc)


async def test_reset_restores_initial_candidates(svc: SchedulerService) -> None:
    await svc.request_reconfig("gpu-0", LayoutType.SPLIT)
    await svc.driver_report("gpu-0", 1, "success", LayoutType.SPLIT)
    await svc.acquire(PartitionSize.LARGE, "job")
    await svc.reset()
    assert len(svc.candidate_slots(PartitionSize.LARGE)) == svc.gpu_count
    assert svc.candidate_slots(PartitionSize.SMALL) == []
    assert all(g.state is GPUState.ACTIVE for g in svc._gpu_list)
    assert all(g.generation == 0 for g in svc._gpu_list)
    _assert_invariants(svc)


# ---------- 并发一致性 ----------

async def test_concurrent_acquire_release_reconfig_keeps_state_and_candidates_consistent() -> None:
    svc = SchedulerService(gpu_count=128)

    async def worker(worker_id: int) -> None:
        for round_id in range(40):
            tag = f"w{worker_id}-r{round_id}"
            try:
                _, gpu, part = await svc.acquire(PartitionSize.LARGE, tag)
            except Exception:
                await asyncio.sleep(0)
                continue
            await asyncio.sleep(0)
            if worker_id == 0 and round_id % 5 == 0:
                # 与领用/归还交错发起改配并立即回报，验证候选同步维护
                target = (
                    LayoutType.SPLIT
                    if gpu.layout is LayoutType.LARGE
                    else LayoutType.LARGE
                )
                try:
                    gen_gpu = await svc.request_reconfig(gpu.id, target)
                    await svc.release(tag)
                    await svc.driver_report(
                        gpu.id, gen_gpu.generation, "success", target
                    )
                except Exception:
                    await svc.release(tag)
            else:
                await svc.release(tag)

    await asyncio.gather(*(worker(i) for i in range(16)))

    # 所有任务都已归还：ACTIVE 卡的空闲分区必须全部回到候选，无泄漏/幽灵槽位
    assert svc._tasks == {}
    _assert_invariants(svc)
    # 结束时不存在停在中间态的卡（每个改配批次都有终态回报）
    assert all(g.state is GPUState.ACTIVE for g in svc._gpu_list)


# ---------- 启动配置（子进程，避免污染本进程的 app 单例） ----------

def _run_in_subprocess(env: dict[str, str], code: str) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra_paths = [project_root]
    # 兼容无 venv 机器上把依赖装在项目内 .pylibs 的情况
    pylibs = os.path.join(project_root, ".pylibs")
    if os.path.isdir(pylibs):
        extra_paths.append(pylibs)
    full_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(
            extra_paths + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        ),
        **env,
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=full_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def test_gpu_count_env_var_sets_pool_size() -> None:
    out = _run_in_subprocess(
        {"GPU_COUNT": "3000"},
        "from app.main import service, _TOTAL_GPUS; "
        "from app.models import PartitionSize; "
        "print(_TOTAL_GPUS, len(service.candidate_slots(PartitionSize.LARGE)))",
    )
    assert out == "3000 3000"


def test_default_gpu_count_is_two() -> None:
    out = _run_in_subprocess(
        {"GPU_COUNT": "", "GPU_IDS": ""},
        "from app.main import _TOTAL_GPUS; print(_TOTAL_GPUS)",
    )
    assert out == "2"


def test_gpu_ids_env_var_overrides_count() -> None:
    out = _run_in_subprocess(
        {"GPU_COUNT": "999", "GPU_IDS": "card-a, card-b ,card-c"},
        "from app.main import service; "
        "print([g.id for g in service._gpu_list])",
    )
    assert out == "['card-a', 'card-b', 'card-c']"
