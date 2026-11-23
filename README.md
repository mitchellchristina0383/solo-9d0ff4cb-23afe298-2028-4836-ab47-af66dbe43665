# GPU 分区改配调度模拟器

推理节点要腾出更多小分区跑轻量模型。本服务用 FastAPI 模拟一个可扩到数千张的
GPU 卡池，支持按规格领用/归还分区、值班员发起改配、模拟驱动异步回报，并严格
保证在跑任务不受影响、调度只按驱动确认的实际布局恢复。卡数在启动时通过环境变量
配置，**默认两张**。

## 一、模型与规则

### 资源模型

- N 张卡（默认 `gpu-0`、`gpu-1`；可配到数千张），每张卡只有两种布局：
  - `large`：一个大分区（`<gpu>-p0`，size=`large`）
  - `split`：两个小分区（`<gpu>-p0`、`<gpu>-p1`，size=`small`）
- 任务按规格精确领用：`large` 任务只能占大分区，`small` 任务只能占小分区；
  任务结束按 `task_id` 归还分区。
- 初始状态：全部卡均为 `large` 布局、`active`。
- 领用顺序固定：**按卡的初始化顺序、再按卡内分区编号，取第一个同规格空位**。

### 卡状态机

| 状态 | 含义 | 是否接新任务 |
| --- | --- | --- |
| `active` | 正常调度 | 是 |
| `draining` | 已发起改配，停接新任务，等在跑任务归还分区 | 否（在跑任务保留） |
| `reconfiguring` | 分区已全部归还，已交给模拟驱动改配，等回报 | 否 |
| `disabled` | 驱动回报失败、实际布局未知，停用等待本次批次补报 | 否 |

流转规则：

1. 值班员对 `active` 卡发起改配（`POST /reconfig`），生成新的改配批次号
   `generation` 并记录目标布局：
   - 卡空闲 → 直接进入 `reconfiguring`；
   - 卡上有在跑任务 → 进入 `draining`，任务照常跑完；最后一个分区归还时
     自动进入 `reconfiguring`。
2. **卡一停接（draining/reconfiguring/disabled），其空位立即退出候选**；
   `draining` 期间归还的分区也**不重回候选**（排空期不可领）。其他卡照常分配，
   所有卡都给不出分区时返回 `503`。
3. 模拟驱动通过 `POST /driver/report` 回报：
   - `success`：必须带 `actual_layout`。服务按驱动确认的实际布局重建分区并恢复
     `active`——**调度只认本次驱动确认的布局，不认申请时的目标布局**；
   - `failed`：实际布局未知，卡进入/保持 `disabled`（分区列表清空，避免按
     未确认的布局分配），等待同批次补报；补报 `success` 后才恢复。
4. **旧改配回报不能改变当前状态**：回报的 `generation` 与卡上当前批次不一致
   （更旧或未来的批次号）一律 `409` 拒收；批次已结束后的重复回报同样拒收。
5. 一张卡同时只允许一个进行中的改配批次，非 `active` 状态再次发起改配返回 `409`。

所有写操作在同一把异步锁内串行执行；卡状态与候选索引在同一临界区内一起变更，
领用、归还、改配并发交错时不会出现状态与候选集不一致。

## 二、大卡池的候选索引（对数复杂度）

每张卡最多两个分区，全部槽位构成固定宇宙：卡 `i` 的 p0/p1 分别占槽位
`2i`、`2i+1`，槽位顺序恰好等于「卡初始化顺序 + 卡内分区编号」。每种规格
（large/small）维护一棵 **0/1 线段树**（`app/service.py` 的 `_AvailabilityTree`）：

- 叶子为 1 ⇔ 该槽位是一张 `active` 卡上的同规格空闲分区（即可领空位）；
- `POST /tasks/acquire` 取树上**最左的 1**：`O(log N)`；无空位时根节点为 0，
  `O(1)` 判定 503，**不再扫遍整个卡池**；
- 单卡状态变化（停接、归还、驱动确认布局、停用）只改写该卡至多两个槽位，
  每次 `update` 沿一条根路径回放：`O(log N)`；
- 两棵树的后备数组各长 `2*next_pow2(2N) ≤ 4N`，整体额外空间 `O(N)`。

| 操作 | 旧实现 | 现实现 |
| --- | --- | --- |
| 找到空位 | 遍历全部卡与分区 `O(N)` | 线段树定位最左叶子 `O(log N)` |
| 无空位返回 503 | 遍历全部卡 `O(N)` | 根节点判空 `O(1)` |
| 单卡变更的候选维护 | 下次领用时全池重扫 | 至多两个槽位更新 `O(log N)` |
| 额外空间 | `O(N)` 分区对象 | 两棵线段树 `O(N)` |

## 三、目录结构

```
app/
  models.py    # 枚举与 Pydantic 请求/响应模型
  service.py   # 调度状态机 + 候选线段树（GPU/分区、领用归还、改配、驱动回报）
  main.py      # FastAPI 路由、GPU_COUNT/GPU_IDS 启动配置与异常映射
tests/
  test_scheduler.py   # 13 个端到端状态机测试（TestClient 打真实路由）
  test_large_pool.py  # 13 个大卡池测试：顺序、对数复杂度、并发一致性、启动配置
scripts/
  smoke.sh            # 真实 uvicorn + curl 冒烟（随 GPU_COUNT 适配 2 卡/大卡池）
  perf_check.py       # 大卡池查找/维护复杂度实测
Dockerfile
docker-compose.yml
requirements.txt / requirements-dev.txt
```

## 四、启动方式（Docker Compose）

```bash
# 默认两张卡
docker compose up --build

# 启动时设置模拟卡数（例如 3000 张）
GPU_COUNT=3000 docker compose up --build

# 也可显式指定卡 ID（与 GPU_COUNT 同时给出时以 GPU_IDS 为准）
GPU_IDS=gpu-a,gpu-b,gpu-c docker compose up --build
```

服务监听 http://localhost:8000 ，交互式文档 http://localhost:8000/docs ，
OpenAPI http://localhost:8000/openapi.json ，`GET /health` 返回卡数。
容器带 healthcheck。本机无 Docker 时也可直接用 Python 运行：

```bash
pip install -r requirements.txt
GPU_COUNT=3000 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 五、API 一览（均可在 /docs 操作）

| 方法/路径 | 说明 |
| --- | --- |
| `GET /gpus` | 全部卡的实时状态（状态、已确认布局、批次号、目标布局、分区占用） |
| `GET /gpus/{gpu_id}` | 单卡状态 |
| `POST /tasks/acquire` | 按 `size` 领用分区，可带 `task_id` 或由服务端生成 |
| `POST /tasks/release` | 任务结束归还分区；draining 卡借此自动推进到 reconfiguring |
| `POST /reconfig` | 值班员发起改配，返回 `generation`（驱动回报必须带同一批次号） |
| `POST /driver/report` | 模拟驱动回报成功（带 `actual_layout`）或失败 |
| `POST /admin/reset` | 重置为初始状态（测试用，候选集恢复初始大分区） |
| `GET /health` | 健康检查，含 `gpu_count` |

### /docs 手工演示序列

打开 http://localhost:8000/docs ，按下列请求依次执行（也可直接 `bash scripts/smoke.sh`）：

```bash
# 1. gpu-0 上跑一个大任务
curl -X POST localhost:8000/tasks/acquire -H 'Content-Type: application/json' \
  -d '{"size":"large","task_id":"job-a"}'
# 2. 发起改配：gpu-0 进入 draining，停接新任务（空位立即退出候选）
curl -X POST localhost:8000/reconfig -H 'Content-Type: application/json' \
  -d '{"gpu_id":"gpu-0","target_layout":"split"}'
# 3. 新任务只能落到 gpu-1（大卡池时继续按顺序落到 gpu-2、gpu-3 …）
curl -X POST localhost:8000/tasks/acquire -H 'Content-Type: application/json' \
  -d '{"size":"large","task_id":"job-b"}'
# 4. job-a 正常结束归还，gpu-0 自动交给驱动（reconfiguring）；排空期归还的分区不可领
curl -X POST localhost:8000/tasks/release -H 'Content-Type: application/json' \
  -d '{"task_id":"job-a"}'
# 5. 模拟驱动回报成功，确认实际布局为 split，随后即可领小分区
curl -X POST localhost:8000/driver/report -H 'Content-Type: application/json' \
  -d '{"gpu_id":"gpu-0","generation":1,"status":"success","actual_layout":"split"}'
# 失败演练：把 status 换成 failed（不带 actual_layout）→ disabled；
# 再用同一 generation 补报 success + actual_layout 才恢复。
```

## 六、实测结果

### 自动化测试（26 项，全部通过）

```
26 passed in 4.00s
```

- `test_scheduler.py`（13 项）：初始布局、领还分区、无匹配分区 503、空闲卡直接
  reconfiguring、**领用与改配交错**（停用卡停接/另一张照常/归还后自动交驱动/按
  确认布局恢复）、**失败后停用与同批次补报**、重复失败保持停用、**旧批次与未来
  批次回报拒收**、批次结束后重复回报拒收、success 缺 actual_layout 拒收、重复
  task_id、同布局改配拒绝、未知 GPU/任务 404、draining 在跑任务保留。
- `test_large_pool.py`（13 项，2048 卡）：初始候选严格按卡序/分区编号、领用恒取
  最左空位、split 恢复后 p0/p1 有序入候选、**无匹配 503 零下降且步数受 log 上界
  约束**、**单卡维护节点写入 ≤ 4·log2(2N)**、树后备空间 O(N)、停接即退出候选、
  排空归还不回树、失败停用至本批次有效 success 才恢复、重置恢复初始候选、
  **16 协程 × 40 轮领用/归还/改配并发后候选不变量成立**、`GPU_COUNT`/`GPU_IDS`
  启动配置（含默认两张）。

### 大卡池复杂度实测（`python3 scripts/perf_check.py`）

| 卡数 N | 服务 503(根判空) | 树下降步数/深度 | 树下降查找 | 树单点更新 | 单卡维护写入(上界) | 树数组长度 |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 19.4 µs | 9/9 | 4.1 µs | 5.3 µs | 30 (≤36) | 1024 |
| 512 | 20.7 µs | 10/10 | 4.5 µs | 7.6 µs | 33 (≤40) | 2048 |
| 1024 | 20.4 µs | 11/11 | 5.0 µs | 8.3 µs | 36 (≤44) | 4096 |
| 2048 | 20.3 µs | 12/12 | 5.4 µs | 9.0 µs | 39 (≤48) | 8192 |
| 4096 | 20.1 µs | 13/13 | 5.8 µs | 9.8 µs | 42 (≤52) | 16384 |
| 8192 | 20.4 µs | 14/14 | 6.2 µs | 10.4 µs | 45 (≤56) | 32768 |

结论：卡数扩大 32 倍（256→8192），服务层 503 耗时基本不变（主要是与 N 无关的
asyncio 锁开销）；纯树查找仅随 `log2(N)` 缓慢增长（约 1.5 倍，绝非线性扫描）；
完整下降步数恒等于树深度 `log2(next_pow2(2N))`，根判空 503 零下降；单卡维护
写入节点数实测低于 `4·log2(2N)` 上界；每棵树后备数组 ≤ 4N，两棵合计 O(N)。

### 真实进程冒烟（uvicorn + curl，`scripts/smoke.sh`）

分别对默认 2 卡与 `GPU_COUNT=3000` 的真实 uvicorn 进程跑通：

1. 交错场景：gpu-0 占用中发起改配 → `draining`；期间新大任务落 gpu-1，
   大卡池下第三个任务按顺序落 gpu-2（两卡池则返回 503）；job-a 归还后 gpu-0
   自动转 `reconfiguring`；驱动 success 确认 split 后两个 small 分区按 p0、p1
   顺序可领。
2. 失败补报：驱动 `failed` → `disabled`、`partitions=[]`、批次号保留；停用期间
   新任务由其他卡承接；同批次再次 failed 保持停用；同批次 success 补报带
   actual_layout 后恢复 `active` 并按实际布局重建候选。
3. 旧回报拒收：批次 #2 进行中拿批次 #1 回报、以及批次号 99 的未来回报，均
   `409` 且卡状态/已确认布局不变。

## 七、余项与边界说明

- 状态保存在进程内存中，重启即回到初始布局；`/admin/reset` 仅用于测试，
  生产化时需替换为持久化存储并加鉴权。
- 布局仍为题目固定的 `large`/`split` 两种、每卡至多两个分区；槽位宇宙按每卡
  两槽预留，新增布局类型时扩展 `_build_partitions` 与建表逻辑即可，索引复杂度
  结论不变。
- 未实现任务超时/抢占：draining 完全等待任务主动归还，符合“在跑任务照常结束”
  的要求。
- 当前环境未安装 Docker，Dockerfile 与 docker-compose.yml 已按 Compose 规范
  编写并通过 YAML 解析校验，实际容器启动需在有 Docker 的机器上执行
  `GPU_COUNT=<卡数> docker compose up --build`；全部功能已在本机 uvicorn 实测。
