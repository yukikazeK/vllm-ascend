# 跨仓库源码映射与版本记录

## 1. 此分支的边界

本分支保留 fork 的 `main` 中已有功能，并基于 `experiments/gdn-workspace-correctness` 的 `48c7cff`，带入此前验证过的 GDN 索引、状态归位和 Prefill 工作空间补丁；另包含早期 C8 ND/Fused Gather 实验提交 `db671e7` 的 cherry-pick。新的长上下文实验入口位于 `examples/c8_long_context/`。早期 `examples/c8_nd_prefill/` 的 MTP0 入口仍独立保留，不能与新入口混用。

| 项目 | 实际范围 | 在分支中如何记录 |
| --- | --- | --- |
| 上游 vLLM 0.30.0 | 调度器、动态 MTP、GDN 状态块、混合 KV 逻辑分组、前缀保留边界 | 已有的统一 diff 补丁与源码/结果 SHA manifest，加上本实验的 KV 管理 import hook 模块 |
| vLLM-Ascend 0.30.0 | V2 worker／GDN／整图调用、C8 校准与物理 KV 布局、ND Attention、融合 gather、LSE 分段与 Q4 折叠 | `examples/dynamic_mtp/patches/` 及 `examples/c8_long_context/runtime/` |
| torch-npu、Ascend Triton、FLA NPU 26.9.1 | 调用既有 API，使用运行时 Triton JIT gather | 保留可复现 Python/Triton 源码；软件包中没有二进制或正式 CANN 模板修改 |
| CANN 9.2.1 | 使用原装 OPP 的 FusedInferAttentionScore、LSE 等运算接口，不修改安装目录与设备算子 | 启动器核验 OPP 版本；只允许已安装、实测使用的 FLA NPU 26.9.1 自带 vendor 目录自动注册，拒绝其他 `ASCEND_CUSTOM_OPP_PATH`（包括另启的原生 INT8 实验 OPP）；不提交自定义 OPP、算子 tiling、CANN 二进制或原生全 INT8 实验 |

本实验中执行的是 **INT8 KV 缓存 + 已安装 CANN 支持的 Attention/反量化** 与 FP32 尾段合并，不是 Q/K/V 全部原生 INT8 算术。此前另一研究线的 CANN host/device 代码、INT8 QK/PV、Vsum、P 量化、tiling 或自定义 OPP 仍未满足完整验证条件，未纳入本分支。实际部署本方案不需要执行那些实验，也不需要安装替代 CANN 包。

## 2. 上游 vLLM：现有动态 MTP 补丁

配套启动器复用 [dynamic_mtp/launch.py](../dynamic_mtp/launch.py)，并启用 `--gdn-k2-indexing --gdn-prefill-grouping`。它将统一 diff 应用到独立运行目录中的源码副本，检查目标文件 SHA-256，不修改原安装文件。文件和内容映射保持为原始可审计的 patch，不把整个上游 vLLM 仓库复制到 fork。

| 上游文件（相对 vLLM 包根目录） | 补丁文件（相对 `examples/dynamic_mtp/patches`） | 修改用途 |
| --- | --- | --- |
| `vllm/v1/engine/core.py` | `engine.patch` | 请求入场前预留状态、解析实际 MTP 策略并设置调度档位 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `lease_manager.patch` | 根据每请求动态 K 预留混合模型状态槽，处理状态搬移和缩档 |
| `vllm/v1/worker/gpu/block_table.py` | `lease_table.patch` | 配合每请求实际状态槽预留，补齐所需块表位置 |
| `vllm/v1/worker/gpu/model_runner.py` | `runner.patch` | 接收运行时选择、执行相应图及草稿张量处理 |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | `candidates.patch`，启用 GDN K2 后选择 `gdn_k2/candidates.patch` | 捕获多种验证宽度及对应 K2 专用目标图 |
| `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` | `base_ar.patch` | 适配运行时草稿宽度 |
| `vllm/v1/worker/mamba_utils.py` | `mamba_utils.patch` | GDN/Mamba 的状态同步、移动与缩档归位 |

源码摘要、结果摘要和实际选用的 patch 在以下原始 manifest 中记录：

- [`manifest.json`](../dynamic_mtp/patches/manifest.json)：动态 MTP 基础 patchset。
- [`gdn_k2/manifest.json`](../dynamic_mtp/patches/gdn_k2/manifest.json)：K2 捕获与 GDN 索引 patchset。
- [`gdn_workspace/manifest.json`](../dynamic_mtp/patches/gdn_workspace/manifest.json)：GDN Prefill 工作空间、降档正确性 patchset；同名 `gdn.py` 记录覆盖前一份，直接从相同未修改的源生成最终版本。

运行时选档、状态管理及 GDN 共用索引等配套逻辑保留在 [`dynamic_mtp/runtime/`](../dynamic_mtp/runtime/)。C8 阶段受控档位测试使用过仅供实验的本地固定选择文件，新发布入口不保留该测试专用机制，仍以动态策略为正常运行方式。

## 3. 上游 vLLM：本次新增的运行时适配

以下几个变化同样属于外部 vLLM 代码行为变化，不过不是直接修改上游 checkout，也不是额外 CANN patch。将每项变化实现成受版本约束的 Python import hook，随本实验启动入口交付：

| 上游文件与函数 | 分支中对应的实际实现 | 修改用途 |
| --- | --- | --- |
| `vllm/v1/core/kv_cache_utils.py` 的 `get_kv_cache_groups`、`validate_kv_cache_layout`、`_get_kv_cache_bytes_per_block`、`get_kv_cache_config_from_groups` | [`runtime/c8_grouping.py`](runtime/c8_grouping.py) | 将准确匹配的 65 层 KV/状态布局恢复为 4 组；支持 Attention 组中混合精度的独立物理页大小，返回和核验统一的真实 block 字节数；通过启动期确定的分配指针相对区间校验，拒绝不兼容几何 |
| `vllm/v1/core/kv_cache_manager.py` 的 `KVCacheManager.__init__` | [`runtime/hybrid_sparse_retention.py`](runtime/hybrid_sparse_retention.py) | 在原构造完成后核验并修正实际 Ascend hybrid coordinator 的稀疏保留间隔和本地 replay 边界 |
| `vllm/v1/core/sched/scheduler.py` 的模块加载 | [`runtime/sitecustomize.py`](runtime/sitecustomize.py) | 只在 scheduler 原模块完成加载后安装稀疏保留 hook，避免提前导入其他需补丁的上游模块；不修改 Scheduler 的请求排序策略 |
| `vllm/v1/core/kv_cache_coordinator.py`、`vllm/v1/core/single_type_kv_cache_manager.py`、`vllm/v1/core/block_pool.py`、`vllm/v1/kv_cache_interface.py` | [`launch.py`](launch.py) 的 `SOURCE_SHA256` 与上述 runtime hooks | 哈希校验及原生写入/布局不变量依赖。新稀疏 hook 利用原有可达状态筛选、块池分配、partial-tail 写入；不伪造未计算状态，不自行更改引用计数 |

混合缓存逻辑组的物理几何仅针对本次实测 Qwen 布局。每个 manager block 1664 tokens；主 INT8 页和 GDN 页分别放在大小为 3,407,872 字节的对应槽位，草稿 BF16 页为其两倍。主 16 层和三组 GDN 共享逻辑块池的不同所有者 block ID；草稿页的物理区域独立且不与 GDN 重叠。每个物理共享逻辑块真实占 61,341,696 字节。启动期由 planner 和 worker 端两层核验实际分配，不把这几何直接推广给 TP2 或其它模型。

稀疏保留 hook 纠正 Ascend 兼容层与 vLLM 0.30.0 配置之间的差异：旧兼容层读取移除的 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 环境变量，导致实际保留值成为 `None`；本方案使用 `prefix_cache_retention_interval=0` 的语义边界，并在实际 `skips_eagle_block_drop` 生效的本地 reader 条件下计算合适的 replay 边界。保留权限不等于创建 KV；仍由原生 manager 在合法已计算的状态上建立缓存键。

## 4. vLLM-Ascend：动态 MTP 补丁和 C8 适配

随第 2 节的现有动态 MTP 共同应用的 Ascend unified diff：

| Ascend 文件 | 补丁 | 用途 |
| --- | --- | --- |
| `vllm_ascend/worker/v2/aclgraph_utils.py` | `acl.patch` | 目标整图记录 |
| `vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py` | `drafter.patch` | 多档位草稿图捕获和选择 |
| `vllm_ascend/worker/v2/model_states/mamba_hybrid.py` | `npu_hybrid.patch` | GDN 状态位置适配 |
| `vllm_ascend/ops/triton/v2/mamba/precopy.py` | `npu_precopy.patch` | 状态移动的 NPU precopy |
| `vllm_ascend/ops/gdn.py` | `gdn_k2/gdn.patch`；最终由 `gdn_workspace/gdn.patch` 生成对应版本 | K2 索引共享、工作空间分组和状态归位 |

本次新增/扩展的功能性 runtime 适配：

| Ascend 文件与接口 | 分支中对应的实现 | 修改用途 |
| --- | --- | --- |
| `vllm_ascend/quantization/methods/kv_cache/kv_c8.py` 的 `AscendC8KVCacheAttentionMethod.process_weights_after_loading` | [`runtime/sitecustomize.py`](runtime/sitecustomize.py) | 核对实际加载的校准 scale、offset；草稿非 C8 层不得错误继承 INT8 KV |
| `vllm_ascend/attention/attention_v1.py` 的 C8 Attention 与 metadata builder | [`runtime/c8_nd_backend.py`](runtime/c8_nd_backend.py)、[`runtime/flat_metadata.py`](runtime/flat_metadata.py)、`runtime/sitecustomize.py` | C8 ND 直接读取、因果多验证 token flat 路径、混合 Decode/Prefill 分离，核验 scale、INT8 KV dtype、底层 interleaved stride 和图输入 |
| 同上（C8 分块 Prefill） | [`runtime/c8_lse_prefill.py`](runtime/c8_lse_prefill.py)、[`runtime/fused_gather.py`](runtime/fused_gather.py)、[`runtime/segment_plan.py`](runtime/segment_plan.py) | 分段获取历史 INT8 KV 并融合反量化到 BF16、用现有 Attention 与 LSE 合并；没有改变 CANN 数值模板 |
| `vllm_ascend/worker/v2/attn_utils.py` 的 `_allocate_kv_cache`、`_reshape_kv_cache_v2` | [`runtime/c8_grouping_worker_audit.py`](runtime/c8_grouping_worker_audit.py) | 在真实分配完成后核对物理 backing、相对偏移、精度、stride、K/V alias 页边界及 GDN 状态 view；原有实际分配器/reshape 仍执行 |
| `vllm_ascend/worker/v2/block_table.py` 的真实持久化 slot mapping；`vllm_ascend/compilation/updatable_graph.py` 的图任务 API；`vllm_ascend/worker/v2/model_runner.py` | [`runtime/c8_fold.py`](runtime/c8_fold.py) | 依赖而非直接重写这些原模块；读取已有 INT32 slot，按 Q4 图形状实现共享历史 Attention、因果尾段合并及回放元数据更新 |
| `vllm_ascend/worker/v2/aclgraph_utils.py` 的 `ModelAclGraphManager._updatable_graph_replay` | [`runtime/c8_fold.py`](runtime/c8_fold.py) | 仅 opt-in 数值诊断 hook；待实际整图回放和图参数更新提交完毕后运行，不在原等待事件的 replay 前置点同步 |
| `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`、`vllm_ascend/patch/platform/patch_kv_cache_utils.py` | [`runtime/hybrid_sparse_retention.py`](runtime/hybrid_sparse_retention.py)、[`runtime/c8_grouping.py`](runtime/c8_grouping.py) | 已安装的兼容层是本方案输入依赖，按已验证语义补充行为，不直接改写文件 |

`runtime/audit_io.py` 保持 C8 审计文件的依赖与写入；数值诊断文件只在相应运行时诊断场景使用，不作为吞吐测量。以上源模块实际核验清单位于 [`launch.py`](launch.py) 的 `SOURCE_SHA256`；混合布局启动期另有独立源码和实际 Tensor 检查。

## 5. 复现、审计与回退

1. 检出此实验分支。准备正确版本、已校准模型和原装 CANN 9.2.1 环境，不需要联网下载 CANN 算子或把实验文件覆盖到 `site-packages`。
2. 运行[本实验启动器](README.md)。它通过 Python 包解析定位两套源码、核对版本与摘要，将既有动态 MTP unified diff 的输入源和结果摘要纳入校验，按不同运行目录生成补丁模块。
3. 从启动器打印的运行目录查看 `launch-config.json`、`audit/runtime-nd-audit.jsonl`、动态 MTP 日志和启动标准输出。`[C8_GROUPING]` 记录计划和实际 Tensor 检查；`[C8_SPARSE_RETENTION]` 记录合法布局的保留策略切换；`fold_graph_capture`、`fold_graph_replay`、`fold_graph_replay_rejected` 等记录验证实际使用路径，不能仅凭配置声称已走 fold。动态 MTP 选择仍由其运行时记录实际 K。
4. 退出实验服务，停止使用本启动器；直接运行原生 `vllm serve` 即回到原安装行为。本方案不是将补丁永久写入运行环境的安装脚本。

源码版本固定在实际 0.30.0 构建所使用的源上。相同版本号的其它构建若 SHA 不符，应先核对接口/补丁和重新实测，不跳过保护检查。两套 vLLM 源版本与 CANN 版本都需要记录；本 repo 中的代码属于 Ascend fork 内可审核的**跨仓库补丁与运行时适配包**，不声称已向上游 vLLM 主仓或 CANN 仓提交变更。无需创建外部仓库的新分支才能运行此实验。
