# Agent 前缀缓存工具设计

[返回用户指南](../README.md)

## 1. 部署阶段构建，服务启动时恢复

面向一组固定 Agent 提示词和部署配置。部署阶段用临时引擎计算前缀，把 vLLM-Ascend 的外部缓存对象保存到持久磁盘，再用全新的引擎和 Store 验证这些对象能被当作真实前缀使用。

正式服务启动时，加载器按引擎的实际缓存布局，将磁盘对象恢复到 Mooncake Store。加载完成后服务进入就绪状态。收到匹配请求时，引擎再把需要的状态从 Store 传到 NPU，继续处理前缀之后的输入并生成回复。服务启动耗时包含读盘和写入 Store，实际请求还承担查找和传输的开销。首 token 延迟能减少多少，取决于命中长度、传输耗时以及原本 prefill 的成本。

```mermaid
flowchart LR
    A["Agent 固定前缀与部署配置"] --> B["临时 vLLM 和 Mooncake 生成缓存"]
    B --> C["写入本地缓存包"]
    C --> D["新引擎和新 Store 验证命中及输出"]
    D --> E["发布版本并激活"]
    E --> F["正式启动时从磁盘恢复到 Store"]
    F --> G["匹配请求加载到 NPU，复用前缀"]
```

## 2. 构建及冷恢复验证

1. CLI 根据 profile 启动仅监听回环地址的临时 vLLM 和 Mooncake master。构建与正式服务使用同一模型和引擎参数，临时运行另选端口，并关闭 Mooncake SSD offload。
2. 工具取得固定前缀的精确 token 序列，向临时服务发送一个贪心探针，保存无外部前缀命中时的 16 个输出 token。
3. [导出钩子](../bootstrap/local_prefix_loader.py)拦截后端的整对象写入，读取实际写进 Store 的对象。工具记录 Store 键、长度、SHA-256 和所属 rank，并保存各 rank 的真实缓存布局。混合模型的状态由实际 connector 暴露的对象决定，Qwen 验证使用其混合缓存布局。
4. 工具关闭构建进程，使用全新的 Mooncake master 和 vLLM 从磁盘恢复。比较同一探针的 16 个贪心 token，读取外部命中累计计数的增量。命中数还须达到 `--min-cached-tokens`，默认至少 1 个。
5. 通过校验的包才会进入注册目录。探针只证明当次部署配置、该前缀和该输出路径成功。可复用长度由 connector 和引擎的块规则决定，要从实际命中数判断。构建测试用 16 个输出 token 验证恢复正确性。

[临时进程、探针与冷恢复实现](../agent_cache/engine.py)

## 3. 磁盘结构、版本及激活

| 文件或目录 | 内容 |
| --- | --- |
| `package.json` | Agent 标识、版本、profile 指纹、模型发行标识、文件索引摘要、构建时间和冷恢复结果。 |
| `probe.json` | 固定前缀 token、基准输出 token，以及使用时的 `cache_salt`。供重新冷验证。 |
| `layouts.json` | 各 rank 的缓存布局，以 PP、PCP、TP 序号组成的键索引。 |
| `objects.json`、`*.bin` | Store 对象索引及原始字节。索引记录文件名、长度、SHA-256 和所属 rank。 |
| `agents/<agent>/<version>/` | 注册目录中的已发布版本。 |
| `builds/` | 构建与冷验证的运行目录，含临时引擎、master 日志和恢复结果。 |
| `deployments/`、`active/<profile 指纹>` | 由多个 Agent 版本合并的部署快照，以及正式启动读取的当前快照链接。 |

版本由 profile 指纹、实际前缀 token 摘要和 salt 摘要共同决定。完全相同的输入配置可直接复用已校验的包。同一 profile 下，多个 Agent 可以同时激活。激活时工具校验各包布局，用硬链接组装对象，并在进程锁内原子替换当前快照链接。相同 Store 键的对象只有在内容一致时才能合并。更新和回滚在下次服务启动时生效。删除未激活版本后，历史部署快照中的硬链接仍可能占用空间。

完整缓存包可用 `export`、`import` 跨机器离线传输。目标机需使用兼容环境，重新激活导入的版本。磁盘校验用于发现文件不一致。前缀 token、salt 和模型内部缓存均应作为部署数据控制访问。

[注册、校验与快照切换实现](../agent_cache/registry.py)

## 4. 哪些配置影响复用

| 变更 | 对缓存的影响和处理 |
| --- | --- |
| 模型、权重、适配器、量化、KV 精度 | 这些配置会改变缓存内容或表示。更改模型发布物时更新 `model_revision`，用新 profile 构建。工具不会自动对整个权重目录计算哈希。 |
| tokenizer、聊天模板、固定消息、工具定义 | 它们可能改变实际 token 序列。部署前重新渲染，检查其是否为生产请求的真实前缀。 |
| TP、PP、PCP、DCP 与缓存布局 | 改变分片和状态组织。每种部署需要单独构建并检查目标环境。 |
| MTP 与其他引擎参数 | `serve_args` 用于计算 profile 指纹。修改 MTP 参数会得到不同的 profile，即使某些设置在理论上可能拥有相同的缓存布局。 |
| vLLM、vLLM-Ascend、Mooncake、设备或环境变量 | 包的 profile 指纹不自动采集这些版本及环境。升级或迁移时用目标环境做冷恢复验证，确认 connector、状态布局和传输路径仍兼容。 |
| `cache_salt`、前缀哈希种子 | salt 参与版本标识，实际请求要使用相同的 salt。`prefix_hash_seed` 可显式固定构建和运行时的 Python 哈希种子。 |
| 服务地址和端口 | `--host` 与 `--port` 排除在 profile 指纹之外。其他 `serve_args` 按原顺序参与指纹，模型路径也参与。 |

管理模式同时检查两层兼容性。CLI 先按 profile 指纹选择部署快照。Worker 注册缓存时，加载器进一步对比实际 rank 布局，包括分组、块长度、块步幅，以及各层缓存张量的形状和精度。布局检查不通过会中止启动。

## 5. AscendStore、Mooncake 与按 rank 恢复

当前适配的是 vLLM-Ascend 的同步整对象 `AscendStoreConnector`，后端使用 Mooncake，角色为 `kv_both`。启动脚本把 manifest 路径交给 Python 启动钩子。管理模式在 Worker 缓存注册后按 rank 导入对象，并校验磁盘内容以及写进 Store 后读回的字节。需要模型后端提供相应的缓存注册及传输能力。

真机用的 Mooncake 0.3.13 会在整对象 `put` 接口拒绝 `prefer_alloc_in_same_node`。导入时将它设为 `False`，仍通过 `preferred_segment` 指定当前后端的本地 segment，并保留按 rank 筛选对象。这个调整影响写入位置的偏好设置。对象的键、字节、精度和切分保持原样。多节点场景的实际放置、导入耗时和请求传输仍需按目标拓扑验证。已测试环境还依赖 Mooncake 的主机缓冲区对齐适配。

[启动器](../vllm_with_prefix_cache.py) · [Python 启动钩子](../bootstrap/sitecustomize.py) · [导入实现](../bootstrap/local_prefix_loader.py)

## 6. 验证边界与排查

Ascend 910B4-1 上已验证 Qwen 27B W8A8、固定 MTP2、`FULL_DECODE_ONLY`。TP1 和 TP2 各从磁盘恢复 2,048 输入 token 中的 896 个外部缓存 token，并与无缓存基线生成相同的 16 个贪心 token。冷验证使用新的 vLLM 和 Mooncake 进程。实际整机重启、其他模型、TP4/TP8 以及多节点路径仍待验证。

| 现象 | 检查项 |
| --- | --- |
| 启动提示没有激活包 | 查看已激活版本与当前 profile 是否对应。模型路径和参数顺序也参与指纹。 |
| 提示 rank 布局不兼容 | 比对实际运行时、并行配置、权重发布物和缓存精度。使用目标配置重新构建。 |
| 日志已加载包，请求却没有外部命中 | 对照真实请求的 token 开头、模板、salt 与哈希设置，查看请求前后的外部前缀命中计数。 |
| 冷验证命中数低于目标 | 查看 `restore-proof.json`、`build-vllm.log` 和 `restore-vllm.log`，确认模型缓存块规则和实际覆盖长度。 |
| Mooncake 注册或写入报错 | 查看 master 和 vLLM 日志，检查配置、缓冲区注册、传输后端及 Store 容量。 |

[本地注册目录回归测试](../tests)可用 `python3 -m unittest discover -s tests -v` 运行。
