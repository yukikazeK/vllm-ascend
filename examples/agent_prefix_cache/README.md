# Agent Prefix Cache Tool

把 Agent 的固定前缀在部署时构建成缓存包，保存到本地磁盘。之后启动 vLLM-Ascend 时，服务从磁盘恢复缓存包。当请求的 token 前缀匹配时，引擎可以复用相应的计算结果，减少需要重新执行的 prefill。

[中文设计说明](docs/design.zh-CN.md)涵盖缓存包结构、版本与生效机制、按 rank 恢复、兼容条件，以及 910B4-1 的真机验证范围。

## 准备运行环境

- 在目标设备的 vLLM-Ascend 环境中运行。确认 `python3`、`vllm`、`mooncake_master` 在 `PATH` 上，且 Python 环境有可用的 Mooncake 和 Ascend 运行时。
- 准备生产使用的模型、vLLM 参数和可用的 Mooncake 配置。构建和冷恢复验证会启动临时引擎，需要独占本次 profile 所用的 NPU。
- 将注册目录放在服务重启后仍可访问的本地磁盘。容器部署时，把目录挂载到主机持久存储，例如 NVMe。构建时工具会启动临时 Mooncake master，正式服务则使用你按部署配置启动的 Mooncake Store。

```bash
cd agent-prefix-cache-tool
export AGENT_CACHE_ROOT=/path/on/persistent/storage/agent-cache
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
./agent-cache --help
```

默认注册目录是 `~/.agent-cache`。也可以用 `./agent-cache --root /path/to/registry <命令>` 指定。

## 1. 编写部署配置

保存为 `profile.json`。以下示例使用两张 NPU 和 `FULL_DECODE_ONLY`。把模型路径、API 名称、并行配置和其余参数改成实际部署值，包括量化、缓存与 MTP 设置。

```json
{
  "name": "field-tp2",
  "model": "field-model",
  "model_revision": "weights-and-adapters-release-1",
  "prefix_hash_seed": 17,
  "serve_args": [
    "/absolute/local/path/to/model",
    "--served-model-name", "field-model",
    "--tensor-parallel-size", "2",
    "--compilation-config", "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}",
    "--host", "127.0.0.1",
    "--port", "8080"
  ]
}
```

- `model` 是接口发送的模型名称，应和服务暴露的名称一致。
- `model_revision` 填固定的发行标识或权重摘要。权重、适配器、tokenizer 或聊天模板内容有变更时，更新版本标识并重新构建。
- `prefix_hash_seed` 可选，会在构建和运行时统一设置 `PYTHONHASHSEED`。省略时要保证两边的哈希环境一致。
- `serve_args` 第一项为模型路径。如果你手写 `--kv-transfer-config`，需使用 Mooncake 的 `AscendStoreConnector`、`kv_both`、同步读取和整对象模式。省略时加载器会补齐兼容的连接器配置。保持 vLLM 的前缀缓存开启。

## 2. 提供 Agent 的固定前缀

保存为 `agent.json`。建议使用生产请求经过相同 tokenizer 和聊天模板后的固定开头，直接填写完整的 `token_ids` 数组。也可以交给临时 vLLM 实例按模型模板渲染：

```json
{
  "messages": [
    {"role": "system", "content": "在这里填入该 Agent 的完整固定系统提示。"}
  ],
  "add_generation_prompt": false
}
```

这是配置结构示例。实际构建请填入完整固定前缀，短文本可能达不到可缓存块的长度。Agent 的工具定义或模板参数固定时，可传入 `tools`、`chat_template_kwargs`。工具会对 `messages` 调用 `/tokenize`，默认关闭 `add_generation_prompt`。渲染后的尾部和后续真实请求必须保持连续匹配，特别是多轮消息的分隔符。若使用 `cache_salt`，将它作为配置的顶层字符串字段，并在服务请求中使用同一个值。

## 3. 构建、检查和启用

```bash
./agent-cache build --agent field-report --config agent.json --profile profile.json
./agent-cache list --agent field-report
./agent-cache verify field-report --profile profile.json
./agent-cache activate field-report --profile profile.json
```

构建会运行一次导出和一次从磁盘开始的冷恢复验证。只有实际命中外部前缀，并与无缓存基线生成相同的 16 个贪心输出 token，才会发布版本。查看结果中的 `version` 和 `external_prefix_hit_tokens`。如需保证业务前缀的最低覆盖，构建时加 `--min-cached-tokens N`，将 `N` 替换为门槛。构建日志在注册目录的 `builds/` 下。模型启动较慢时可用 `--timeout` 调整临时引擎的启动超时，默认 600 秒。

## 4. 每次启动加载

先启动正式 Mooncake Store，使 `MOONCAKE_CONFIG_PATH` 指向它的配置。已有 Store 时可直接使用。单机部署若使用独立 master，可在另一个终端启动下面的进程，并确保 Mooncake 配置中的 `master_server_address` 是 `127.0.0.1:50051`。

```bash
mooncake_master --rpc_address=127.0.0.1 --rpc_port=50051 \
  --metrics_host=127.0.0.1 --metrics_port=50052
```

用以下命令启动模型服务。日志可以用于确认缓存已加载。

```bash
./agent-cache serve --profile profile.json > vllm-agent-cache.log 2>&1
```

启动时，工具选择该 profile 已激活的缓存，校验磁盘对象并写入 Store。请求到达后，引擎查找与请求开头匹配的对象并将所需状态传到 NPU。每次启动执行的是加载流程。首次部署以及前缀或配置变更时才执行 `build` 或 `update`。

在另一个终端查看加载日志。发送一条真实 Agent 请求前后，可分别读取一次命中累计计数，比较增量。

```bash
grep 'Loaded local prefix package' vllm-agent-cache.log
curl -fsS http://127.0.0.1:8080/metrics | grep '^vllm:external_prefix_cache_hits_total'
```

## 5. 更新、回滚和迁移

修改 `agent.json` 后构建新版本，再切换激活版本。以下脚本中的 `NEW_VERSION` 和 `OLD_VERSION` 是版本 ID，需要换成 `build` 或 `list` 返回的值。切换后需要重启 vLLM 才会加载所选版本。

```bash
./agent-cache update --agent field-report --config agent.json --profile profile.json
./agent-cache activate field-report@NEW_VERSION --profile profile.json
# 需要回滚时执行
./agent-cache rollback field-report@OLD_VERSION --profile profile.json
```

| 命令 | 用途 |
| --- | --- |
| `list [--agent AGENT]`、`show AGENT[@VERSION] [--profile FILE]` | 查看版本、前缀命中数和激活状态。 |
| `verify AGENT[@VERSION]` | 校验磁盘包。加 `--cold --profile FILE` 会另启临时实例，复查冷恢复命中和输出一致性。 |
| `export AGENT@VERSION DIRECTORY`、`import DIRECTORY` | 导出目录供离线传输，在兼容目标环境导入。导出目标目录必须尚不存在。导入后在目标机校验并激活。 |
| `remove AGENT@VERSION` | 删除指定的非激活版本。 |

同一 profile 可启用多个 Agent。缓存包保存原始前缀 token，可能包括 `cache_salt` 和敏感提示状态，请按 Agent 配置的数据权限保存与传输。

## 直接给启动命令指定缓存文件

已有兼容缓存包时，也可以直接使用本仓库的启动脚本，保留生产所需的全部 vLLM 参数。路径可填包目录或 `objects.json` 文件。开关默认关闭。

```bash
python3 vllm_with_prefix_cache.py serve /path/to/model \
  --load-local-prefix-cache \
  --local-prefix-cache-path /path/to/package
```

## 已验证范围

Ascend 910B4-1 上使用 Qwen 27B W8A8、固定 MTP2 和 `FULL_DECODE_ONLY`。TP1 和 TP2 均从磁盘恢复了 2,048 输入 token 中的 896 个外部缓存 token，无缓存基线与恢复后的 16 个输出 token 全部一致。TP2 还验证了版本复用、激活、导出和导入。切换模型、精度、TP4/TP8 或其他并行组合时，请用实际 profile 构建并在目标环境验证。
