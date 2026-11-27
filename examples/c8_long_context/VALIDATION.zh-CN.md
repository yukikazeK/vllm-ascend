# 分支交付验证

2026-11-27，vLLM/vLLM-Ascend 0.30.0、单张 Ascend 910B4-1、CANN 9.2.1、原装 FLA NPU 26.9.1。除早期实验数据外，对本分支实际启动入口重新做了集成验证。没有修改正式安装、模型权重或 CANN/FLA 包。

## 主机回归

- `python3 -m unittest discover -s examples/c8_long_context/tests -v`：启动器、分组与地址边界、稀疏保留、flat metadata、LSE 分段、Fold shape/padding 与更新顺序等 27 项便携测试通过。
- 指定 `C8_TEST_VLLM_SOURCE` 为已验证上游 vLLM 源码根目录后重测：另 2 项实际源码 AST 合约测试通过（分配器与原生 GDN 可达块掩码）。
- `python3 -m unittest discover -s examples/dynamic_mtp/tests -v`：已有 6 项动态 MTP/GDN 单测通过。

## 新启动器实际验证

`--verify-only` 在目标 Python 环境核验 26 个安装源码文件 SHA-256、CANN 9.2.1、模型几何/校准文件及动态策略，通过。注意正常 Python 启动会注册已安装 FLA NPU 自带的 vendor 目录；启动器允许该已验证依赖，而非任意实验 OPP。

使用默认 8384 批预算，不启用诊断快照，单卡服务成功启动，实际分配 4 个缓存组和 444 个逻辑块。16 层 INT8 target、1 层 BF16 draft、48 层 GDN 的真机存储地址、dtype 与 stride 认证通过；稀疏保留从 `None` 修正为 `0`；动态 MTP 实际 K3、Q4 整图回放均有日志证据。

| 阶段 | 输入 token | 实际前缀命中 | 实际输出 | 状态 |
| --- | ---: | ---: | ---: | --- |
| 短请求兼容检查 | 128 | 0 | 512 | 完成 |
| 独立历史准备（不计正式性能） | 11648 | 0 | 1 | 完成 |
| 历史续算 | 12160 | 11648 | 512 | 完成 |

以上 smoke 是对启动器及适配层完整链路的检查，不是重新采集全性能矩阵。完整既有吞吐与真实缓存数值抽样范围见 [README](README.md)。本次仅发布可审核源码、测试及聚合数据，不发布私有请求内容、模型权重、服务日志或校准张量。
