# SP、PCP 与 PD 分离实验

基线为 910B4-1、Qwen 27B W8A8 和 vLLM-Ascend 0.30.0。

| 项目 | 已观察到的结果 |
| --- | --- |
| TP2 序列并行 | 32K 单并发 TTFT 在关闭和开启 SP 时分别约为 6.10 和 6.17 秒。开启 SP 的两卡采样 HBM 各降低约 260 MiB。 |
| GDN PCP2 原型 | 单请求 32K 的 TTFT 约从 7.23–7.26 秒降至 6.91–6.92 秒。完整耗时接近，输出 token 与 TP1 参考存在差异。 |
| 同机 1P1D | 指定 Agent 负载的整组输出吞吐为 616.24 tok/s，双独立 TP1 对照为 1,394.07 tok/s。该结果依赖实际负载与缓存命中。 |

`sitecustomize.py` 是 PCP2 原型的实验入口。只在隔离环境通过 `PYTHONPATH` 和 `GDN_PCP_PILOT=1` 加载，用于 TP1×PCP2、单并发、MTP 关闭、无前缀缓存的定向诊断。SP 和 PD 使用原生配置验证。
