# 910B4 动态 MTP 与解码整图

在同一 V2 推理进程中，根据本实例的输入长度和运行、等待请求数，选择每步生成的草稿 token 数 K（0～3）。保留 `FULL_DECODE_ONLY`。图查找会按验证宽度寻找兼容的整图。降到 K=0 时，丢弃待验证草稿，并在 GDN/Mamba 状态预拷贝中归位已接受的状态。

请求首次进入调度器时计算状态块预留上限，每个调度步按批内最低上限约束 K。K=0 的新请求仍预留一个临时状态槽。跳过草稿的旧请求在结束前保持在普通解码路径，新的请求在负载回落后可以重新使用推测解码。

## 使用

需要 vLLM 和 vLLM-Ascend **0.30.0**、GNU `patch`，以及可用的 Ascend 环境。请用同一个已安装 vLLM 的 Python 环境执行启动脚本。启动脚本校验相关模块的源码摘要，在运行目录构建补丁模块，再将它们注入 vLLM 进程。默认直接使用原生 `vllm serve` 的服务不受影响。

以下命令展示了必要参数。模型、量化和其他参数按实际部署填写。

```bash
cd examples/dynamic_mtp
python3 launch.py --state-dir /tmp/ascend-dynamic-mtp --policy policy.example.json -- \
  serve /path/to/qwen-27b-w8a8 \
  --tensor-parallel-size 1 --max-num-seqs 128 --no-async-scheduling \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,1,3],[2,3,2],[4,7,1],[8,128,0]]}'
```

`policy.example.json` 的外层键是输入 token 档位，内层键是**单实例**并发上限，对应的值是目标 K。输入长度选择比例上最接近的档位，并发向上取表内档位，超过最大档时使用该行最后一档。运行时的实际 K 还受批内请求状态约束。服务日志 `Dynamic MTP selected K=...` 展示调度器的选择。

## 验证范围

历史 910B4-1 真机测试覆盖 Qwen 27B W8A8、TP1、V2 同步调度和单机两个独立 TP1 实例。验证包含 K 切换、解码整图回放和逐 token 的定向正确性。自动选 K 仅在部分场景取得优势，持续高并发和首 token 尾延迟仍应按业务测试。示例策略来自测过的输入与并发档位，长输入和不同模型需要自行标定。TP2、异步调度以及与磁盘前缀恢复同时使用仍待验证。

本地测试可运行 `python3 -m unittest discover -s tests -v`。
