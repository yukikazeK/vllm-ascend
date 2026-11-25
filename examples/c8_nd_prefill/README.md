# 910B4-1 的 256 维 C8 ND 注意力与 Prefill 收集融合

本实验使用 vLLM、vLLM-Ascend 0.30.0、CANN 9.2.1，验证 Qwen3.8-27B W8A8C8、TP1、MTP0、`FULL_DECODE_ONLY`。模型的 24 个 Query Head、4 个 KV Head 与 256 Head dim 对应本次验证形状。

## 使用

使用安装了相应 vLLM 的 Python 环境，模型目录需要包含校准导出的 `quant_model_weights-kv-cache-c8.safetensors`。启动器核对主要源文件摘要，加载 C8 适配，并保存各层 scale 和 INT8 缓存的核验记录。

```bash
cd examples/c8_nd_prefill
ASCEND_RT_VISIBLE_DEVICES=0 python3 launch.py --state-dir /tmp/c8-runs --fused-gather -- \
  serve /path/to/qwen3.8-27B-w8a8c8-rebuilt \
  --tensor-parallel-size 1 --dtype bfloat16 --quantization ascend --kv-cache-dtype int8 \
  --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 --language-model-only --trust-remote-code \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --mamba-ssm-cache-dtype float32 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,4,8,16]}' \
  --host 0.0.0.0 --port 8080
```

去掉 `--fused-gather` 可使用本实验的原 ND C8 路径进行对照。当前启动器按已验证的 TP1、MTP0 和零前缀命中配置提供实验入口，MTP、TP2 以及前缀缓存命中的组合需要单独验证。

## 工作原理

ND C8 Decode 保持原混合 KV 池的交错页布局。K/V 各使用偏移视图，分页索引映射到对应物理页，FIA 直接读取 INT8 K/V。Decode 整图回放更新序列长度与页表。

分块 Prefill 续算使用新的 Triton kernel，根据原始页表一次读取 INT8 K/V、应用对应的逐通道 scale，并直接写入连续 BF16 TND 输出。使用原始交错页的物理 stride，保存最后一页和各请求的真实长度边界。元数据中的累计 token 偏移在同一次前向内跨层复用。BF16 FIA 注意力计算和纯新 Prefill 路径沿用既有实现。

## 算子微测

使用实际模型校准 scale、随机物理页及非完整尾页。八个匹配形状包含单请求与变长双请求，对照旧 gather + dequant 的 K/V 输出均逐元素一致。

| KV 历史长度 | 旧收集反量化 ms | 融合完整路径 ms | 旧显存增量 MiB | 融合增量 MiB |
| ---: | ---: | ---: | ---: | ---: |
| 4093 | 1.13 | 0.47 | 32.03 | 15.99 |
| 32768 | 1.71 | 0.55 | 192.04 | 128.00 |
| 131072 | 5.74 | 1.06 | 768.14 | 512.00 |

32K、128K 的完整收集融合分别约为 3.1 倍、5.4 倍速度。上表记录的是收集与反量化算子阶段，整机 Prefill 还包含 Attention、GDN 和线性层。

### 直接 INT8 Prefill 算子调研

公开的 FIA V3 文档列出多 token 的反量化接口。本机组合测试中，FP16 多 token 分页 KV 合并 scale 在 tiling 阶段失败，分离 scale 变体也未通过。连续的非分页 FP16 Query + INT8 KV 通过八个形状的数值对照，长历史 Attention 计算成本明显高于现有 BF16 FIA。例如 32K 历史、2048 Query 的完整算子路径为 50.74 ms，既有路径为 12.49 ms。高性能模式在长历史形状还出现数值异常。本实验选择上述收集融合路径。

[FIA V3 公开接口说明](https://github.com/hicann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md)

## 真实服务对照

同一张 910B4-1、同一份重建 W8A8C8 模型、TP1、MTP0、HBM 利用率 0.9、8192 批次 token 预算、解码整图捕获尺寸 1/4/8/16，输出 256 token。两臂均使用 INT8 KV，配对请求的提示哈希、真实输入和输出计数相同，前缀缓存实测零命中。

| 输入 / 并发 | 原路径输出 tok/s | 融合输出 tok/s | 吞吐变化 | 原 TTFT 中位 s | 融合 TTFT 中位 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K / 1（三轮热态） | 15.20 | 15.38 | +1.2% | 7.526 | 7.339 |
| 32K / 4 | 23.59 | 24.07 | +2.0% | 20.584 | 20.056 |
| 32K / 16 | 28.31 | 29.15 | +3.0% | 67.565 | 65.444 |
| 128K / 1 | 4.15 | 4.21 | +1.6% | 51.263 | 50.288 |

初次融合请求包含 Triton JIT 编译开销，单独保留原始样本。三轮长请求预热后的 32K / 1 对照，每对完整输出 token 均完全相同。其余性能点各一组，逐请求完成、计数和异常重复字符检查均通过。高并发时完整回答的 token 哈希存在差异；业务质量仍需进一步独立测试。

打包后的 `launch.py --fused-gather` 入口另行在真机启动并完成 32K 输入、256 输出请求，16 个全注意力层均确认使用融合 kernel。
