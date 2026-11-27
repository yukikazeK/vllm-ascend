# 实验分支：W8A8C8 长上下文优化（910B4-1）

本分支 `experiments/c8-long-context-optimization` 在保留 `main` 已有功能的基础上，增加已实测的 Qwen 27B C8（INT8 KV）动态 MTP 长上下文推理方案。**不会替换原有 `vllm serve` 路径，也不需要修改 CANN。**

| 新增优化 | 实测效果（单卡 TP1，适用特定实验配置） |
| --- | --- |
| 混合精度 KV Cache 逻辑分组与存储核验 | 65 组恢复为 4 组；相同 46,592 历史、2 并发输出吞吐 38.96 → 85.84 tok/s |
| 稀疏 GDN 历史状态保留 | 63,232 历史、8 并发完整命中从 2/8 增至 8/8 |
| Q4 共享历史 Attention 与因果尾部合并 | 116,480 历史单并发合并优化后 48.65 tok/s；58,240 历史、8 并发预建历史的输出吞吐 195.06 tok/s（W8A8 参考 32.96 tok/s，实际命中不同） |
| Prefill 批预算对齐 | 8384 预算的同负载冷输入吞吐 26.25 tok/s，接近 W8A8 参考 26.33 tok/s；保留绝大部分缓存池 |

**[中文启动与实测说明](examples/c8_long_context/README.md)** · **[跨仓库源码映射、版本与 CANN 边界](examples/c8_long_context/SOURCE_MAP.zh-CN.md)**

该分支携带外部 vLLM 的版本锁定补丁及运行时适配模块，使用原装 CANN 9.2.1 和已安装的 FLA NPU 26.9.1；不纳入未完成的原生 INT8 Attention 算术研究。当前验证范围限于单卡 TP1，其他模型、TP2/DP2 及与磁盘前缀恢复组合仍需验证。

---

# vLLM Ascend 0.30.0｜910B4 推理改动

本 fork 的 `main` 基于官方 [v0.30.0](https://github.com/vllm-project/vllm-ascend/tree/v0.30.0)。当前集成了两项可按需启用的功能，面向 910B4-1 上的 Qwen 27B W8A8 部署。

| `main` 中的功能 | 改了什么 | 验证范围 |
| --- | --- | --- |
| **本地前缀缓存构建与启动加载** | 部署时一次构建 Agent 固定前缀并保存到 NVMe。服务启动时校验缓存布局、按 rank 恢复到 Mooncake。请求命中后可跳过对应的 prefill 计算。 | TP1 和 TP2 从磁盘冷恢复，2,048 输入中命中 896 个 token，16 个贪心输出 token 与基线一致。[使用指南](examples/agent_prefix_cache/README.md) · [详细设计](examples/agent_prefix_cache/docs/design.zh-CN.md) |
| **动态 MTP，保留解码整图** | 按输入长度与本实例负载选择 K=0～3，使用 `FULL_DECODE_ONLY`，处理降档时的 GDN 状态和请求的状态块预留。 | 在单卡 TP1 和双独立 TP1 上验证过。不同场景的吞吐和首 token 尾延迟表现不同。按业务负载选择固定或动态策略。[使用与限制](examples/dynamic_mtp/README.md) |

两项功能各有启动入口和适用条件。直接运行原生 `vllm serve` 使用上游行为。两项功能同时开启、动态 MTP 的 TP2 路径和其他模型组合仍待验证。

## 实验分支

| 分支 | 内容与结论 |
| --- | --- |
| [`experiments/gdn-mtp-indexing`](https://github.com/yukikazeK/vllm-ascend/tree/experiments/gdn-mtp-indexing) | 为动态 MTP 补齐常用 K2 整图尺寸，复用 GDN 状态索引，减少解码执行开销。 |
| [`experiments/gdn-workspace-correctness`](https://github.com/yukikazeK/vllm-ascend/tree/experiments/gdn-workspace-correctness) | GDN Prefill 工作空间分组与 MTP 降档时的状态归位修正。基于 910B4-1 的 OOM 与输出异常复现，并通过对应算子和状态对照。[实验说明](examples/dynamic_mtp/README.md) |
| [`experiments/c8-prefill-fused-gather`](https://github.com/yukikazeK/vllm-ascend/tree/experiments/c8-prefill-fused-gather) | 910B4-1 的 256 维 INT8 ND KV Decode 路径及分块 Prefill 的分页收集、反量化 Triton 融合。[使用与实验说明](examples/c8_nd_prefill/README.md) |
| [`experiments/gdn-layout`](https://github.com/yukikazeK/vllm-ascend/tree/experiments/gdn-layout) | GDN 的 QKV 布局复用、状态转置融合和 MegaGDN 对照。布局复用带来小幅收益，MegaGDN 在实测形状下慢于现有 FLA。 |
| [`experiments/prefill-graphs`](https://github.com/yukikazeK/vllm-ascend/tree/experiments/prefill-graphs) | 混合图加 `npugraph_ex` 和固定块 Prefill 整图。收益较小，后者在 TP2/MTP2 的混合续算中未通过输出校验。 |
| [`experiments/context-parallel`](https://github.com/yukikazeK/vllm-ascend/tree/experiments/context-parallel) | SP、GDN PCP2 原型、同机 1P1D 和双 TP1／TP2 配置对照。PCP2 的 32K 请求可执行，输出一致性仍待解决。 |

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/vllm-project/vllm-ascend)

</div>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>Documentation</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/"><b>Support Matrix</b></a> | <a href="https://slack.vllm.ai"><b>#SIG-Ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>Users Forum</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>Weekly Meeting</b></a> |
</p>

<p align="center">
<a ><b>English</b></a> | <a href="README.zh.md"><b>中文</b></a>
</p>

---
*Latest News* 🔥

- [2026/09] We released the new release candidate [v0.26.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.26.0rc1)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/) to start using vLLM Ascend Plugin on Ascend.
- [2026/08] We released the new official version [v0.23.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0/) to start using vLLM Ascend Plugin on Ascend.
- [2026/05] We released the new official version [v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.18.0/) to start using vLLM Ascend Plugin on Ascend.
- [2026/02] We released the new official version [v0.13.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.13.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.13.0/) to start using vLLM Ascend Plugin on Ascend.

<details>
<summary>More</summary>

- [2025/12] We released the new official version [v0.11.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.11.0/) to start using vLLM Ascend Plugin on Ascend.
- [2025/09] We released the new official version [v0.9.1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html) to start deploying large-scale Expert Parallelism (EP) on Ascend.
- [2025/08] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/7n8OYNrCC_I9SJaybHA_-Q) with vLLM and Tencent! Please find the [meetup slides](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/06] [User stories](https://docs.vllm.ai/projects/ascend/en/latest/community/user_stories/index.html) page is now live! It kicks off with LLaMA-Factory/verl/TRL/GPUStack to demonstrate how vLLM Ascend assists Ascend users in enhancing their experience across fine-tuning, evaluation, reinforcement learning (RL), and deployment scenarios.
- [2025/06] [Contributors](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html) page is now live! All contributions deserve to be recorded, thanks for all contributors.
- [2025/05] We've released the first official version [v0.7.3](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3)! We collaborated with the vLLM community to publish a blog post sharing our practice: [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html).
- [2025/03] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/VtxO9WXa5fC-mKqlxNUJUQ) with vLLM team! Please find the [meetup slides](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/02] vLLM community officially created [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) repo for running vLLM seamlessly on the Ascend NPU.
- [2024/12] We are working with the vLLM community to support [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

</details>

---

## Overview

vLLM Ascend (`vllm-ascend`) is a community maintained hardware plugin for running vLLM seamlessly on the Ascend NPU.

It is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Experts (MoE), Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

For detailed information on supported models and features, please refer to the [support matrix](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/).

## DeepWiki

[**DeepWiki**](https://deepwiki.com/vllm-project/vllm-ascend) is a dynamic knowledge base collaboratively maintained by the community and AI, designed to provide you with deeper technical insights beyond conventional documentation. If official documentation serves as a "quick start" guide to help you get going, the DeepWiki is your technical companion for "deep understanding". Here, you can explore: core architecture and design principles, key source code interpretations, technical context, and decision-making logic.

Whether you are a performance tuner looking to deploy vLLM-Ascend in production or a contributor aiming to build upon it for secondary development, DeepWiki offers invaluable references for you. Welcome to explore now and join us in delving deep into the technical core of vLLM-Ascend.

## Prerequisites

- Hardware: Atlas 800I A2 Inference series, Atlas A2 Training series, Atlas 800I A3 Inference series, Atlas A3 Training series, Atlas 300I Duo (Experimental)
- OS: Linux
- Software:
    - Python >= 3.10, < 3.13
    - CANN == 9.1.0 (For Ascend HDK version, please refer to the [Release Notes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/releasenote/9.1.0/release-notes.md))
    - PyTorch == 2.10.0, TorchNPU == 2.10.0.post4
    - vLLM (the same version as vllm-ascend)

## Accessing Ascend NPU

If you need to access Ascend NPU computing resources for development or testing, please visit the [HiDevLab - Online Development](https://hidevlab.huawei.com/online-develop-intro) page on the HiDevLab platform to apply for and use them.

## Getting Started

Please use the following recommended versions to get started quickly:

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
| v0.30.0rc1 | Release candidate | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/installation.html) for more details |
| v0.26.0rc1 | Release candidate | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/installation.html) for more details |
| v0.23.0 | Latest stable version | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/v0.23.0/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html) for more details |

## Branch

vllm-ascend has a main branch and a dev branch.

- **main**: main branch, corresponds to the vLLM main branch, and is continuously monitored for quality through Ascend CI.
- **releases/vX.Y.Z**: development branch, created alongside new releases of vLLM. For example, `releases/v0.13.0` is the dev branch for vLLM `v0.13.0` version.

Below are the maintained branches:

| Branch           | Status       | Note                                 |
|------------------|--------------|--------------------------------------|
| main             | Maintained   | CI commitment for vLLM main branch and vLLM v0.30.0 tag |
| releases/v0.13.0 | Maintained   | Only bug fixes are allowed, and no new release tags anymore. |
| releases/v0.18.0 | Maintained   | CI commitment for vLLM 0.18.0 version |
| releases/v0.23.0 | Maintained   | CI commitment for vLLM 0.23.0 version |
| rfc/<feature-name> | Maintained   | [Feature branches](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#feature-branches) for collaboration |

Please refer to [Versioning policy](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html) for more details.

## Contributing

See [CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html) for more details, which is a step-by-step guide to help you set up the development environment, build and test.

We welcome and value any contributions and collaborations:

- Please let us know if you encounter a bug by [filing an issue](https://github.com/vllm-project/vllm-ascend/issues)
- Please use [User forum](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support) for usage questions and help.

## Weekly Meeting

- vLLM Ascend Weekly Meeting: <https://tinyurl.com/vllm-ascend-meeting>
- Wednesday, 15:00 - 16:00 (UTC+8, [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
