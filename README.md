# Jev-Duplex：用通用决策模型探索话轮判断

**全双工语音交互**允许助手在说话的同时接收用户输入。用户可以随时插话、提问或附和，助手则需要根据这些输入判断何时继续讲、何时停下，以及何时开始回应。

例如，助手正在讲解时，用户说“停一下，我想先问第一步”，助手应该暂停；用户说“嗯，你继续”，则可以接着讲。用户停顿时也有类似的问题：“我想问一下……”可能还没说完，而“我的问题就这些”则表达了结束。这些判断属于 **turn-taking（话轮交替）** 问题，涉及轮次结束、打断与附和等交互现象 [1]。

已有研究通过不同方式处理这些交互动态：Moshi 等原生全双工大模型直接让模型处理用户输入和模型输出双流，实现模型层面的全双工 [2]；Easy Turn 和 SoulX-Duplug 等工作则研究独立的话轮状态预测模块，实现系统层面的全双工 [3, 4]。

话轮预测的一个关键问题是泛化：在自有测试集上取得高分的方法，换到其他场景后可能明显退化。FastTurn 的跨测试集评测显示，EasyTurn 在自身测试集上的总体正确率为 96.38%，在 FastTurn 和 Smart Turn 中文集上分别为 78.05% 和 57.16%；该研究也指出 EasyTurn 测试集对语义线索的依赖 [7]。

这个项目尝试用 **Jev 这类通用决策模型，仅根据文本判断话轮状态**，检验 EasyTurn 测试集中的标签在多大程度上可以由语义恢复。现成 ASR 加上固定状态说明即可完成测试，无需针对 EasyTurn 训练决策模型。这个简单基线为理解测试集的语义偏好及后续数据建设提供参照。

## 为什么尝试决策模型

决策模型接收当前情景和一组候选结果，返回选择及其概率。话轮判断可以自然地写成这样的任务：给出用户表达和必要上下文，询问它是否已经说完、是否在附和，或是否希望暂停交互。

Jev 是 TypeSafe 提供的决策模型，其官方介绍将训练方法称为 **RLCD（Reinforcement Learning for Calibrated Decisions）**，关注决策概率的校准 [5]。这让我们除了检查“标签是否选对”，还可以观察“概率是否可靠”。例如，模型给出约 80% 的把握时，是否意味着这项决策的正确性真的有约 80% 的概率？校准对于全双工预测器尤其重要，因为全双工系统在设计中可以利用校准的概率判断何时立即响应、何时等待更多输入。


## 项目结构

- [决策模型入门](SPX-CD_calibration_learning.ipynb)：面向第一次接触 Decision Model 的读者。使用纯文本全双工情景，逐步学习 Choice、Noul、Score、MultiChoice，读取候选概率，理解校准，再做少量模型比较。
- [Jev × EasyTurn 实验](Jev_EasyTurn_text_experiment.ipynb)：用 Qwen3-ASR-1.7B 转写官方测试音频，经 Surd AI 的 Python SDK 调用 Jev，进行 zero-shot 四类话轮预测，并用参考文本对照分析测试集的语义可预测性。
- [实验报告](EXPERIMENT_REPORT.md)：以 Jev 的文本基线结果讨论 EasyTurn 的语义偏好，结合跨测试集表现，提出后续话轮数据标注与合成的启示。

入门篇选用 Surd AI 的 Simplex CD（SPX-CD），方便读者利用近期的免费体验动手。当前政策见 [官网](https://surdai.com/en)；到 [注册页面](https://surdai.com/en/register)创建账户，在 [Access tokens](https://surdai.com/en/platform/keys) 获取 key。接口说明见 [平台文档](https://surdai.com/en/platform/docs)与 [Python SDK](https://github.com/SPX-AI968/spx-sdk)。

实验篇沿用 `simplex_ai` 和 Surd AI 的 key，把模型设为 `typesafe-ai/jev`。使用 TypeSafe 自有 key 的读者可以参考入门篇末尾的官方 SDK 迁移示例。

## EasyTurn 实验

核心问题是：**无需任务训练，Jev 仅凭 ASR 文本能多接近 EasyTurn？这样的结果揭示了测试集的哪些语义特性？**

实验使用 [EasyTurn 官方测试集](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Testset)：800 条语音，complete/incomplete 各 300 条，backchannel/wait 各 100 条，真实与合成各 400 条。每段音频独立转写，Jev 的主实验输入只包含转写文本。四类状态沿用论文第 2.1 节 [3]：

- `<complete>`：用户已完整表达意图，期待回应。
- `<incomplete>`：用户尚未说完，需要继续表达。
- `<backchannel>`：用户用简短附和表达倾听或理解。
- `<wait>`：用户明确要求暂停或结束交互。

ASR 选用 **Qwen3-ASR-1.7B** [6]。实验使用本地 Transformers 后端、BF16 和自动语言识别，先检查小样本，再跑完整测试集。

**ASR → Jev 的四类平均正确率为 92.17%，距 EasyTurn 论文的 95.75% 相差 3.58 个百分点；参考文本 → Jev 为 95.42%，只差 0.33 个百分点。** 两种输入的总体正确率分别为 89.75% 和 94.38%。这组结果为 EasyTurn 测试集较强的语义偏好提供了佐证。

EasyTurn 的训练数据大量通过文本模型标注与筛选，合成部分先生成文本再做 TTS；其 Whisper + LLM 的 ASR+Turn-Detection 路径又显式利用转写来预测状态 [3]。本项目从通用文本决策模型的角度补充一个参照：这套测试中的大量状态，可以由现成模型直接从文本中恢复。参考文本对照、逐类指标、来源切片和概率分析用于进一步解释这个结果。

后续全双工数据建设需要深入考虑声学与对话接续信息。标注时结合音频中的停顿、韵律和前后话轮，提供超出文字完整性的监督；合成时同时设计文本与声学表达，覆盖相近文本对应不同话轮状态的情况，并参照真实对话分布核验样本。具体论证与结果见 [实验报告](EXPERIMENT_REPORT.md)。

耗时统计读取 API 返回的服务端 `inference_ms`；本次 Jev 响应未提供该字段，因此报告没有耗时结果。失败或无效输出保留在分类分母中，概率指标同时报告有效覆盖率。

## 目录与阅读顺序

```text
Jev-Duplex/
├── README.md                          # 项目定位、安装与运行
├── SPX-CD_calibration_learning.ipynb   # 决策模型入门教程
├── Jev_EasyTurn_text_experiment.ipynb  # ASR / 参考文本对照实验
├── EXPERIMENT_REPORT.md               # 实验结果与分析
├── figures/                           # 报告中的两张图
├── results/easyturn/                   # evaluation.jsonl、metrics.json；cache/ 保存运行数据
├── requirements.txt                   # 入门教程依赖
├── requirements-asr.txt               # 完整实验依赖
└── .env.example                       # API key 配置模板
```

第一次接触决策模型，先阅读入门教程；想了解实验结论，直接打开实验报告。实验 notebook 保留已执行的输出，GitHub 上可以直接阅读。[完整预测与指标](results/easyturn/README.md)提供逐条预测和指标汇总。

本地运行还会生成 `data/`（约 143 MB 测试音频）、`models/`（约 4.4 GB ASR 权重）、`results/easyturn/cache/`（采集缓存）。这些目录由 `.gitignore` 排除，运行时按需生成。

## 运行方法

在仓库根目录启动 Jupyter，让 notebook 的相对路径指向项目目录。建议使用 Python 3.11 的独立环境：

```bash
mamba create -n jev-duplex python=3.11 -y
mamba activate jev-duplex
python -m pip install -r requirements.txt
python -m ipykernel install --user --name jev-duplex --display-name 'Python (jev-duplex)'
jupyter lab
```

入门篇按顺序运行会调用 API。可以在 notebook 的隐藏输入框中填写 key，也可以先配置环境变量：

```bash
cp .env.example .env
# 编辑 .env，填入自己的 SIMPLEX_API_TOKEN。
set -a
source .env
set +a
jupyter lab
```

完整实验另外安装 ASR 依赖：

```bash
python -m pip install -r requirements-asr.txt
```

## 引用

[1] Jiang et al. (2026). [TurnBench: A Multi-Domain Benchmark for Turn-Taking Dynamics in Spoken Dialogue](https://arxiv.org/abs/2608.25218).

[2] Défossez et al. (2024). [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037).

[3] Li et al. (2025). [Easy Turn: Integrating Acoustic and Linguistic Modalities for Robust Turn-Taking in Full-Duplex Spoken Dialogue Systems](https://arxiv.org/abs/2509.23938).

[4] Yan et al. (2026). [SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation](https://arxiv.org/abs/2603.14877).

[5] TypeSafe AI. [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)（官方技术介绍）。

[6] Qwen Team (2026). [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337). [官方实现](https://github.com/QwenLM/Qwen3-ASR)。

[7] Wang et al. (2026). [FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection](https://arxiv.org/html/2604.01897v2#S3.SS4)（跨测试集结果见表 3）。

## 许可

本项目由 Dove 编写，原创代码、notebook 和说明文档采用 [MIT License](LICENSE)。
使用、修改和分发时请保留版权与许可声明。

EasyTurn 数据集、模型及第三方 SDK 和依赖遵循各自的许可证与服务条款；
其来源见上文引用及 notebook 中的链接。
