# EasyTurn 测试结果

800 条测试样本，模型为 `typesafe-ai/jev`，比较官方参考文本与 Qwen3-ASR-1.7B 转写文本。
参考文本有 800 条有效决策，ASR 文本有 799 条有效决策、1 条空转写。

ASR → Jev 的四类平均正确率为 **92.17%**，参考文本 → Jev 为 **95.42%**；
EasyTurn 论文报告 **95.75%**。Jev 在本项目中仅使用固定状态说明，没有任务训练或 few-shot 示例。
这个文本基线为测试集较强的语义偏好提供佐证；结果与跨数据集表现、后续音频标注及合成工作的联系见 [实验报告](../../EXPERIMENT_REPORT.md)。

## 分析结果

- `evaluation.jsonl`：全部 1,600 条样本 × 输入形式的标注、预测、归一化概率、状态、服务端耗时及 `correct`。筛选 `correct=false` 查看所有错误与缺失预测。
- `metrics.json`：合并后的指标与实验设置。`overall` 包含正确率、四类平均正确率、覆盖率、NLL、Brier、ECE 和服务端耗时；`per_class`、`per_source`、`per_source_class` 提供类别与来源切片；`confusion_matrices` 保存混淆计数；`reliability` 保存各组合的概率分箱；`asr_quality` 保存 CER、完全匹配率和转写状态；`paired_input_changes` 保存共同有效样本上的文本变化与两种输入的正确情况；`experiment` 记录本轮提示词、版本和运行参数。

分类统计以每种输入的全部 800 条为分母；概率指标使用有效决策。
候选概率经过归一化，处理 API 小数舍入产生的总和误差。
服务端耗时缺失时记录有效数量为 0，其他耗时指标为 null。

## 运行缓存

`cache/` 由 Git 忽略，供断点重跑使用：

- `config.json`：检查本轮配置与已有缓存是否一致。
- `manifest.jsonl`：样本 ID、音频路径、参考文本、标注和音频哈希。
- `asr.jsonl`：转写、语言与状态；同一 ID 取最后一条记录。
- `decisions.jsonl`：实际请求、原始 API 响应与状态；同一样本 × 模型 × 输入形式取最后一条记录，复用成功调用。
- `model_catalog.json`：采集时的模型目录与提供方信息。

`evaluation.jsonl` 可以按 ID 与 `manifest.jsonl`、`asr.jsonl` 连接，查看具体文本。
运行 notebook 的分析章节会更新外层两个结果文件，采集缓存继续保留。
`OUTPUT_DIR` 默认取 `CACHE_DIR` 的上一级；另一轮实验可使用 `results/easyturn/trial/cache/`，分析结果写入 `trial/`。
本次实验使用 MPS、batch size 1；notebook 支持 CUDA、MPS 与 CPU。

详细解释见 [实验报告](../../EXPERIMENT_REPORT.md)，计算过程见 [实验 notebook](../../Jev_EasyTurn_text_experiment.ipynb)。
