# KRIS-Bench 评测工具

[KRIS-Bench](https://arxiv.org/abs/2505.16707)（Knowledge-based Reasoning in Image-editing Systems Benchmark）的评测流水线。本仓库包含评测脚本与模型输出，用于复现基准测试结果。评测使用 GPT-4o 作为自动评判，对编辑后的图片按 1–5 分进行打分。

## 快速开始

### 1. 克隆仓库 

```bash
git clone https://github.com/nameoffly/Kris_Bench_output.git
cd Kris_Bench_output
```

### 2. 从 HuggingFace 下载数据

```bash
# 如未安装 huggingface-cli，先安装
pip install huggingface_hub

# 下载两个 zip 文件
huggingface-cli download hisheep/kris_outputs KRIS_Bench.zip --local-dir . --repo-type dataset
huggingface-cli download hisheep/kris_outputs output_bagel.zip --local-dir . --repo-type dataset
```

- `KRIS_Bench.zip` — 基准数据（原始图片 + 标注）
- `output_bagel.zip` — 模型生成的编辑图片（6 种语言）

### 3. 解压数据

```bash
unzip KRIS_Bench.zip
unzip output_bagel.zip
```

## 目录结构

解压后的完整目录结构如下：

```
Kris_Bench_output/
├── KRIS_Bench/                          # 基准数据（原始图片 + 标注）
│   ├── color_change/
│   │   ├── annotation.json              # 英文标注
│   │   ├── annotation_zh_ins.json       # 翻译标注（zh/ar/es/ko/yo）
│   │   ├── 1.jpg                        # 原始图片
│   │   └── ...
│   ├── biology/
│   ├── abstract_reasoning/
│   └── ...（共 20 个类别）
├── output_bagel/                        # 模型生成的编辑图片
│   ├── outputs_en/
│   │   └── bagel/
│   │       ├── color_change/
│   │       │   ├── 1.png
│   │       │   └── ...
│   │       └── ...（共 20 个类别）
│   ├── outputs_zh/
│   ├── outputs_ar/
│   ├── outputs_es/
│   ├── outputs_ko/
│   └── outputs_yo/
├── eval_results/                        # 评测结果（由脚本生成）
│   └── bagel/{category}/metrics_{lang}.json
├── metrics_common.py                    # 评测：通用类别
├── metrics_knowledge.py                 # 评测：知识类别
├── metrics_multi_element.py             # 评测：多元素组合
├── metrics_temporal_prediction.py       # 评测：时序预测
├── metrics_view_change.py               # 评测：视角变化
├── eval_bagel.sh                        # 一键运行全部评测
└── utils/
    └── prompts.py                       # GPT-4o 评判提示词模板
```

### 20 个评测类别

| 脚本 | 类别 |
|------|------|
| `metrics_common.py` | count_change, color_change, anomaly_correction, position_movement, size_adjustment, part_completion, multi-instruction_execution |
| `metrics_knowledge.py` | abstract_reasoning, mathematics, practical_knowledge, medicine, rule-based_reasoning, biology, geography, chemistry, humanities, physics |
| `metrics_multi_element.py` | multi-element_composition |
| `metrics_temporal_prediction.py` | temporal_prediction |
| `metrics_view_change.py` | viewpoint_change |

## 环境安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install openai pillow tqdm
```

需要 Python 3.8+。

## API 配置

评测脚本通过 OpenAI API 调用 GPT-4o。设置 API Key：

```bash
export OPENAI_API_KEY="sk-..."
```

**使用第三方兼容 API**（可选）：通过环境变量或命令行参数设置 Base URL：

```bash
# 方式一：环境变量
export OPENAI_BASE_URL="https://your-api-provider.com/v1"

# 方式二：命令行参数（优先级高于环境变量）
python metrics_common.py --base-url "https://your-api-provider.com/v1" ...
```

## 运行评测

### 使用 `eval_bagel.sh` 运行全部评测

```bash
# 评测所有默认语言（en zh ar yo）
./eval_bagel.sh

# 仅评测指定语言
./eval_bagel.sh en zh
```

该脚本会遍历每种语言，依次运行五个评测脚本，自动设置 `--results-dir`、`--output-dir` 和 `--lang` 参数。

### 单独运行某个评测脚本

```bash
# 评测英文的通用类别
python metrics_common.py \
    --models bagel \
    --results-dir output_bagel/outputs_en \
    --output-dir eval_results \
    --lang en

# 评测中文的知识类别
python metrics_knowledge.py \
    --models bagel \
    --results-dir output_bagel/outputs_zh \
    --output-dir eval_results \
    --lang zh

# 使用第三方 API
python metrics_view_change.py \
    --models bagel \
    --results-dir output_bagel/outputs_en \
    --output-dir eval_results \
    --lang en \
    --base-url "https://your-api-provider.com/v1"
```

### 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--models` | 要评测的模型名称（可多个） | `doubao gpt gemini` |
| `--results-dir` | 模型输出所在目录 | `results` |
| `--output-dir` | 评测结果保存目录 | 与 `--results-dir` 相同 |
| `--lang` | 语言代码（`en`/`zh`/`ar`/`es`/`ko`/`yo`），追加到输出文件名 | `None`（保存为 `metrics.json`） |
| `--base-url` | OpenAI API Base URL | `OPENAI_BASE_URL` 环境变量 |

## 评测结果说明

评测结果保存在 `eval_results/<model>/<category>/metrics_<lang>.json`，以图片 ID 为键的 JSON 字典：

```json
{
  "1": {
    "instruction": "Change this green pepper to yellow.",
    "explain": "",
    "consistency_score": 4,
    "consistency_reasoning": "...",
    "instruction_score": 5,
    "instruction_reasoning": "...",
    "quality_score": 4,
    "quality_reasoning": "..."
  }
}
```

### 评分字段

| 字段 | 说明 | 适用脚本 |
|------|------|----------|
| `consistency_score` | 编辑图片对无关内容的保持程度（1–5） | 全部 |
| `instruction_score` | 编辑结果对指令的遵循程度（1–5） | common, multi_element, temporal, view_change |
| `knowledge_score` | 编辑结果是否体现正确的领域知识（1–5） | 仅 knowledge |
| `quality_score` | 编辑后图片的整体质量（1–5） | 全部 |
| `*_reasoning` | GPT-4o 对每项评分的文字解释 | 全部 |

## 许可证

见 [LICENSE](LICENSE)。
