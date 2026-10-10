# Transformer 英译中

这是一个基于 PyTorch 实现的 Transformer 英译中项目，包含双语数据加载、SentencePiece 子词分词、模型训练、BLEU 验证和交互式单句翻译。

## 功能

- 使用 Transformer 编码器—解码器架构进行英译中。
- 使用 SentencePiece BPE 分别处理英文和中文，词表大小均为 32,000。
- 训练时使用 Noam 学习率调度和交叉熵损失。
- 使用 Beam Search 解码，并通过 SacreBLEU 计算验证集 BLEU。
- 支持交互式输入英文句子并输出中文翻译。

## 环境要求

- Python 3.10（项目依赖说明中指定）
- PyTorch
- CUDA 兼容的 NVIDIA GPU（当前训练配置默认使用 `cuda:0`）

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 未包含 PyTorch。请根据本机操作系统和 CUDA 版本，从 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)选择对应命令安装。

## 项目结构

```text
.
├── config.py                 # 模型、训练、设备及文件路径配置
├── main.py                   # 训练入口及验证流程
├── translate.py              # 交互式单句翻译入口
├── web_app.py                # 本地网页和翻译 API 服务入口
├── beam_decoder.py           # Beam Search 解码
├── model/
│   ├── tf_model.py           # Transformer 模型
│   └── train_utils.py        # 损失计算与 Noam 优化器
├── tools/
│   ├── data_loader.py        # 双语数据集、批处理与掩码
│   ├── tokenizer_utils.py    # SentencePiece 加载
│   └── create_exp_folder.py  # 训练输出目录创建
├── data/
│   ├── json/                 # train.json、dev.json、test.json
│   ├── corpus.en             # 英文语料
│   └── corpus.ch             # 中文语料
├── tokenizer/                # 已训练的中英文 SentencePiece 模型与词表
├── web/
│   └── index.html            # 翻译演示网页
├── weights/                  # 项目权重
└── run/train/                # 训练实验及其权重
```

## 数据格式

数据集位于 `data/json/`，每个文件是 JSON 数组，每条数据为英文和中文组成的二元数组：

```json
[
  ["An English sentence.", "对应的中文句子。"]
]
```

当前数据集规模：

| 数据集 | 样本数 |
| --- | ---: |
| 训练集 | 176,943 |
| 验证集 | 25,278 |
| 测试集 | 50,556 |

项目已提供中英文语料文件和分词器模型，通常不需要重新生成。若要从 JSON 数据重新生成语料文件，可在 `data` 目录运行：

```bash
python get_corpus.py
```

重新训练分词器前，请检查 `tokenizer/tokenize.py` 中相对于当前工作目录的输入路径。脚本默认读取 `../data/corpus.en` 和 `../data/corpus.ch`，并在当前目录生成分词器文件；从项目根目录运行时，可先切换到 `tokenizer` 目录再执行：

```bash
cd tokenizer
python tokenize.py
```

## 训练

在项目根目录运行：

```bash
python main.py
```

训练会使用 `config.py` 中的训练集和验证集路径，训练轮数默认为 3。每次训练的权重会写入新的 `run/train/exp*/weights/` 目录，其中 `best_bleu_*.pth` 为验证集 BLEU 最佳权重，`last_bleu_*.pth` 为最后一轮权重。

当前配置默认使用 `cuda:0`。如需更换 GPU，请检查并修改 `config.py` 中的 `gpu_id` 和 `device_id`，并确认 PyTorch 能使用对应设备。训练流程使用 `DataParallel`，按当前实现需要 CUDA 环境。

模型结构、批大小、训练轮数、解码长度和 Beam Search 大小等参数均可在 `config.py` 中调整。

## 单句翻译

在项目根目录运行：

```bash
python translate.py
```

程序会循环提示输入英文句子并打印中文翻译。默认加载的权重路径配置为：

```text
run/train/exp/weights/best_bleu_26.30.pth
```

如需使用其他检查点，请将 `config.py` 中的 `test_model_path` 修改为对应权重文件路径。默认配置使用 `cuda:0`；若要在 CPU 上进行单句翻译，将 `config.py` 中的 `gpu_id` 设为空字符串：

```python
gpu_id = ''
```

## 网页演示

确保配置的模型权重文件和 `tokenizer/` 中的中英文分词器模型存在，然后从项目根目录启动本地服务：

```bash
python web_app.py
```

首次启动会加载模型权重；就绪后在浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。页面调用本地模型进行翻译，不依赖第三方翻译 API，也不需要安装额外的 Web 框架。按 `Ctrl+C` 可停止服务。

默认使用 `config.py` 中的 `test_model_path` 和 `device`。网页服务只绑定本机地址 `127.0.0.1`，默认最多接受 1,000 个字符的句子。

若没有可用 GPU，可在 `config.py` 中将 `gpu_id` 设为空字符串后再启动；CPU 上的推理速度取决于本机性能。

启动前需确保 `test_model_path` 指向的权重文件存在。当前示例检查点约 394 MB；公开到 GitHub 时不建议直接提交该文件，可通过 Git LFS 或 Releases 提供下载，并在运行前放到配置所指向的位置。

## 配置与输出

主要配置均在 `config.py` 中，包括：

- 模型维度：`d_model=512`、`n_heads=8`、`n_layers=6`、`d_ff=2048`
- 训练参数：`batch_size=32`、`epoch_num=3`、`lr=3e-4`
- 解码参数：`max_len=60`、`beam_size=3`
- 数据和检查点路径：`train_data_path`、`dev_data_path`、`test_data_path`、`test_model_path`

训练入口当前执行训练集训练和验证集评估；测试集加载代码已存在，但测试流程默认未启用。
