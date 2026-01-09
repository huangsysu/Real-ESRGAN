# AI 编程助手指南

> 本文档专为 AI 编程助手（如 Cursor、Copilot、CodeMaker）设计，帮助快速理解项目并执行任务。

---

## 项目概述

这是 **Real-ESRGAN** 图像超分辨率工具的企业内部部署版本，用于批量处理游戏资源图片（PNG 透明通道）。

**核心能力**：
- 将低分辨率图像放大 2x/4x，AI 智能补充细节
- 支持透明通道 PNG 图像
- 并行批量处理数千张图片
- 超分后可降采样到任意目标尺寸

---

## 快速任务索引

### 任务 1：批量放大图片

**需求**：将某个目录下的图片放大

**解决方案**：
```cmd
.venv\Scripts\activate
python scripts/batch_process.py -i <输入目录> -o <输出目录> --parallel -r
```

**关键文件**：`scripts/batch_process.py`

---

### 任务 2：按参考目录匹配处理

**需求**：只处理在某个参考目录中存在的文件（跨目录匹配）

**解决方案**：
```cmd
python scripts/process_by_reference.py ^
  --ref <参考目录-包含文件名列表> ^
  --source <源目录-高清原图> ^
  --output <输出目录> ^
  --parallel
```

**关键文件**：`scripts/process_by_reference.py`

**支持参数**：
- `--parallel` - 启用多线程 I/O
- `--io-workers N` - I/O 线程数量
- `--final-size N` - 超分后降采样到指定尺寸
- `--scale N` - 放大倍数
- `--dry-run` - 仅列出匹配文件，不处理

---

### 任务 3：放大到非整数倍尺寸

**需求**：将 64×64 放大到 120×120（非 2x/4x 整数倍）

**解决方案**：先超分到更大尺寸，再高质量降采样
```cmd
python scripts/batch_process.py -i <输入> -o <输出> --scale 2 --final-size 120
```

**原理**：64 → 128（AI超分） → 120（INTER_AREA降采样）

---

### 任务 4：首次部署环境

**需求**：在新机器上安装和配置

**解决方案**：阅读 `docs/Deployment_Guide.md` 的第 2 章

**关键步骤**：
1. 创建虚拟环境：`python -m venv .venv`
2. 安装 PyTorch (CUDA 11.8)
3. 安装依赖：`pip install -e .`
4. 下载模型到 `weights/` 目录

---

## 项目结构

```
Real-ESRGAN/
├── scripts/                    # 🔧 用户脚本（重点关注）
│   ├── batch_process.py        # 核心批处理脚本
│   └── process_by_reference.py # 按参考目录匹配处理
│
├── docs/                       # 📖 文档
│   └── Deployment_Guide.md     # 部署和使用指南（最重要）
│
├── realesrgan/                 # 📦 核心库（一般不需要修改）
│   ├── utils.py                # RealESRGANer 类定义
│   └── archs/                  # 神经网络架构
│
├── weights/                    # 🧠 模型文件
│   └── RealESRGAN_x4plus.pth   # 预训练模型
│
└── .venv/                      # Python 虚拟环境
```

---

## 关键参数速查

### batch_process.py 参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `-i, --input` | 输入路径 | 必填 |
| `-o, --output` | 输出目录 | 必填 |
| `--scale` | 放大倍数 | `2` 或 `4` |
| `--model` | 模型名称 | `RealESRGAN_x4plus` |
| `--parallel` | 启用并行处理 | 批量时推荐 |
| `--recursive, -r` | 递归处理子目录 | 常用 |
| `--final-size` | 最终输出尺寸 | 如 `120` |
| `--io-workers` | I/O 线程数 | `4`-`8` |

---

## 技术约束

1. **必须使用 FP32**：FP16 会导致透明通道图像全黑
2. **NumPy 版本**：必须 `numpy<2`，否则 basicsr ABI 不兼容
3. **PyTorch 版本**：推荐 `torch==2.0.1` + `CUDA 11.8`

---

## 常见问题快速定位

| 问题现象 | 原因 | 解决方案 |
|----------|------|----------|
| 输出图像全黑 | FP16 精度问题 | 确保使用 FP32 |
| NumPy ABI 错误 | NumPy 版本过高 | `pip install "numpy<2"` |
| CUDA 不可用 | PyTorch 版本不匹配 | 重装 CUDA 版 PyTorch |
| 内存不足 OOM | 图片太大或显存不够 | 减小 tile 参数 |

---

## 给 AI 助手的提示

当用户提出需求时：

1. **先检查任务索引**是否有现成方案
2. **优先使用现有脚本**，避免重复造轮子
3. **修改代码前**先读取相关文件的完整内容
4. **测试命令时**可以用 `--dry-run`（如果支持）或处理单个文件
5. **关键配置**都在 `docs/Deployment_Guide.md` 中有说明

---

*文档版本：1.0*
*适用于：AI 编程助手（Cursor、Copilot、CodeMaker 等）*
