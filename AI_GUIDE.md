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
- `--palette` - 使用调色板模式压缩 PNG（减小文件大小）

---

### 任务 3：放大到非整数倍尺寸

**需求**：将 64×64 放大到 120×120（非 2x/4x 整数倍）

**解决方案**：先超分到更大尺寸，再高质量降采样
```cmd
python scripts/batch_process.py -i <输入> -o <输出> --scale 2 --final-size 120
```

**原理**：64 → 128（AI超分） → 120（Area降采样，抗锯齿平滑）

**可选缩放算法**：`--resize-method` 支持 `area`(默认) / `lanczos` / `nearest` 等

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

### 任务 5：减小输出文件大小

**需求**：AI 输出的 PNG 文件太大（32位 RGBA），需要匹配原始游戏资源的文件大小（8位索引色）

**解决方案**：使用 `--palette` 参数将输出转换为调色板/索引色模式
```cmd
python scripts/batch_process.py -i <输入> -o <输出> --palette
```

**效果**：文件大小减少约 75%（如 30KB → 8KB）

**适用场景**：
- 游戏 UI 图标、按钮等颜色数量有限的素材
- 像素艺术风格的资源
- 需要严格控制文件大小的场景

**注意事项**：
- 仅对 PNG 格式生效
- 调色板最多支持 256 色，渐变丰富的图像可能产生色带
- 可与 `--final-size` 组合使用

---

### 任务 6：处理卡通风格游戏图标（推荐）

**需求**：处理 UI 图标、道具图标、卡通角色头像、Q版立绘等

**解决方案**：使用动漫模型 + Area 降采样（默认）
```cmd
python scripts/batch_process.py -i <输入> -o <输出> ^
    --model RealESRGAN_x4plus_anime_6B ^
    --scale 2 ^
    --final-size 120 ^
    --parallel
```

**参数说明**：
| 参数 | 值 | 说明 |
|------|-----|------|
| `--model` | `RealESRGAN_x4plus_anime_6B` | 动漫/卡通专用模型，边缘锐利 |
| `--scale` | `2` | 2倍放大（64→128） |
| `--final-size` | `120` | 超分后降采样到目标尺寸 |
| `--parallel` | - | 并行处理，加速 I/O |

> 💡 **关键**：使用 `anime_6B` 模型是提升卡通图标清晰度的核心，降采样默认 `area` 算法即可。
>
> ⚠️ **注意**：卡通风格**不要**使用 `--face-enhance`，该功能仅适用于写实人脸照片。

---

### 任务 7：处理像素艺术风格

**需求**：处理复古像素游戏、像素风 UI

**解决方案**：使用最近邻插值保持硬边缘
```cmd
python scripts/batch_process.py -i <输入> -o <输出> ^
    --model RealESRGAN_x4plus_anime_6B ^
    --scale 2 ^
    --resize-method nearest ^
    --parallel
```

**关键点**：`--resize-method nearest` 使用最近邻插值，完全锐利的像素边缘。

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
| `--model` | 模型名称 | 见下表 |
| `--parallel` | 启用并行处理 | 批量时推荐 |
| `--recursive, -r` | 递归处理子目录 | 常用 |
| `--final-size` | 最终输出尺寸 | 如 `120` |
| `--resize-method` | 缩放算法 | `area`(默认推荐) / `nearest`(像素风) |
| `--io-workers` | I/O 线程数 | `4`-`8` |
| `--palette` | 调色板压缩（减小文件大小） | 游戏UI资源 |
| `--face-enhance` | GFPGAN 人脸增强 | 仅限写实人脸 |

### 模型选择指南

| 模型名称 | 适用场景 | 特点 |
|----------|----------|------|
| `RealESRGAN_x4plus` | 写实照片、通用 | 细节丰富，偏写实纹理 |
| `RealESRGAN_x4plus_anime_6B` | **动漫/卡通**（推荐） | 边缘锐利，保持平涂风格 |
| `RealESRGAN_x2plus` | 2倍放大场景 | 原生2倍，质量更好 |
| `RealESRNet_x4plus` | 保守增强 | 细节增强较少，更忠实原图 |

### 缩放算法选择

| 算法 | 特点 | 推荐场景 |
|------|------|----------|
| `area` | **抗锯齿，平滑自然** | **卡通图标（默认推荐）** |
| `lanczos` | 锐利，可能过锐 | 需要极致锐度时 |
| `cubic` | 平滑，柔和 | 照片类 |
| `linear` | 快速 | 对质量要求不高 |
| `nearest` | 完全锐利硬边缘 | **像素艺术** |

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
| 卡通图边缘模糊 | 模型不匹配 | 改用 `RealESRGAN_x4plus_anime_6B` |
| 卡通图变写实风 | 模型风格不匹配 | 改用 `anime_6B` 模型 |
| 边缘过于锐利 | lanczos 过锐 | 使用默认 `area` 算法 |

---

## 给 AI 助手的提示

当用户提出需求时：

1. **先检查任务索引**是否有现成方案
2. **优先使用现有脚本**，避免重复造轮子
3. **修改代码前**先读取相关文件的完整内容
4. **测试命令时**可以用 `--dry-run`（如果支持）或处理单个文件
5. **关键配置**都在 `docs/Deployment_Guide.md` 中有说明

---

## 算法说明

Real-ESRGAN 是**超分辨率重建**算法，属于"增强型"而非"生成型" AI：

| 特性 | 说明 |
|------|------|
| ✅ 忠实于原图 | 不会添加原图中不存在的物体/元素 |
| ✅ 结构保持 | 不会改变构图、移动或删除元素 |
| ⚠️ 纹理推测 | 会根据训练数据推测合理的细节纹理 |
| ⚠️ 风格倾向 | 不同模型的纹理风格不同（写实/动漫） |

**与生成式 AI 的区别**：
- Real-ESRGAN：重建/增强，忠实于原图
- Stable Diffusion：生成/创作，可自由发挥

---

*文档版本：1.2*
*最后更新：2026-01-09*
*适用于：AI 编程助手（Cursor、Copilot、CodeMaker 等）*
