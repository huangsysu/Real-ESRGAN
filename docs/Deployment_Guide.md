# Real-ESRGAN 部署指南

> 本文档描述如何在 Windows Patch 机上部署 Real-ESRGAN 图像超分辨率工具。

---

## 目录

1. [系统要求](#1-系统要求)
2. [部署步骤](#2-部署步骤)
3. [使用方法](#3-使用方法)
4. [SVN 工作流程](#4-svn-工作流程)
5. [故障排除](#5-故障排除)
6. [维护更新](#6-维护更新)

---

## 1. 系统要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3060 或更高 |
| 显存 | 4GB | 8GB+ |
| 内存 | 8GB | 16GB+ |
| 磁盘 | 5GB 可用空间 | SSD 推荐 |

### 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Windows | 10/11 或 Server 2019+ | 64位 |
| Python | 3.8 - 3.10 | **推荐 3.10** |
| CUDA | 11.8 | 需与 PyTorch 版本匹配 |
| cuDNN | 8.6+ | 对应 CUDA 11.8 |
| Git | 2.30+ | 用于更新工具 |
| SVN | 1.14+ | 代码管理 |

### 验证 CUDA 安装

```cmd
nvidia-smi
```

应显示 GPU 信息和 CUDA 版本。

---

## 2. 部署步骤

### 2.1 创建部署目录

```cmd
mkdir D:\Tools\Real-ESRGAN
cd D:\Tools\Real-ESRGAN
```

### 2.2 获取项目文件

**方式一：从 Git 克隆（首次部署）**

```cmd
git clone https://github.com/huangsysu/Real-ESRGAN.git .
```

**方式二：从 SVN 检出（后续更新）**

```cmd
svn checkout svn://your-svn-server/tools/Real-ESRGAN .
```

### 2.3 创建 Python 虚拟环境

```cmd
python -m venv .venv
.venv\Scripts\activate
```

> ⚠️ 如果遇到 "禁止运行脚本" 错误，请以管理员身份运行：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2.4 安装依赖

```cmd
:: 升级 pip
python -m pip install --upgrade pip

:: 安装 PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

:: 安装其他依赖
pip install basicsr>=1.4.2
pip install facexlib>=0.2.5
pip install gfpgan>=1.3.5
pip install "numpy<2"
pip install opencv-python
pip install Pillow
pip install tqdm

:: 安装 Real-ESRGAN
pip install -e .
```

### 2.5 下载预训练模型

模型文件需放置在 `weights/` 目录下：

```cmd
mkdir weights
cd weights

:: 下载通用图像模型 (推荐)
curl -L -o RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

:: 下载动漫模型 (可选)
curl -L -o RealESRGAN_x4plus_anime_6B.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

cd ..
```

### 2.6 验证安装

```cmd
:: 激活环境
.venv\Scripts\activate

:: 验证 PyTorch CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

:: 验证 Real-ESRGAN
python -c "from realesrgan import RealESRGANer; print('Real-ESRGAN imported successfully')"
```

---

## 3. 使用方法

### 3.1 批处理脚本

项目提供了 `scripts/batch_process.py` 用于批量处理图像。

#### 基本语法

```cmd
python scripts/batch_process.py -i <输入路径> -o <输出目录> [选项]
```

#### 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入文件或目录路径 | (必填) |
| `--output` | `-o` | 输出目录路径 | (必填) |
| `--model` | `-m` | 模型名称 | `RealESRGAN_x4plus` |
| `--scale` | `-s` | 放大倍数 | `4` |
| `--recursive` | `-r` | 递归处理子目录 | `False` |
| `--overwrite` | | 覆盖已存在的文件 | `False` |
| `--fp16` | | 使用半精度 (省显存) | `False` |
| `--parallel` | | 启用并行处理模式 | `False` |
| `--io-workers` | | I/O 线程数量 | `4` |
| `--queue-size` | | 队列大小（控制内存） | `10` |
| `--final-size` | | 最终输出尺寸（超分后降采样） | `None` |
| `--palette` | | 使用调色板模式压缩 PNG（减小文件大小） | `False` |

#### 使用示例

```cmd
:: 处理单个文件
python scripts/batch_process.py -i D:\res\icon.png -o D:\output

:: 处理整个目录
python scripts/batch_process.py -i D:\res\items -o D:\output

:: 递归处理所有子目录
python scripts/batch_process.py -i D:\res -o D:\output --recursive

:: 使用动漫模型
python scripts/batch_process.py -i D:\res -o D:\output --model RealESRGAN_x4plus_anime_6B

:: 覆盖已有文件
python scripts/batch_process.py -i D:\res -o D:\output --overwrite

:: 启用并行模式（推荐批量处理时使用）
python scripts/batch_process.py -i D:\res -o D:\output --parallel --recursive

:: 并行模式 + 自定义线程数
python scripts/batch_process.py -i D:\res -o D:\output --parallel --io-workers 8
```

### 3.2 并行处理模式

脚本支持 **生产者-消费者模式** 的并行处理，可以显著提升批量处理的效率。

#### 工作原理

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   读取线程池     │ ──→ │   GPU推理线程    │ ──→ │   保存线程池     │
│  (多线程 I/O)   │     │   (单线程)       │     │  (多线程 I/O)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- **读取线程池**：多线程从磁盘读取图像
- **GPU 推理线程**：单线程执行神经网络推理（GPU 资源串行使用）
- **保存线程池**：多线程将结果写入磁盘

#### 适用场景

| 场景 | 推荐模式 |
|------|----------|
| 处理少量图片（< 10 张） | 串行模式（默认） |
| 批量处理大量图片 | **并行模式** `--parallel` |
| 图片来自机械硬盘/网络存储 | **并行模式** `--parallel` |
| 需要调试问题 | 串行模式（日志更清晰） |

#### 注意事项

- GPU 推理仍是单线程，并行模式主要优化 I/O 等待时间
- 如果 GPU 利用率已接近 100%，并行模式提升有限
- `--queue-size` 控制内存使用，大图片时可适当减小

### 3.3 超分后降采样（--final-size）

当目标尺寸不是原图的整数倍放大时，可以使用 `--final-size` 参数实现 **"超分 → 降采样"** 的高质量缩放。

#### 使用场景

例如：将 64×64 的图像放大到 120×120

| 方法 | 流程 | 效果 |
|------|------|------|
| ❌ 直接插值 | 64 → 120 | 模糊、锯齿明显 |
| ✅ 超分+降采样 | 64 → 128 → 120 | 清晰、细节丰富 |

#### 工作原理

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  原图    │ ──→ │  AI 超分辨率  │ ──→ │ 高质量降采样  │
│  64×64   │     │   128×128    │     │   120×120    │
└──────────┘     └──────────────┘     └──────────────┘
                  (Real-ESRGAN)        (cv2.INTER_AREA)
```

1. **AI 超分**：Real-ESRGAN 智能补充纹理、锐化边缘、去除伪影
2. **降采样**：使用 `INTER_AREA` 算法（抗锯齿效果最佳）缩小到目标尺寸

#### 使用示例

```cmd
:: 64×64 → 2x超分 → 降采样到 120×120
python scripts/batch_process.py -i D:\res -o D:\output --model RealESRGAN_x2plus --scale 2 --final-size 120

:: 使用 4x 模型但只放大 2 倍，然后降采样到 120
python scripts/batch_process.py -i D:\res -o D:\output --scale 2 --final-size 120

:: 结合并行模式批量处理
python scripts/batch_process.py -i D:\res -o D:\output --parallel --scale 2 --final-size 120
```

#### 参数选择建议

| 原图尺寸 | 目标尺寸 | 推荐配置 |
|----------|----------|----------|
| 64×64 | 120×120 | `--scale 2 --final-size 120` |
| 64×64 | 200×200 | `--scale 4 --final-size 200` |
| 128×128 | 300×300 | `--scale 4 --final-size 300` |
| 任意 | 保持原比例 | 只指定 `--final-size`，会基于最大边缩放 |

#### 技术细节

- **缩放策略**：基于图像最大边计算缩放比例，保持宽高比
- **插值算法**：使用 `cv2.INTER_AREA`，专门优化降采样场景
- **透明通道**：完整支持 RGBA 图像，Alpha 通道同样被处理
- **跳过保护**：如果超分后尺寸已小于 `final-size`，则不进行降采样

### 3.4 按参考目录处理（process_by_reference.py）

当需要根据一个目录的文件名列表，从另一个目录筛选并处理文件时，使用此脚本。

#### 业务场景

```
REF_DIR (参考目录)          SOURCE_DIR (源目录)           OUTPUT_DIR (输出)
├── icon_001.png      →    ├── icon_001.png (高清)   →   ├── icon_001.png
├── icon_002.png      →    ├── icon_002.png (高清)   →   ├── icon_002.png
└── icon_003.png      →    ├── icon_003.png (高清)   →   └── icon_003.png
                           ├── icon_004.png (不处理)
                           └── icon_005.png (不处理)
```

#### 基本语法

```cmd
python scripts/process_by_reference.py --ref <参考目录> --source <源目录> --output <输出目录> [选项]
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ref` | 参考目录（文件名清单来源） | (必填) |
| `--source` | 源目录（实际处理的文件位置） | (必填) |
| `--output` | 输出目录 | (必填) |
| `--model` | 模型名称 | `RealESRGAN_x4plus` |
| `--scale` | 放大倍数 | `4` |
| `--recursive` | 递归查找子目录 | `False` |
| `--overwrite` | 覆盖已存在文件 | `False` |
| `--dry-run` | 仅列出匹配文件，不处理 | `False` |
| `--parallel` | 启用并行处理模式 | `False` |
| `--io-workers` | I/O 线程数量 | `4` |
| `--final-size` | 最终输出尺寸（超分后降采样） | `None` |
| `--palette` | 使用调色板模式压缩 PNG（减小文件大小） | `False` |

#### 使用示例

```cmd
:: 查看匹配文件（不执行处理）
python scripts/process_by_reference.py --ref E:\ref\item --source G:\source\item --output D:\output --dry-run

:: 基本处理
python scripts/process_by_reference.py --ref E:\ref\item --source G:\source\item --output D:\output

:: 并行处理 + 超分降采样（64→128→120）
python scripts/process_by_reference.py --ref E:\ref\item --source G:\source\item --output D:\output --scale 2 --final-size 120 --parallel

:: 完整示例（8线程 + 递归 + 覆盖）
python scripts/process_by_reference.py --ref E:\ref --source G:\source --output D:\output --scale 2 --final-size 120 --parallel --io-workers 8 --recursive --overwrite
```

### 3.5 调色板模式压缩（--palette）

当需要减小输出 PNG 文件大小时，可以使用 `--palette` 参数将 32 位 RGBA 图像转换为 8 位索引色/调色板模式。

#### 适用场景

- **游戏 UI 资源**：图标、按钮等颜色数量有限的素材
- **像素艺术**：原始素材就是低色彩数的图像
- **文件大小敏感**：需要匹配原始游戏资源的文件大小

#### 工作原理

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   AI 超分辨率     │ ──→ │  自适应调色板     │ ──→ │   8位索引色PNG    │
│  32位 RGBA 输出  │     │  256色量化       │     │  文件大小约 1/4   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                          (PIL.Image.ADAPTIVE)
```

1. **超分处理**：Real-ESRGAN 输出 32 位 RGBA 图像（约 30KB）
2. **调色板量化**：使用 PIL 的自适应算法提取最优 256 色调色板
3. **保存压缩**：输出 8 位索引色 PNG（约 8KB，减小约 75%）

#### 使用示例

```cmd
:: 超分 + 调色板压缩
python scripts/batch_process.py -i D:\res -o D:\output --palette

:: 完整流程：超分 → 降采样 → 调色板压缩
python scripts/batch_process.py -i D:\res -o D:\output --scale 2 --final-size 120 --palette

:: 结合并行模式
python scripts/batch_process.py -i D:\res -o D:\output --parallel --palette

:: process_by_reference.py 同样支持
python scripts/process_by_reference.py --ref E:\ref --source G:\source --output D:\output --palette
```

#### 注意事项

- 仅对 **PNG 格式** 生效，其他格式会忽略此选项
- 调色板模式最多支持 **256 色**，对渐变丰富的图像可能产生色带
- 透明通道会被保留，但精度降低为 1 位（完全透明/完全不透明）
- 对于 UI 图标、像素艺术等素材效果极佳；对于照片类图像不推荐

#### 文件大小对比示例

| 文件 | 原始（32位 RGBA） | 调色板模式（8位） | 压缩率 |
|------|-------------------|-------------------|--------|
| skillicon_001.png | 28 KB | 7 KB | 75% |
| item_icon.png | 32 KB | 9 KB | 72% |
| button_bg.png | 45 KB | 12 KB | 73% |

### 3.6 返回码

脚本执行后会返回状态码：

| 返回码 | 含义 |
|--------|------|
| `0` | 全部成功 |
| `1` | 输入路径不存在 |
| `2` | 模型初始化失败 |
| `3` | 部分文件处理失败 |

### 3.4 输出日志示例

```
================================================================================
Real-ESRGAN Batch Processor
================================================================================
Input path  : D:\res\items
Output path : D:\output
Model       : RealESRGAN_x4plus
Scale       : 4x
Recursive   : True
--------------------------------------------------------------------------------
Initializing model...
Model loaded successfully
Found 15 images to process
--------------------------------------------------------------------------------
[  1/15 ]   6.7% | Processing: items\icon_001.png ... [OK]
[  2/15 ]  13.3% | Processing: items\icon_002.png ... [OK]
...
[ 15/15 ] 100.0% | Processing: items\icon_015.png ... [OK]
================================================================================
Processing Complete!
  Total    : 15
  Success  : 15
  Failed   : 0
  Skipped  : 0
================================================================================
```

---

## 4. SVN 工作流程

### 4.1 目录结构

```
SVN Repository
├── /tools/Real-ESRGAN/          # 工具代码
│   ├── scripts/
│   ├── realesrgan/
│   ├── weights/                  # 模型文件
│   └── ...
│
├── /project/resources/
│   ├── res/                      # 原始资源 (SOURCE_RES_PATH)
│   │   ├── items/
│   │   ├── icons/
│   │   └── ...
│   │
│   └── res_upscaled/             # 放大后资源 (OUTPUT_PATH)
│       ├── items/
│       ├── icons/
│       └── ...
```

### 4.2 忽略文件配置

在 SVN 中设置忽略规则：

```
# 忽略虚拟环境和缓存
.venv/
__pycache__/
*.pyc
*.pyo

# 忽略临时输出
results/
experiments/
```

---

## 5. 故障排除

### 5.1 常见问题

#### CUDA 不可用

**症状**：`torch.cuda.is_available()` 返回 `False`

**解决方案**：
1. 确认已安装 NVIDIA 驱动：`nvidia-smi`
2. 确认 CUDA 版本匹配：
   ```cmd
   nvcc --version
   ```
3. 重新安装 PyTorch：
   ```cmd
   pip uninstall torch torchvision
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

#### 输出图像全黑

**症状**：处理后的图像是纯黑色

**原因**：FP16 精度在处理小图像或特殊透明通道时可能失败

**解决方案**：
- 确保 `batch_process.py` 使用 FP32（默认已配置）
- 如果手动调用，不要使用 `--fp16` 参数

#### 内存不足 (OOM)

**症状**：`CUDA out of memory`

**解决方案**：
1. 减小 tile 大小（修改 `batch_process.py` 中的 `tile` 参数）
2. 使用 FP16 节省显存（但可能影响透明通道处理）
3. 关闭其他占用 GPU 的程序

#### NumPy ABI 错误

**症状**：`RuntimeError: module compiled against ABI version 0x1000009`

**解决方案**：
```cmd
pip install "numpy<2"
```

#### 模块导入错误

**症状**：`ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`

**解决方案**：
```cmd
pip install torchvision==0.15.2
```

### 5.2 日志收集

如需提交问题报告，请收集以下信息：

```cmd
:: 系统信息
systeminfo | findstr /B /C:"OS"

:: GPU 信息
nvidia-smi

:: Python 环境
python --version
pip list | findstr -i "torch numpy basicsr realesrgan"

:: 错误日志
python scripts/batch_process.py -i test -o output 2>&1
```

---

## 6. 维护更新

### 6.1 定期维护

| 任务 | 频率 | 说明 |
|------|------|------|
| 检查磁盘空间 | 每周 | 清理临时文件和旧输出 |
| 更新 GPU 驱动 | 每季度 | 确保兼容性 |
| 检查工具更新 | 按需 | 从 Git 拉取最新版本 |

### 6.2 备份策略

建议备份以下内容：

- `weights/` - 模型文件（约 200MB）
- `scripts/batch_process.py` - 自定义脚本
- `.venv/` - 可选，或记录 `pip freeze > requirements.lock`

### 6.3 性能优化建议

1. **使用 SSD**：显著提升图像读写速度
2. **批量处理**：一次性处理多个文件比逐个处理更高效
3. **合理的 tile 大小**：根据 GPU 显存调整，通常 400-512 较佳
4. **避免过小图像**：极小的图像（如 < 32x32）可能处理不稳定

---

## 附录

### A. 快速命令参考

```cmd
:: 激活环境
D:\Tools\Real-ESRGAN\.venv\Scripts\activate

:: 处理单个目录
python scripts/batch_process.py -i D:\res\items -o D:\output

:: 递归处理
python scripts/batch_process.py -i D:\res -o D:\output -r

:: 使用动漫模型
python scripts/batch_process.py -i D:\res -o D:\output -m RealESRGAN_x4plus_anime_6B

:: 并行模式批量处理（推荐）
python scripts/batch_process.py -i D:\res -o D:\output --parallel -r

:: 并行模式 + 8 个 I/O 线程
python scripts/batch_process.py -i D:\res -o D:\output --parallel --io-workers 8 -r

:: 超分后降采样到指定尺寸 (64→128→120)
python scripts/batch_process.py -i D:\res -o D:\output --scale 2 --final-size 120

:: 并行 + 超分降采样 完整示例
python scripts/batch_process.py -i D:\res -o D:\output --parallel --scale 2 --final-size 120 -r

:: 使用调色板模式压缩（减小文件大小）
python scripts/batch_process.py -i D:\res -o D:\output --palette

:: 完整流程：超分 + 降采样 + 调色板压缩
python scripts/batch_process.py -i D:\res -o D:\output --parallel --scale 2 --final-size 120 --palette -r

:: ====== process_by_reference.py 命令 ======

:: 查看匹配文件（不执行）
python scripts/process_by_reference.py --ref E:\ref --source G:\source --output D:\output --dry-run

:: 按参考目录处理（并行 + 超分降采样）
python scripts/process_by_reference.py --ref E:\ref --source G:\source --output D:\output --scale 2 --final-size 120 --parallel

:: 完整命令（8线程 + 递归）
python scripts/process_by_reference.py --ref E:\ref --source G:\source --output D:\output --scale 2 --final-size 120 --parallel --io-workers 8 --recursive

:: 使用调色板模式压缩
python scripts/process_by_reference.py --ref E:\ref --source G:\source --output D:\output --scale 2 --final-size 120 --palette --parallel
```

### B. 支持的模型

| 模型名称 | 用途 | 放大倍数 |
|----------|------|----------|
| `RealESRGAN_x4plus` | 通用图像 | 4x |
| `RealESRGAN_x4plus_anime_6B` | 动漫图像 | 4x |
| `RealESRGAN_x2plus` | 通用图像 | 2x |
| `realesr-animevideov3` | 动漫视频 | 4x |

---

*文档版本：1.4*
*最后更新：2026-01-09*
