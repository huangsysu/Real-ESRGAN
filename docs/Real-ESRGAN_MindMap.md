# Real-ESRGAN 思维导图

```
                                    Real-ESRGAN
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
        📦 核心组件                  🎯 应用场景                 🔧 技术特点
            │                           │                           │
    ┌───────┴───────┐           ┌───────┴───────┐           ┌───────┴───────┐
    │               │           │               │           │               │
  模型架构      推理引擎      图像超分      视频增强      纯合成数据    实用性优化
    │               │           │               │           │               │
┌───┴───┐     ┌─────┴─────┐    ├─真实照片     ├─动漫视频   ├─退化模型      ├─Tile处理
│       │     │           │    ├─动漫插图     └─老旧视频   ├─高阶退化      ├─Alpha通道
RRDBNet SRVGG RealESRGANer │    └─老照片修复               └─Sinc滤波      └─16位图像
│       │     │           │
│       │     ├─pre_process
│       │     ├─process
│       │     ├─tile_process
│       │     └─post_process
│       │
├─RRDB块(23个)
├─64通道特征
├─残差密集连接
└─4x放大

        │
        └─VGG风格紧凑网络
          ├─16个卷积层
          ├─PixelShuffle上采样
          └─残差学习


                        🗂️ 项目结构
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
inference_realesrgan.py  realesrgan/            weights/
(推理入口脚本)               │                    (预训练模型)
                            │
            ┌───────────────┼───────────────┐
            │               │               │
        archs/          models/          utils.py
        (网络架构)        (训练模型)      (RealESRGANer类)
            │               │
    ┌───────┴───────┐   ┌───┴───┐
    │               │   │       │
srvgg_arch.py  discriminator  realesrgan_model
(SRVGGNetCompact)  _arch.py   realesrnet_model


                        📊 可用模型
                            │
    ┌───────────────────────┼───────────────────────────────┐
    │                       │                               │
RealESRGAN_x4plus    RealESRGAN_x4plus_anime_6B    realesr-animevideov3
(通用图像4x放大)          (动漫图像优化)               (动漫视频)
    │                       │                               │
├─RRDBNet 23块          ├─RRDBNet 6块                 ├─SRVGGNetCompact
├─67MB模型大小          ├─17MB模型大小                ├─轻量级模型
└─适合真实照片          └─适合动漫插图                └─适合视频处理
```

## 核心概念说明

### 1. ESRGAN → Real-ESRGAN 的演进
- **ESRGAN**: 增强型超分辨率GAN，使用真实图像对训练
- **Real-ESRGAN**: 使用纯合成数据训练，能处理真实世界的复杂退化

### 2. 网络架构
- **RRDBNet**: 残差密集块网络，23个RRDB块，效果最佳
- **SRVGGNetCompact**: VGG风格紧凑网络，速度更快，适合视频

### 3. 退化模型 (训练时使用)
```
高清图像 → 模糊 → 下采样 → 噪声 → JPEG压缩 → (重复) → 低清图像
         ↓       ↓        ↓         ↓
       高斯/各向异性  双三次/双线性  高斯噪声  随机压缩
```

### 4. 推理处理流程
1. **输入预处理**: 归一化、填充、通道分离
2. **模型推理**: 神经网络超分辨率
3. **后处理**: 裁剪填充、合并通道、缩放调整
