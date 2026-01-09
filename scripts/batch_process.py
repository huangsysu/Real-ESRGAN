#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 批量处理脚本
用于 Jenkins 自动化流水线，处理 PNG 透明通道图片的超分辨率放大

使用方法:
    python batch_process.py -i <输入目录> -o <输出目录> [选项]

示例:
    python batch_process.py -i D:/resources/input -o D:/resources/output
    python batch_process.py -i D:/resources/input -o D:/resources/output --scale 2
    python batch_process.py -i D:/resources/input -o D:/resources/output --recursive
"""

import argparse
import cv2
import glob
import os
import sys
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# 添加项目根目录到 Python 路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class BatchProcessor:
    """批量图片处理器"""

    # 支持的图片格式
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    def __init__(
        self,
        model_name: str = 'RealESRGAN_x4plus',
        scale: int = 4,
        tile: int = 0,
        gpu_id: int = None,
        fp32: bool = True,  # 默认使用 FP32，对透明通道图片更稳定
        final_size: int = None  # 最终输出尺寸（可选，用于超分后降采样）
    ):
        """
        初始化处理器

        Args:
            model_name: 模型名称
            scale: 放大倍数
            tile: 分块大小，0 表示不分块
            gpu_id: GPU ID
            fp32: 是否使用 FP32 精度
            final_size: 最终输出尺寸，如果指定则会在超分后降采样到此尺寸
        """
        self.model_name = model_name
        self.scale = scale
        self.tile = tile
        self.gpu_id = gpu_id
        self.fp32 = fp32
        self.final_size = final_size
        self.upsampler = None

        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
        self.failed_files = []

        # 多线程相关
        self._lock = threading.Lock()
        self._processed_count = 0

    def _init_model(self):
        """初始化模型（延迟加载）"""
        if self.upsampler is not None:
            return

        self._log("正在初始化模型...")

        # 根据模型名称选择网络架构
        if self.model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif self.model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif self.model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        # 模型路径
        model_path = PROJECT_ROOT / 'weights' / f'{self.model_name}.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 创建 upsampler
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(model_path),
            model=model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=0,
            half=not self.fp32,
            gpu_id=self.gpu_id
        )

        self._log(f"模型初始化完成: {self.model_name}")
        self._log(f"  - 网络放大倍数: {netscale}x")
        self._log(f"  - 输出放大倍数: {self.scale}x")
        self._log(f"  - 精度: {'FP32' if self.fp32 else 'FP16'}")
        self._log(f"  - 分块大小: {self.tile if self.tile > 0 else '不分块'}")
        if self.final_size:
            self._log(f"  - 最终尺寸: {self.final_size}x{self.final_size}")

    def _resize_to_final_size(self, img):
        """
        将图像缩放到指定的最终尺寸

        使用 INTER_AREA 进行降采样（最佳抗锯齿效果）
        保持正方形假设，取最大边缩放到 final_size

        Args:
            img: OpenCV 图像 (numpy array)

        Returns:
            缩放后的图像
        """
        if self.final_size is None:
            return img

        h, w = img.shape[:2]

        # 计算缩放比例（基于最大边）
        max_dim = max(h, w)
        if max_dim <= self.final_size:
            # 如果图像已经小于目标尺寸，不做缩放
            return img

        scale_factor = self.final_size / max_dim
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # 使用 INTER_AREA 进行高质量降采样
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    def _log(self, message: str, level: str = 'INFO'):
        """输出日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Windows 控制台兼容处理
        try:
            print(f"[{timestamp}] [{level}] {message}", flush=True)
        except UnicodeEncodeError:
            # 回退到 ASCII 安全输出
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            print(f"[{timestamp}] [{level}] {safe_message}", flush=True)

    def _get_image_files(self, input_path: Path, recursive: bool = False) -> list:
        """获取所有图片文件"""
        files = []

        if input_path.is_file():
            if input_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(input_path)
        elif input_path.is_dir():
            if recursive:
                for ext in self.SUPPORTED_EXTENSIONS:
                    files.extend(input_path.rglob(f'*{ext}'))
                    files.extend(input_path.rglob(f'*{ext.upper()}'))
            else:
                for ext in self.SUPPORTED_EXTENSIONS:
                    files.extend(input_path.glob(f'*{ext}'))
                    files.extend(input_path.glob(f'*{ext.upper()}'))

        # 去重并排序
        files = sorted(set(files))
        return files

    def _process_single_image(self, input_path: Path, output_path: Path) -> bool:
        """
        处理单张图片

        Returns:
            bool: 是否处理成功
        """
        try:
            # 读取图片
            img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                self._log(f"无法读取图片: {input_path}", 'ERROR')
                return False

            # 检测图片模式
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            else:
                img_mode = 'RGB'

            # 执行超分辨率
            output, _ = self.upsampler.enhance(img, outscale=self.scale)

            # 如果指定了最终尺寸，进行降采样
            output = self._resize_to_final_size(output)

            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存结果（RGBA 图片强制保存为 PNG）
            if img_mode == 'RGBA':
                save_path = output_path.with_suffix('.png')
            else:
                save_path = output_path

            cv2.imwrite(str(save_path), output)
            return True

        except Exception as e:
            self._log(f"处理失败 [{input_path}]: {str(e)}", 'ERROR')
            return False

    def _update_stats(self, success: bool, failed_file: str = None):
        """线程安全地更新统计信息"""
        with self._lock:
            self._processed_count += 1
            if success:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
                if failed_file:
                    self.failed_files.append(failed_file)
            return self._processed_count

    def process(
        self,
        input_path: str,
        output_path: str,
        recursive: bool = False,
        keep_structure: bool = True,
        overwrite: bool = False
    ) -> int:
        """
        批量处理图片

        Args:
            input_path: 输入路径（文件或目录）
            output_path: 输出目录
            recursive: 是否递归处理子目录
            keep_structure: 是否保持目录结构
            overwrite: 是否覆盖已存在的文件

        Returns:
            int: 返回码（0=成功，非0=有失败）
        """
        self.stats['start_time'] = time.time()

        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        self._log("=" * 60)
        self._log("Real-ESRGAN 批量处理任务开始")
        self._log("=" * 60)
        self._log(f"输入路径: {input_path}")
        self._log(f"输出路径: {output_path}")
        self._log(f"递归处理: {'是' if recursive else '否'}")
        self._log(f"保持结构: {'是' if keep_structure else '否'}")
        self._log(f"覆盖文件: {'是' if overwrite else '否'}")
        self._log("-" * 60)

        # 检查输入路径
        if not input_path.exists():
            self._log(f"输入路径不存在: {input_path}", 'ERROR')
            return 1

        # 初始化模型
        try:
            self._init_model()
        except Exception as e:
            self._log(f"模型初始化失败: {str(e)}", 'ERROR')
            return 2

        # 获取所有图片文件
        image_files = self._get_image_files(input_path, recursive)
        self.stats['total'] = len(image_files)

        if self.stats['total'] == 0:
            self._log("未找到任何图片文件", 'WARNING')
            return 0

        self._log(f"找到 {self.stats['total']} 个图片文件")
        self._log("-" * 60)

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

        # 处理每个文件
        for idx, img_file in enumerate(image_files, 1):
            # 计算输出路径
            if keep_structure and input_path.is_dir():
                relative_path = img_file.relative_to(input_path)
                out_file = output_path / relative_path
            else:
                out_file = output_path / img_file.name

            # 检查是否跳过
            if not overwrite and out_file.exists():
                self._log(f"[{idx}/{self.stats['total']}] 跳过（已存在）: {img_file.name}")
                self.stats['skipped'] += 1
                continue

            # 处理进度
            progress = idx / self.stats['total'] * 100
            self._log(f"[{idx}/{self.stats['total']}] ({progress:.1f}%) 处理中: {img_file.name}")

            # 处理图片
            if self._process_single_image(img_file, out_file):
                self.stats['success'] += 1
                self._log(f"  ✓ 完成 -> {out_file.name}")
            else:
                self.stats['failed'] += 1
                self.failed_files.append(str(img_file))

        # 输出统计
        self.stats['end_time'] = time.time()
        elapsed = self.stats['end_time'] - self.stats['start_time']

        self._log("-" * 60)
        self._log("处理完成!")
        self._log(f"  总计: {self.stats['total']} 个文件")
        self._log(f"  成功: {self.stats['success']} 个")
        self._log(f"  失败: {self.stats['failed']} 个")
        self._log(f"  跳过: {self.stats['skipped']} 个")
        self._log(f"  耗时: {elapsed:.2f} 秒")

        if self.stats['success'] > 0:
            avg_time = elapsed / self.stats['success']
            self._log(f"  平均: {avg_time:.2f} 秒/张")

        if self.failed_files:
            self._log("-" * 60)
            self._log("失败文件列表:", 'WARNING')
            for f in self.failed_files:
                self._log(f"  - {f}", 'WARNING')

        self._log("=" * 60)

        # 返回状态码
        if self.stats['failed'] > 0:
            return 3  # 部分失败
        return 0  # 全部成功

    def process_parallel(
        self,
        input_path: str,
        output_path: str,
        recursive: bool = False,
        keep_structure: bool = True,
        overwrite: bool = False,
        num_io_workers: int = 4,
        queue_size: int = 10
    ) -> int:
        """
        使用生产者-消费者模式的并行处理

        多线程读取图像 -> 单线程 GPU 推理 -> 多线程保存结果

        这种模式可以：
        - 利用多线程加速 I/O（读取和保存）
        - GPU 推理仍然是串行的（避免显存争抢）
        - 通过队列实现流水线处理，提高整体吞吐量

        Args:
            input_path: 输入路径（文件或目录）
            output_path: 输出目录
            recursive: 是否递归处理子目录
            keep_structure: 是否保持目录结构
            overwrite: 是否覆盖已存在的文件
            num_io_workers: I/O 线程数量
            queue_size: 队列大小（控制内存使用）

        Returns:
            int: 返回码（0=成功，非0=有失败）
        """
        self.stats['start_time'] = time.time()
        self._processed_count = 0

        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        self._log("=" * 60)
        self._log("Real-ESRGAN 批量处理任务开始 [并行模式]")
        self._log("=" * 60)
        self._log(f"输入路径: {input_path}")
        self._log(f"输出路径: {output_path}")
        self._log(f"递归处理: {'是' if recursive else '否'}")
        self._log(f"保持结构: {'是' if keep_structure else '否'}")
        self._log(f"覆盖文件: {'是' if overwrite else '否'}")
        self._log(f"I/O线程数: {num_io_workers}")
        self._log(f"队列大小: {queue_size}")
        self._log("-" * 60)

        # 检查输入路径
        if not input_path.exists():
            self._log(f"输入路径不存在: {input_path}", 'ERROR')
            return 1

        # 初始化模型
        try:
            self._init_model()
        except Exception as e:
            self._log(f"模型初始化失败: {str(e)}", 'ERROR')
            return 2

        # 获取所有图片文件
        image_files = self._get_image_files(input_path, recursive)
        self.stats['total'] = len(image_files)

        if self.stats['total'] == 0:
            self._log("未找到任何图片文件", 'WARNING')
            return 0

        self._log(f"找到 {self.stats['total']} 个图片文件")
        self._log("-" * 60)

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

        # 准备任务列表
        tasks = []
        for img_file in image_files:
            if keep_structure and input_path.is_dir():
                relative_path = img_file.relative_to(input_path)
                out_file = output_path / relative_path
            else:
                out_file = output_path / img_file.name

            # 检查是否跳过
            if not overwrite and out_file.exists():
                self.stats['skipped'] += 1
                continue

            tasks.append((img_file, out_file))

        if not tasks:
            self._log("所有文件都已存在，无需处理")
            return 0

        actual_total = len(tasks)
        self._log(f"需要处理: {actual_total} 个文件 (跳过: {self.stats['skipped']})")

        # 队列定义
        # read_queue: 待读取的任务 (input_path, output_path)
        # process_queue: 已读取的数据 (img_data, input_path, output_path)
        # save_queue: 待保存的结果 (output_data, output_path, is_rgba, input_path)
        read_queue = queue.Queue()
        process_queue = queue.Queue(maxsize=queue_size)
        save_queue = queue.Queue(maxsize=queue_size)

        # 停止信号
        stop_event = threading.Event()
        read_done = threading.Event()
        process_done = threading.Event()

        def reader_worker():
            """读取线程：从磁盘读取图像"""
            while not stop_event.is_set():
                try:
                    task = read_queue.get(timeout=0.1)
                except queue.Empty:
                    if read_done.is_set():
                        break
                    continue

                input_file, output_file = task
                try:
                    img = cv2.imread(str(input_file), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        process_queue.put((img, input_file, output_file))
                    else:
                        self._log(f"无法读取: {input_file.name}", 'ERROR')
                        self._update_stats(False, str(input_file))
                except Exception as e:
                    self._log(f"读取错误 [{input_file.name}]: {e}", 'ERROR')
                    self._update_stats(False, str(input_file))
                finally:
                    read_queue.task_done()

        def processor_worker():
            """GPU 处理线程：执行超分辨率（单线程）"""
            while not stop_event.is_set():
                try:
                    item = process_queue.get(timeout=0.1)
                except queue.Empty:
                    if read_done.is_set() and read_queue.empty() and process_queue.empty():
                        break
                    continue

                img, input_file, output_file = item
                try:
                    # 检测图片模式
                    is_rgba = len(img.shape) == 3 and img.shape[2] == 4

                    # GPU 推理
                    output, _ = self.upsampler.enhance(img, outscale=self.scale)

                    # 如果指定了最终尺寸，进行降采样
                    output = self._resize_to_final_size(output)

                    # 放入保存队列
                    save_queue.put((output, output_file, is_rgba, input_file))
                except Exception as e:
                    self._log(f"处理错误 [{input_file.name}]: {e}", 'ERROR')
                    self._update_stats(False, str(input_file))
                finally:
                    process_queue.task_done()

            process_done.set()

        def saver_worker():
            """保存线程：将结果写入磁盘"""
            while not stop_event.is_set():
                try:
                    item = save_queue.get(timeout=0.1)
                except queue.Empty:
                    if process_done.is_set() and save_queue.empty():
                        break
                    continue

                output_data, output_file, is_rgba, input_file = item
                try:
                    # 确保输出目录存在
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # RGBA 强制保存为 PNG
                    if is_rgba:
                        save_path = output_file.with_suffix('.png')
                    else:
                        save_path = output_file

                    cv2.imwrite(str(save_path), output_data)

                    # 更新统计并打印进度
                    count = self._update_stats(True)
                    progress = count / actual_total * 100
                    self._log(f"[{count}/{actual_total}] ({progress:.1f}%) 完成: {input_file.name}")

                except Exception as e:
                    self._log(f"保存错误 [{output_file.name}]: {e}", 'ERROR')
                    self._update_stats(False, str(input_file))
                finally:
                    save_queue.task_done()

        # 将所有任务放入读取队列
        for task in tasks:
            read_queue.put(task)
        read_done.set()  # 标记任务已全部入队

        # 启动线程
        # 读取线程池
        read_threads = []
        for _ in range(num_io_workers):
            t = threading.Thread(target=reader_worker, daemon=True)
            t.start()
            read_threads.append(t)

        # GPU 处理线程（单线程）
        process_thread = threading.Thread(target=processor_worker, daemon=True)
        process_thread.start()

        # 保存线程池
        save_threads = []
        for _ in range(num_io_workers):
            t = threading.Thread(target=saver_worker, daemon=True)
            t.start()
            save_threads.append(t)

        # 等待所有任务完成
        try:
            # 等待读取完成
            for t in read_threads:
                t.join()

            # 等待处理完成
            process_thread.join()

            # 等待保存完成
            for t in save_threads:
                t.join()

        except KeyboardInterrupt:
            self._log("用户中断，正在停止...", 'WARNING')
            stop_event.set()
            return 4

        # 输出统计
        self.stats['end_time'] = time.time()
        elapsed = self.stats['end_time'] - self.stats['start_time']

        self._log("-" * 60)
        self._log("处理完成!")
        self._log(f"  总计: {self.stats['total']} 个文件")
        self._log(f"  成功: {self.stats['success']} 个")
        self._log(f"  失败: {self.stats['failed']} 个")
        self._log(f"  跳过: {self.stats['skipped']} 个")
        self._log(f"  耗时: {elapsed:.2f} 秒")

        if self.stats['success'] > 0:
            avg_time = elapsed / self.stats['success']
            self._log(f"  平均: {avg_time:.2f} 秒/张")

        if self.failed_files:
            self._log("-" * 60)
            self._log("失败文件列表:", 'WARNING')
            for f in self.failed_files:
                self._log(f"  - {f}", 'WARNING')

        self._log("=" * 60)

        # 返回状态码
        if self.stats['failed'] > 0:
            return 3  # 部分失败
        return 0  # 全部成功

    def process_file_list(
        self,
        file_list: list,
        output_dir: str,
        overwrite: bool = False,
        parallel: bool = False,
        num_io_workers: int = 4,
        queue_size: int = 10
    ) -> int:
        """
        处理指定的文件列表（供外部脚本调用）

        Args:
            file_list: 文件列表，每项为 (input_path, relative_path)
                       - input_path: 输入文件的完整路径
                       - relative_path: 输出时的相对路径（用于保持目录结构）
            output_dir: 输出目录
            overwrite: 是否覆盖已存在的文件
            parallel: 是否使用并行模式
            num_io_workers: I/O 线程数量（并行模式）
            queue_size: 队列大小（并行模式）

        Returns:
            int: 返回码（0=成功，非0=有失败）
        """
        self.stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        self.failed_files = []
        self.stats['start_time'] = time.time()
        self._processed_count = 0

        output_dir = Path(output_dir).resolve()

        # 初始化模型
        try:
            self._init_model()
        except Exception as e:
            self._log(f"模型初始化失败: {str(e)}", 'ERROR')
            return 2

        # 准备任务列表，过滤已存在的文件
        tasks = []
        for input_path, relative_path in file_list:
            input_path = Path(input_path)
            out_file = output_dir / relative_path

            if not overwrite and out_file.exists():
                self.stats['skipped'] += 1
                continue

            tasks.append((input_path, out_file))

        self.stats['total'] = len(file_list)
        actual_total = len(tasks)

        self._log("=" * 60)
        self._log(f"Real-ESRGAN 文件列表处理 [{'并行' if parallel else '串行'}模式]")
        self._log("=" * 60)
        self._log(f"输出目录: {output_dir}")
        self._log(f"文件总数: {self.stats['total']}")
        self._log(f"待处理: {actual_total}")
        self._log(f"已跳过: {self.stats['skipped']}")
        if self.final_size:
            self._log(f"最终尺寸: {self.final_size}×{self.final_size}")
        self._log("-" * 60)

        if not tasks:
            self._log("所有文件都已存在，无需处理")
            return 0

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        if parallel:
            return self._process_tasks_parallel(tasks, actual_total, num_io_workers, queue_size)
        else:
            return self._process_tasks_serial(tasks, actual_total)

    def _process_tasks_serial(self, tasks: list, actual_total: int) -> int:
        """串行处理任务列表"""
        for idx, (input_file, output_file) in enumerate(tasks, 1):
            progress = idx / actual_total * 100
            self._log(f"[{idx}/{actual_total}] ({progress:.1f}%) 处理中: {input_file.name}")

            if self._process_single_image(input_file, output_file):
                self.stats['success'] += 1
                self._log(f"  -> 完成")
            else:
                self.stats['failed'] += 1
                self.failed_files.append(str(input_file))

        self._print_final_stats()
        return 3 if self.stats['failed'] > 0 else 0

    def _process_tasks_parallel(
        self,
        tasks: list,
        actual_total: int,
        num_io_workers: int,
        queue_size: int
    ) -> int:
        """并行处理任务列表（生产者-消费者模式）"""
        self._log(f"启动并行处理: {num_io_workers} 个 I/O 线程")

        # 队列定义
        read_queue = queue.Queue()
        process_queue = queue.Queue(maxsize=queue_size)
        save_queue = queue.Queue(maxsize=queue_size)

        # 停止信号
        stop_event = threading.Event()
        read_done = threading.Event()
        process_done = threading.Event()

        def reader_worker():
            """读取线程"""
            while not stop_event.is_set():
                try:
                    task = read_queue.get(timeout=0.1)
                except queue.Empty:
                    if read_done.is_set():
                        break
                    continue

                input_file, output_file = task
                try:
                    img = cv2.imread(str(input_file), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        process_queue.put((img, input_file, output_file))
                    else:
                        self._log(f"无法读取: {input_file.name}", 'ERROR')
                        self._update_stats(False, str(input_file))
                except Exception as e:
                    self._log(f"读取错误 [{input_file.name}]: {e}", 'ERROR')
                    self._update_stats(False, str(input_file))
                finally:
                    read_queue.task_done()

        def processor_worker():
            """GPU 处理线程（单线程）"""
            while not stop_event.is_set():
                try:
                    item = process_queue.get(timeout=0.1)
                except queue.Empty:
                    if read_done.is_set() and read_queue.empty() and process_queue.empty():
                        break
                    continue

                img, input_file, output_file = item
                try:
                    is_rgba = len(img.shape) == 3 and img.shape[2] == 4
                    output, _ = self.upsampler.enhance(img, outscale=self.scale)
                    output = self._resize_to_final_size(output)
                    save_queue.put((output, output_file, is_rgba, input_file))
                except Exception as e:
                    self._log(f"处理错误 [{input_file.name}]: {e}", 'ERROR')
                    self._update_stats(False, str(input_file))
                finally:
                    process_queue.task_done()

            process_done.set()

        def saver_worker():
            """保存线程"""
            while not stop_event.is_set():
                try:
                    item = save_queue.get(timeout=0.1)
                except queue.Empty:
                    if process_done.is_set() and save_queue.empty():
                        break
                    continue

                output_data, output_file, is_rgba, input_file = item
                try:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    save_path = output_file.with_suffix('.png') if is_rgba else output_file
                    cv2.imwrite(str(save_path), output_data)

                    count = self._update_stats(True)
                    progress = count / actual_total * 100
                    self._log(f"[{count}/{actual_total}] ({progress:.1f}%) 完成: {input_file.name}")
                except Exception as e:
                    self._log(f"保存错误 [{output_file.name}]: {e}", 'ERROR')
                    self._update_stats(False, str(input_file))
                finally:
                    save_queue.task_done()

        # 将任务放入队列
        for task in tasks:
            read_queue.put(task)
        read_done.set()

        # 启动线程
        read_threads = [threading.Thread(target=reader_worker, daemon=True) for _ in range(num_io_workers)]
        for t in read_threads:
            t.start()

        process_thread = threading.Thread(target=processor_worker, daemon=True)
        process_thread.start()

        save_threads = [threading.Thread(target=saver_worker, daemon=True) for _ in range(num_io_workers)]
        for t in save_threads:
            t.start()

        # 等待完成
        try:
            for t in read_threads:
                t.join()
            process_thread.join()
            for t in save_threads:
                t.join()
        except KeyboardInterrupt:
            self._log("用户中断", 'WARNING')
            stop_event.set()
            return 4

        self._print_final_stats()
        return 3 if self.stats['failed'] > 0 else 0

    def _print_final_stats(self):
        """打印最终统计信息"""
        self.stats['end_time'] = time.time()
        elapsed = self.stats['end_time'] - self.stats['start_time']

        self._log("-" * 60)
        self._log("处理完成!")
        self._log(f"  总计: {self.stats['total']} 个文件")
        self._log(f"  成功: {self.stats['success']} 个")
        self._log(f"  失败: {self.stats['failed']} 个")
        self._log(f"  跳过: {self.stats['skipped']} 个")
        self._log(f"  耗时: {elapsed:.2f} 秒")

        if self.stats['success'] > 0:
            avg_time = elapsed / self.stats['success']
            self._log(f"  平均: {avg_time:.2f} 秒/张")

        if self.failed_files:
            self._log("-" * 60)
            self._log("失败文件:", 'WARNING')
            for f in self.failed_files[:20]:
                self._log(f"  - {f}", 'WARNING')
            if len(self.failed_files) > 20:
                self._log(f"  ... 还有 {len(self.failed_files) - 20} 个", 'WARNING')

        self._log("=" * 60)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN 批量处理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个目录
  python batch_process.py -i D:/input -o D:/output

  # 递归处理子目录
  python batch_process.py -i D:/input -o D:/output --recursive

  # 使用动漫模型，2倍放大
  python batch_process.py -i D:/input -o D:/output --model RealESRGAN_x4plus_anime_6B --scale 2

  # 覆盖已存在的文件
  python batch_process.py -i D:/input -o D:/output --overwrite

返回码:
  0 - 全部成功
  1 - 输入路径不存在
  2 - 模型初始化失败
  3 - 部分文件处理失败
        """
    )

    # 必需参数
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入路径（文件或目录）')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='输出目录')

    # 可选参数
    parser.add_argument('--model', type=str, default='RealESRGAN_x4plus',
                        choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus'],
                        help='模型名称 (默认: RealESRGAN_x4plus)')
    parser.add_argument('--scale', type=int, default=4,
                        help='输出放大倍数 (默认: 4)')
    parser.add_argument('--tile', type=int, default=0,
                        help='分块大小，用于大图处理，0表示不分块 (默认: 0)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID (默认: 自动选择)')

    # 处理选项
    parser.add_argument('--recursive', action='store_true',
                        help='递归处理子目录')
    parser.add_argument('--flat', action='store_true',
                        help='输出到同一目录（不保持目录结构）')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的文件')
    parser.add_argument('--fp16', action='store_true',
                        help='使用FP16精度（默认FP32，对透明通道更稳定）')

    # 并行处理选项
    parser.add_argument('--parallel', action='store_true',
                        help='启用并行处理模式（生产者-消费者模式）')
    parser.add_argument('--io-workers', type=int, default=4,
                        help='I/O线程数量 (默认: 4)')
    parser.add_argument('--queue-size', type=int, default=10,
                        help='队列大小，控制内存使用 (默认: 10)')

    # 后处理选项
    parser.add_argument('--final-size', type=int, default=None,
                        help='最终输出尺寸，超分后会降采样到此尺寸 (例如: 64->128->120)')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建处理器
    processor = BatchProcessor(
        model_name=args.model,
        scale=args.scale,
        tile=args.tile,
        gpu_id=args.gpu,
        fp32=not args.fp16,
        final_size=args.final_size
    )

    # 根据模式选择处理方法
    if args.parallel:
        # 并行处理模式
        return_code = processor.process_parallel(
            input_path=args.input,
            output_path=args.output,
            recursive=args.recursive,
            keep_structure=not args.flat,
            overwrite=args.overwrite,
            num_io_workers=args.io_workers,
            queue_size=args.queue_size
        )
    else:
        # 串行处理模式（默认）
        return_code = processor.process(
            input_path=args.input,
            output_path=args.output,
            recursive=args.recursive,
            keep_structure=not args.flat,
            overwrite=args.overwrite
        )

    sys.exit(return_code)


if __name__ == '__main__':
    main()
