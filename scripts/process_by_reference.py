#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据参考目录筛选并处理图片

业务场景：
    REF_DIR 中的 PNG 文件名作为"需要处理"的清单
    在 SOURCE_DIR 中找到同名文件进行超分辨率处理
    输出到 OUTPUT_DIR

使用方法:
    python process_by_reference.py --ref <参考目录> --source <源目录> --output <输出目录>

示例:
    python process_by_reference.py \
        --ref E:\G18\client\cbg\cbg_online\g18cbg_res\res\item \
        --source G:\WorkSpace\Project_duty\Resources\res\item \
        --output D:\output\item
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from batch_process import BatchProcessor


def log(message: str, level: str = 'INFO'):
    """输出日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        print(f"[{timestamp}] [{level}] {message}", flush=True)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', errors='replace').decode('ascii')
        print(f"[{timestamp}] [{level}] {safe_message}", flush=True)


def find_matching_files(ref_dir: Path, source_dir: Path, recursive: bool = False) -> list:
    """
    根据参考目录中的文件名，在源目录中查找匹配的文件

    Args:
        ref_dir: 参考目录（文件名清单来源）
        source_dir: 源目录（实际要处理的文件位置）
        recursive: 是否递归查找

    Returns:
        list: 匹配的源文件路径列表 [(source_file, relative_path), ...]
    """
    # 获取参考目录中所有 PNG 文件名
    # 注意：Windows 上 *.png 和 *.PNG 会匹配相同文件，需要去重
    if recursive:
        ref_files_raw = list(ref_dir.rglob('*.png')) + list(ref_dir.rglob('*.PNG'))
    else:
        ref_files_raw = list(ref_dir.glob('*.png')) + list(ref_dir.glob('*.PNG'))

    # 使用路径字符串去重（Windows 不区分大小写）
    ref_files_seen = set()
    ref_files = []
    for f in ref_files_raw:
        key = str(f).lower()
        if key not in ref_files_seen:
            ref_files_seen.add(key)
            ref_files.append(f)

    # 提取文件名（去重）
    ref_names = set()
    for f in ref_files:
        ref_names.add(f.name.lower())  # 使用小写以忽略大小写

    log(f"参考目录中找到 {len(ref_names)} 个唯一 PNG 文件名")

    # 在源目录中查找匹配的文件
    # 同样需要去重
    if recursive:
        source_files_raw = list(source_dir.rglob('*.png')) + list(source_dir.rglob('*.PNG'))
    else:
        source_files_raw = list(source_dir.glob('*.png')) + list(source_dir.glob('*.PNG'))

    # 使用路径字符串去重
    source_files_seen = set()
    source_files = []
    for f in source_files_raw:
        key = str(f).lower()
        if key not in source_files_seen:
            source_files_seen.add(key)
            source_files.append(f)

    log(f"源目录中共有 {len(source_files)} 个 PNG 文件")

    # 筛选匹配的文件
    matched = []
    for sf in source_files:
        if sf.name.lower() in ref_names:
            # 计算相对路径（用于保持目录结构）
            try:
                relative = sf.relative_to(source_dir)
            except ValueError:
                relative = Path(sf.name)
            matched.append((sf, relative))

    log(f"匹配到 {len(matched)} 个需要处理的文件")

    return matched


def process_matched_files(
    matched_files: list,
    output_dir: Path,
    model_name: str = 'RealESRGAN_x4plus',
    scale: int = 4,
    parallel: bool = False,
    overwrite: bool = False,
    final_size: int = None,
    io_workers: int = 4,
    queue_size: int = 10,
    output_8bit: bool = False,
    use_palette: bool = False,
    face_enhance: bool = False,
    resize_method: str = 'area'
) -> int:
    """
    处理匹配的文件列表

    Args:
        matched_files: [(source_file, relative_path), ...]
        output_dir: 输出目录
        model_name: 模型名称
        scale: 放大倍数
        parallel: 是否使用并行模式
        overwrite: 是否覆盖已存在文件
        final_size: 最终输出尺寸（超分后降采样）
        io_workers: I/O 线程数量（并行模式）
        queue_size: 队列大小（并行模式）
        output_8bit: 是否强制输出为 8-bit 位深度
        use_palette: 是否使用调色板模式输出（减小文件大小）
        face_enhance: 是否使用 GFPGAN 进行人脸增强
        resize_method: 缩放算法 (area/lanczos/cubic/linear/nearest)

    Returns:
        int: 返回码
    """
    if not matched_files:
        log("没有需要处理的文件", 'WARNING')
        return 0

    # 初始化处理器
    processor = BatchProcessor(
        model_name=model_name,
        scale=scale,
        fp32=True,  # 默认使用 FP32
        final_size=final_size,
        output_8bit=output_8bit,
        use_palette=use_palette,
        face_enhance=face_enhance,
        resize_method=resize_method
    )

    # 转换为 process_file_list 所需的格式: [(input_path, relative_path), ...]
    file_list = [(str(source_file), relative_path) for source_file, relative_path in matched_files]

    # 调用 BatchProcessor 的通用文件列表处理方法
    return processor.process_file_list(
        file_list=file_list,
        output_dir=str(output_dir),
        overwrite=overwrite,
        parallel=parallel,
        num_io_workers=io_workers,
        queue_size=queue_size
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='根据参考目录筛选并处理图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python process_by_reference.py --ref E:\\ref\\item --source G:\\source\\item --output D:\\output

  # 递归查找
  python process_by_reference.py --ref E:\\ref --source G:\\source --output D:\\output --recursive

  # 覆盖已存在文件
  python process_by_reference.py --ref E:\\ref --source G:\\source --output D:\\output --overwrite
        """
    )

    # 必需参数
    parser.add_argument('--ref', type=str, required=True,
                        help='参考目录路径（文件名清单来源）')
    parser.add_argument('--source', type=str, required=True,
                        help='源目录路径（实际要处理的文件位置）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录路径')

    # 可选参数
    parser.add_argument('--model', type=str, default='RealESRGAN_x4plus',
                        choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus','RealESRNet_x4plus'],
                        help='模型名称 (默认: RealESRGAN_x4plus)')
    parser.add_argument('--scale', type=int, default=4,
                        help='放大倍数 (默认: 4)')
    parser.add_argument('--recursive', action='store_true',
                        help='递归查找子目录')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的文件')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅列出匹配文件，不执行处理')

    # 并行处理选项
    parser.add_argument('--parallel', action='store_true',
                        help='启用并行处理模式（多线程 I/O）')
    parser.add_argument('--io-workers', type=int, default=4,
                        help='I/O 线程数量 (默认: 4)')
    parser.add_argument('--queue-size', type=int, default=10,
                        help='队列大小 (默认: 10)')

    # 后处理选项
    parser.add_argument('--final-size', type=int, default=None,
                        help='最终输出尺寸，超分后降采样到此尺寸 (例如: --scale 2 --final-size 120)')
    parser.add_argument('--resize-method', type=str, default='area',
                        choices=['area', 'lanczos', 'cubic', 'linear', 'nearest'],
                        help='缩放算法: area(抗锯齿), lanczos(锐利), cubic(平滑), linear, nearest(像素风) (默认: area)')
    parser.add_argument('--8bit', dest='output_8bit', action='store_true',
                        help='强制输出为 8-bit 位深度 PNG（某些游戏引擎需要）')
    parser.add_argument('--palette', action='store_true',
                        help='使用调色板模式输出 PNG（8-bit 索引色，大幅减小文件大小）')

    # 人脸增强选项
    parser.add_argument('--face-enhance', action='store_true',
                        help='使用 GFPGAN 进行人脸增强（需要安装 gfpgan 包）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    ref_dir = Path(args.ref).resolve()
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()

    log("=" * 60)
    log("Real-ESRGAN 按参考目录处理")
    log("=" * 60)
    log(f"参考目录 (REF_DIR)   : {ref_dir}")
    log(f"源目录 (SOURCE_DIR)  : {source_dir}")
    log(f"输出目录 (OUTPUT_DIR): {output_dir}")
    log(f"递归查找: {'是' if args.recursive else '否'}")
    log(f"覆盖文件: {'是' if args.overwrite else '否'}")
    log("-" * 60)

    # 验证目录
    if not ref_dir.exists():
        log(f"参考目录不存在: {ref_dir}", 'ERROR')
        sys.exit(1)

    if not source_dir.exists():
        log(f"源目录不存在: {source_dir}", 'ERROR')
        sys.exit(1)

    # 查找匹配文件
    matched = find_matching_files(ref_dir, source_dir, args.recursive)

    if not matched:
        log("未找到任何匹配的文件", 'WARNING')
        sys.exit(0)

    # Dry run 模式
    if args.dry_run:
        log("-" * 60)
        log("匹配的文件列表（Dry Run 模式，不执行处理）:")
        for source_file, relative_path in matched:
            log(f"  {relative_path}")
        log("-" * 60)
        log(f"共 {len(matched)} 个文件")
        sys.exit(0)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理文件
    return_code = process_matched_files(
        matched_files=matched,
        output_dir=output_dir,
        model_name=args.model,
        scale=args.scale,
        parallel=args.parallel,
        overwrite=args.overwrite,
        final_size=args.final_size,
        io_workers=args.io_workers,
        queue_size=args.queue_size,
        output_8bit=args.output_8bit,
        use_palette=args.palette,
        face_enhance=args.face_enhance,
        resize_method=args.resize_method
    )

    sys.exit(return_code)


if __name__ == '__main__':
    main()
