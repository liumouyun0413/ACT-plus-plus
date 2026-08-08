"""
分析 aligned_joints.h5 数据结构，并将结果保存到 check_h5_result.txt
用法: python check_h5.py [episode_dir]
"""

import h5py
import numpy as np
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
EPISODE_DIRS = [
    BASE_DIR / 'sorting_block_20260416101728',
    BASE_DIR / 'sorting_block_20260416104222',
    BASE_DIR / 'sorting_block_20260416110535',
]
OUTPUT_FILE = BASE_DIR / 'check_h5_result.txt'


def analyze_h5(h5_path: Path, out_lines: list):
    def w(s=''):
        out_lines.append(s)
        print(s)

    w(f'\n{"=" * 70}')
    w(f'文件: {h5_path}')
    w(f'{"=" * 70}')

    if not h5_path.exists():
        w(f'  ❌ 文件不存在')
        return

    with h5py.File(str(h5_path), 'r') as f:
        # ── 属性 ──
        if f.attrs:
            w('\n[文件属性]')
            for k, v in f.attrs.items():
                w(f'  {k}: {v}')

        # ── 目录树 ──
        w('\n[数据集树形结构]')
        def print_tree(name, obj):
            indent = '  ' + '  ' * name.count('/')
            if isinstance(obj, h5py.Dataset):
                w(f'{indent}📊 {name.split("/")[-1]}: shape={obj.shape}, dtype={obj.dtype}')
            else:
                w(f'{indent}📁 {name.split("/")[-1]}/')
        f.visititems(print_tree)

        # ── 数值预览 ──
        w('\n[数值预览]')
        def preview(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            data = obj[()]
            if data.dtype.kind in ('f', 'i', 'u') and data.size > 0:
                w(f'  {name}:')
                w(f'    shape={data.shape}, dtype={data.dtype}')
                w(f'    min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}')
                if data.ndim == 2 and data.shape[1] <= 32:
                    w(f'    第0帧: {np.round(data[0], 4).tolist()}')
                    if data.shape[0] > 1:
                        w(f'    第1帧: {np.round(data[1], 4).tolist()}')
                elif data.ndim == 1:
                    w(f'    前10: {np.round(data[:10], 4).tolist()}')
        f.visititems(preview)


def analyze_meta(meta_path: Path, out_lines: list):
    def w(s=''):
        out_lines.append(s)
        print(s)

    if not meta_path.exists():
        return
    w(f'\n[meta_info.json]')
    with open(meta_path) as f:
        meta = json.load(f)
    for k, v in meta.items():
        w(f'  {k}: {v}')


def analyze_camera_txt(ep_dir: Path, out_lines: list):
    def w(s=''):
        out_lines.append(s)
        print(s)

    cam_dir = ep_dir / 'camera'
    if not cam_dir.exists():
        return
    w(f'\n[相机时间戳文件 (.txt) 预览]')
    for cam in sorted(cam_dir.iterdir()):
        txt = cam / f'{cam.name}.txt'
        if txt.exists():
            lines = txt.read_text().strip().splitlines()
            w(f'  {cam.name}: {len(lines)} 帧, 前3行: {lines[:3]}')


def main():
    out_lines = []

    for ep_dir in EPISODE_DIRS:
        h5_path = ep_dir / 'record' / 'aligned_joints.h5'
        meta_path = ep_dir / 'meta_info.json'

        analyze_meta(meta_path, out_lines)
        analyze_h5(h5_path, out_lines)
        analyze_camera_txt(ep_dir, out_lines)

    # 保存结果
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f'\n\n✅ 结果已保存至: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
