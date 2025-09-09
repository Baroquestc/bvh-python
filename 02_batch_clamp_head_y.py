# -*- coding: utf-8 -*-
"""  
批量约束 BVH 中 Head 的 Yrotation ∈ [‑14, 0]°  
基于第三方库  bvh‑python（pip install bvh）  
"""  
import sys 
import argparse  
from pathlib import Path  

from utils.bvh import Bvh             
from tqdm import tqdm
from utils.helpers import logger             

JOINT_NAME  = "head"  
# JOINT_NAME  = "pelvis"  
CHANNEL     = "Yrotation"  
LOW, HIGH   = -14.0, 0.0            # 约束区间（度）  

# -------------------------------------------------------------  

def clamp(v: float, lo: float, hi: float) -> float:  
    """数值裁剪"""  
    return max(lo, min(hi, v))  

def process_one(src: Path, dst: Path) -> None:  
    """  
    读取 BVH → 裁剪 → 写回  
    """  
    # --- 读取并解析 ---  
    text = src.read_text(encoding="utf-8", errors="ignore")
    mocap = Bvh(text)

    # --- 获取目标通道索引 ---  
    try:  
        joint_index = mocap.get_joint_channels_index(JOINT_NAME)  
        channel_index = mocap.get_joint_channel_index(JOINT_NAME, CHANNEL)
        ch_idx = joint_index + channel_index
    except ValueError:  
        print(f"[跳过] {src.relative_to(src.parents[1])}  无 {JOINT_NAME}.{CHANNEL}")  
        return  

    # --- 遍历帧并裁剪 ---  
    for idx, frame in enumerate(mocap.frames): 
        mocap.frames[idx][ch_idx] = f"{clamp(float(frame[ch_idx]), LOW, HIGH):.6f}"

    # --- 写入文件：使用文本分割法 ---  
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(mocap.as_bvh(), encoding="utf-8")  # 使用 bvh 库的 as_bvh 方法
    
    print(f"[完成] {src.name}")

def main(args):  
    log = logger(args.logfile)
    log.info("python {} {}".format(__file__, " ".join([f"--{k} {v}" for k, v in vars(args).items()])))

    src_dir, dst_dir = Path(args.src), Path(args.dst)  
    if not src_dir.is_dir():  
        sys.exit(f"输入目录不存在: {src_dir}")  

    bvh_files = list(src_dir.rglob(f"*.{args.ext}"))  
    if not bvh_files:  
        sys.exit("未找到任何 BVH 文件")  

    for src_file in tqdm(bvh_files, desc="处理 BVH", unit="file"):  
        rel = src_file.relative_to(src_dir)  
        dst_file = dst_dir / rel  
        process_one(src_file, dst_file)  

    print(f"\n全部完成，共处理 {len(bvh_files)} 个文件，输出至 {dst_dir.resolve()}")  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="批量约束 BVH Head.Yrotation")
    parser.add_argument("-log", "--logfile", default="./log/02_batch_clamp_head_y.log", type=str, help="log file")  
    parser.add_argument("--src", required=True, help="输入目录")  
    parser.add_argument("--dst", required=True, help="输出目录（自动创建）")  
    parser.add_argument("--ext", default="bvh", help="扩展名过滤，默认 bvh")  
    args = parser.parse_args() 
    main(args)
