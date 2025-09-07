import numpy as np

class BVHData:
    """
    用于存储解析后的 BVH 数据:
    - hierarchy_lines: BVH 文件中骨架(HIERARCHY)部分的所有行（字符串）
    - frame_count: 动作帧总数
    - frame_time: 每帧时间(1/FPS)
    - motion_data: shape = (frame_count, channel_count)，每帧的通道数据
    """
    def __init__(self):
        self.hierarchy_lines = []
        self.frame_count = 0
        self.frame_time = 0.0
        self.motion_data = None

def read_bvh(file_path):
    """
    读取并解析 BVH 文件
    """
    bvh_data = BVHData()

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 1) 找到 HIERARCHY 与 MOTION 分界
    motion_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("MOTION"):
            motion_idx = i
            break
    
    # 存储骨架部分
    bvh_data.hierarchy_lines = lines[:motion_idx]

    # 剩余部分为 MOTION + 数据
    motion_lines = lines[motion_idx:]
    
    # 2) 解析 MOTION 标头
    # 例如:
    # MOTION
    # Frames: 120
    # Frame Time: 0.0333333
    idx = 0
    while idx < len(motion_lines):
        line = motion_lines[idx].strip()
        if line.upper().startswith("MOTION"):
            idx += 1
            continue
        if line.startswith("Frames:"):
            bvh_data.frame_count = int(line.split(":")[-1].strip())
            idx += 1
            continue
        if line.startswith("Frame Time:"):
            bvh_data.frame_time = float(line.split(":")[-1].strip())
            idx += 1
            break
        idx += 1
    
    # 3) 读取每帧动作通道数据
    motion_values = []
    for i in range(bvh_data.frame_count):
        line = motion_lines[idx + i].strip()
        float_vals = [float(v) for v in line.split()]
        motion_values.append(float_vals)
    
    bvh_data.motion_data = np.array(motion_values)
    
    return bvh_data

def write_bvh(file_path, bvh_data):
    """
    根据 BVHData 对象，将骨架和动作数据写入新的 BVH 文件
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        # 写入 HIERARCHY 部分
        for line in bvh_data.hierarchy_lines:
            f.write(line)
        # 写入 MOTION 部分
        f.write("MOTION\n")
        f.write(f"Frames: {bvh_data.motion_data.shape[0]}\n")
        f.write(f"Frame Time: {bvh_data.frame_time:.6f}\n")
        # 写入合并后的动作数据
        for frame_idx in range(bvh_data.motion_data.shape[0]):
            line_vals = bvh_data.motion_data[frame_idx]
            line_str = " ".join(f"{v:.6f}" for v in line_vals)
            f.write(line_str + "\n")
