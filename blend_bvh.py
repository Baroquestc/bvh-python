# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from bvh import Bvh

# 全局缓存
_bvh_cache = {}
_channel_cache = {}

def get_bvh(path): # 移除了类型提示
    """加载BVH文件（使用缓存）"""
    # path 已经是 Path 对象，使用 resolve() 获取绝对路径以确保缓存键的一致性
    path_str = str(Path(path).resolve()) # 确保 path 是 Path 对象
    if path_str in _bvh_cache:
        return _bvh_cache[path_str]
        
    for enc in ['utf-8', 'gbk', 'latin1']:
        try:
            bvh = Bvh(Path(path).read_text(encoding=enc))
            _bvh_cache[path_str] = bvh
            return bvh
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码: {path}")

def write_bvh(bvh, out_path, frames):
    """将BVH对象写入文件"""
    header_parts = bvh.as_bvh().split('MOTION')
    motion_header = f"MOTION\nFrames: {len(frames)}\nFrame Time: {bvh.frame_time}\n"
    frames_text = "\n".join(" ".join(f"{v:.6f}" for v in frame) for frame in frames)
    bvh_text = header_parts[0] + motion_header + frames_text
    Path(out_path).write_text(bvh_text, encoding='utf-8')
    print(f"混合完成，输出文件: {out_path}")

def get_channels(bvh):
    """获取通道索引信息（使用缓存）"""
    bvh_id = id(bvh)
    if bvh_id in _channel_cache:
        return _channel_cache[bvh_id]
    
    # 获取根节点位移索引
    root = next(bvh.root.filter('ROOT')).name
    pos_idx = [bvh.get_joint_channel_index(root, f"{ax}position") for ax in "XYZ"]
    
    # 获取关节旋转信息
    rot_info = {}
    for joint in bvh.get_joints():
        name = joint.name
        channels = bvh.joint_channels(name)
        rot_chs = [c for c in channels if "rotation" in c.lower()]
        if rot_chs:
            base = bvh.get_joint_channels_index(name)
            idxs = [base + bvh.get_joint_channel_index(name, c) for c in rot_chs]
            # 保存旋转序列和索引
            seq = "".join(c[0].upper() for c in rot_chs)
            rot_info[name] = (idxs, seq)
    
    result = (pos_idx, rot_info)
    _channel_cache[bvh_id] = result
    return result

def blend_bvh(bvh_a, bvh_b, out_path, t_sec=0.5, fps=60): # 移除了类型提示
    """BVH动作混合主函数"""
    bvh_a_path, a_range = bvh_a
    bvh_b_path, b_range = bvh_b
    # 加载BVH文件
    get_bvh_a = get_bvh(bvh_a_path)
    get_bvh_b = get_bvh(bvh_b_path)
    
    # 检查骨架一致性
    if [j.name for j in get_bvh_a.get_joints()] != [j.name for j in get_bvh_b.get_joints()]:
        raise ValueError("骨架不一致")

    if not a_range:
        a_range = [0, len(get_bvh_a.frames) - 1]
    if not b_range:
        b_range = [0, len(get_bvh_b.frames) - 1]

    a_start, a_end = a_range[0], a_range[1]
    b_start, b_end = b_range[0], b_range[1]
    
    frames_a_str = get_bvh_a.frames[a_start : a_end+1]
    frames_b_str = get_bvh_b.frames[b_start : b_end+1]
    
    frames_a = np.array(frames_a_str, dtype=float)
    frames_b = np.array(frames_b_str, dtype=float)
    print(f"BVH A 帧数: {frames_a.shape[0]}, BVH B 帧数: {frames_b.shape[0]}")

    # 判断frames_a和frames_b shape是否一致
    if frames_a.shape[1] != frames_b.shape[1]:
        print(f"{bvh_a_path}帧通道数: {frames_a.shape[1]}, {bvh_b_path}帧通道数: {frames_b.shape[1]}")
        raise ValueError("BVH A 和 BVH B 的帧通道数不一致, 无法进行混合。")
    
    # 获取混合帧数
    n_blend = int(round(t_sec * fps))
    # n_blend = int(len(get_bvh_a.frames) - a_end)  # 根据A的帧数和过渡时间计算过渡帧数
    print(f"过渡帧数: {n_blend}, 过渡时间: {t_sec}秒, 帧率: {fps}fps")
    
    if n_blend <= 0:
        print("警告: 过渡时间为0或帧数不足，不生成过渡帧。")
        # 如果没有过渡帧，直接拼接A和B（如果B存在）
        if frames_b.size > 0:
            all_frames_np = np.vstack([frames_a, frames_b])
        else:
            all_frames_np = frames_a
    else:
        # 获取通道索引
        pos_idx, rot_info = get_channels(get_bvh_a) # pos_idx 是一个包含根节点位置通道索引的列表
        
        # 准备关键帧
        frame_a_last = frames_a[-1]  # A尾帧 (1D NumPy array)
        frame_b_first = frames_b[0]   # B首帧 (1D NumPy array)
        
        # 预分配过渡帧内存
        blend_frames = np.empty((n_blend, len(frame_a_last)), dtype=float)
        
        # 创建旋转插值器
        interpolators = {}
        for name, (idxs, seq) in rot_info.items():
            eul_a = frame_a_last[idxs] 
            eul_b = frame_b_first[idxs]
            
            quat_a = R.from_euler(seq, eul_a, degrees=True)
            quat_b = R.from_euler(seq, eul_b, degrees=True)
            
            interpolators[name] = (
                Slerp([0, 1], R.from_quat([quat_a.as_quat(), quat_b.as_quat()])),
                idxs, # idxs 是一个包含特定关节旋转通道索引的列表
                seq
            )
        
        # 生成过渡帧
        for i in range(n_blend):
            t = (i + 1) / (n_blend + 1)  # 插值因子
            current_blend_frame_view = blend_frames[i] # 获取当前过渡帧的视图
            
            # 将A的最后一帧作为当前过渡帧的基础
            current_blend_frame_view[:] = frame_a_last
            
            # 根位移线性插值 (向量化)
            current_blend_frame_view[pos_idx] = (1-t) * frame_a_last[pos_idx] + t * frame_b_first[pos_idx]
            
            # 关节旋转球面插值 (向量化赋值)
            for name, (slerp_obj, joint_idxs, joint_seq) in interpolators.items():
                current_blend_frame_view[joint_idxs] = slerp_obj([t])[0].as_euler(joint_seq, degrees=True)

        return blend_frames, get_bvh_a, frames_a, frames_b

if __name__ == "__main__":
    idle_bvh = ('./tests/idle.bvh', [0, 1])
    target_bvh = ('./motions/yuliao2_001_v03.bvh', [75, 105])

    out_path = './debug/blend.bvh'
    t_sec = 0.7
    fps = 60
    blend_frames, load_primary_bvh, primary_animation_frames, secondary_animation_frames = blend_bvh(idle_bvh, target_bvh, out_path, t_sec, fps)
    print(f"混合帧数: {blend_frames.shape[0]}")
    total_frames = np.vstack([primary_animation_frames, blend_frames, secondary_animation_frames])
    print(f"总帧数: {total_frames.shape[0]}")
    # 将混合帧写入BVH文件
    write_bvh(load_primary_bvh, out_path, total_frames)
