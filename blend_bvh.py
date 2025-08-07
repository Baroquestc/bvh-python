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
    # bone_info = {
    #     "root_position": pos_idx,
    #     "rot_info": rot_info
    # }
    # # 将骨骼信息写入JSON文件
    # with open('bone_info.json', 'w', encoding='utf-8') as f:
    #     json.dump(bone_info, f, ensure_ascii=False, indent=4)
    # print(f"骨骼信息已保存到 bone_info.json")
    _channel_cache[bvh_id] = result
    return result

def blend_bvh(primary_bvh, secondary_bvh, out_path, t_sec=0.5, fps=60): # 移除了类型提示
    """BVH动作混合主函数"""
    primary_bvh_file_path, primary_frame_range = primary_bvh
    secondary_bvh_file_path, secondary_frame_range = secondary_bvh
    # 加载BVH文件
    load_primary_bvh = get_bvh(primary_bvh_file_path)
    load_secondary_bvh = get_bvh(secondary_bvh_file_path)
    
    # 检查骨架一致性
    if [j.name for j in load_primary_bvh.get_joints()] != [j.name for j in load_secondary_bvh.get_joints()]:
        raise ValueError("骨架不一致")

    if not primary_frame_range:
        primary_frame_range = [0, len(load_primary_bvh.frames) - 1]
    if not secondary_frame_range:
        secondary_frame_range = [0, len(load_secondary_bvh.frames) - 1]

    primary_frame_start, primary_frame_end = primary_frame_range[0], primary_frame_range[1]
    secondary_frame_start, secondary_frame_end = secondary_frame_range[0], secondary_frame_range[1]
    
    primary_frames = load_primary_bvh.frames[primary_frame_start : primary_frame_end+1]
    secondary_frames = load_secondary_bvh.frames[secondary_frame_start : secondary_frame_end+1]
    
    primary_animation_frames = np.array(primary_frames, dtype=float)
    secondary_animation_frames = np.array(secondary_frames, dtype=float)
    print(f"BVH A 帧数: {primary_animation_frames.shape[0]}, BVH B 帧数: {secondary_animation_frames.shape[0]}")

    # 判断frames_a和frames_b shape是否一致
    if primary_animation_frames.shape[1] != secondary_animation_frames.shape[1]:
        print(f"{primary_bvh_file_path}帧通道数: {primary_animation_frames.shape[1]}, {secondary_bvh_file_path}帧通道数: {secondary_animation_frames.shape[1]}")
        raise ValueError("BVH A 和 BVH B 的帧通道数不一致, 无法进行混合。")
    
    # 获取混合帧数
    n_blend = int(round(t_sec * fps))
    # n_blend = int(len(get_bvh_a.frames) - a_end)  # 根据A的帧数和过渡时间计算过渡帧数
    print(f"过渡帧数: {n_blend}, 过渡时间: {t_sec}秒, 帧率: {fps}fps")
    
    if n_blend <= 0:
        print("警告: 过渡时间为0或帧数不足，不生成过渡帧。")
        # 如果没有过渡帧，直接拼接A和B（如果B存在）
        if secondary_animation_frames.size > 0:
            all_frames_np = np.vstack([primary_animation_frames, secondary_animation_frames])
        else:
            all_frames_np = primary_animation_frames
    else:
        # 获取通道索引
        pos_idx, rot_info = get_channels(load_primary_bvh) # pos_idx 是一个包含根节点位置通道索引的列表
        
        # 准备关键帧
        primary_last_frame = primary_animation_frames[-1]  # A尾帧 (1D NumPy array)
        secondary_start_frame = secondary_animation_frames[0]   # B首帧 (1D NumPy array)
        
        # 预分配过渡帧内存
        blend_frames = np.empty((n_blend, len(primary_last_frame)), dtype=float)
        
        # 创建旋转插值器（优化版本）
        interpolators = {}
        for name, (idxs, seq) in rot_info.items():
            eul_a = primary_last_frame[idxs] 
            eul_b = secondary_start_frame[idxs]
            
            # 直接创建四元数，避免重复的quat()调用
            quat_a = R.from_euler(seq, eul_a, degrees=True).as_quat()
            quat_b = R.from_euler(seq, eul_b, degrees=True).as_quat()
            
            # 预计算插值器，避免在循环中重复创建
            slerp_obj = Slerp([0, 1], R.from_quat([quat_a, quat_b]))
            interpolators[name] = (slerp_obj, idxs, seq)

        # 预计算插值时间点和根位移插值
        t_values = np.linspace(1/(n_blend + 1), n_blend/(n_blend + 1), n_blend)
        pos_a = primary_last_frame[pos_idx]
        pos_b = secondary_start_frame[pos_idx]
        
        # 向量化根位移插值预计算 (n_blend x 3)
        pos_interpolated = np.outer(1 - t_values, pos_a) + np.outer(t_values, pos_b)

        # 生成过渡帧（向量化优化版本）
        for i, t in enumerate(t_values):
            current_blend_frame_view = blend_frames[i]
            
            # 复制基础帧
            current_blend_frame_view[:] = primary_last_frame
            
            # 直接赋值预计算的位移插值
            current_blend_frame_view[pos_idx] = pos_interpolated[i]
            
            # 批量处理旋转插值
            for slerp_obj, joint_idxs, joint_seq in interpolators.values():
                # 使用单个标量t而非数组[t]
                current_blend_frame_view[joint_idxs] = slerp_obj(t).as_euler(joint_seq, degrees=True)

        return blend_frames, load_primary_bvh, primary_animation_frames, secondary_animation_frames

        # all_frames_np = np.vstack([frames_a, blend_frames, frames_b])


if __name__ == "__main__":
    idle_bvh = ('./tests/idle.bvh', [0, 1])
    target_bvh = ('./motions/yuliao2_001_v03.bvh', [75, 105])

    out_path = './debug/blend.bvh'
    t_sec = 0.7
    fps = 60
    blend_frames, load_primary_bvh, primary_animation_frames, secondary_animation_frames = blend_bvh(idle_bvh, target_bvh, out_path, t_sec, fps)
    print(f"混合帧数: {blend_frames.shape[0]}")
    total_frames = np.vstack([primary_animation_frames, blend_frames, secondary_animation_frames])
    write_bvh(load_primary_bvh, out_path, total_frames)
