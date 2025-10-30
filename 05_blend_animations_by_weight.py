#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
from pathlib import Path
from bvh import Bvh
from scipy.spatial.transform import Rotation as R, Slerp
from typing import Tuple, Optional
import warnings

# 忽略万向锁警告
warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)

# 全局缓存
_bvh_cache = {}
_channel_cache = {}

def get_bvh(path):
    """加载BVH文件（使用缓存）"""
    path_str = str(Path(path).resolve())
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
            seq = "".join(c[0].upper() for c in rot_chs)
            rot_info[name] = (idxs, seq)
    
    result = (pos_idx, rot_info)
    _channel_cache[bvh_id] = result
    return result

def align_animation_length(frames_a, frames_b, mode='loop'):
    """
    对齐两个动画的长度
    参数:
        frames_a: 第一个动画的帧数据 (numpy array)
        frames_b: 第二个动画的帧数据 (numpy array)
        mode: 对齐模式 'loop'(循环重复) 或 'stretch'(时间拉伸)
    返回:
        对齐后的两个动画帧数据
    """
    len_a, len_b = len(frames_a), len(frames_b)
    
    if len_a == len_b:
        return frames_a, frames_b
    
    # target_len = min(len_a, len_b)
    target_len = len_a

    
    def align_frames(frames, target_length, align_mode):
        current_len = len(frames)
        if current_len == target_length:
            return frames
            
        if align_mode == 'loop':
            # 循环重复模式
            repeat_count = target_length // current_len
            remainder = target_length % current_len
            
            if repeat_count > 0:
                repeated = np.tile(frames, (repeat_count, 1))
                if remainder > 0:
                    partial = frames[:remainder]
                    result = np.vstack([repeated, partial])
                else:
                    result = repeated
            else:
                result = frames[:target_length]
                
        elif align_mode == 'stretch':
            # 时间拉伸模式 - 使用线性插值
            old_indices = np.linspace(0, current_len - 1, current_len)
            new_indices = np.linspace(0, current_len - 1, target_length)
            result = np.zeros((target_length, frames.shape[1]))
            
            for i in range(frames.shape[1]):
                result[:, i] = np.interp(new_indices, old_indices, frames[:, i])
        else:
            raise ValueError(f"不支持的对齐模式: {align_mode}")
            
        return result
    
    aligned_a = align_frames(frames_a, target_len, mode)
    aligned_b = align_frames(frames_b, target_len, mode)
    
    return aligned_a, aligned_b

def blend_animations_by_weight(
    anim_a_path,
    anim_b_path, 
    output_path,
    weight_a=0.5,
    weight_b=0.5,
    a_range=None,
    b_range=None,
    align_mode='loop'
):
    """
    按权重混合两个BVH动画
    
    参数:
        anim_a_path: 第一个动画文件路径
        anim_b_path: 第二个动画文件路径 (通常是idle动画)
        output_path: 输出文件路径
        weight_a: 动画A的权重 (0.0-1.0)
        weight_b: 动画B的权重 (0.0-1.0)
        a_range: 动画A的帧范围 [start, end] 或 None
        b_range: 动画B的帧范围 [start, end] 或 None  
        align_mode: 长度对齐模式 'loop' 或 'stretch'
    """
    
    # 权重归一化
    total_weight = weight_a + weight_b
    if total_weight == 0:
        raise ValueError("权重总和不能为0")
    
    weight_a = weight_a / total_weight
    weight_b = weight_b / total_weight
    
    print(f"混合权重: A={weight_a:.3f}, B={weight_b:.3f}")
    
    # 加载BVH文件
    bvh_a = get_bvh(anim_a_path)
    bvh_b = get_bvh(anim_b_path)
    
    # 检查骨架一致性
    joints_a = [j.name for j in bvh_a.get_joints()]
    joints_b = [j.name for j in bvh_b.get_joints()]
    if joints_a != joints_b:
        raise ValueError("骨架结构不一致，无法混合")
    
    # 提取指定范围的帧
    if a_range is None:
        a_start, a_end = 0, bvh_a.nframes - 1
    else:
        a_start, a_end = a_range[0], a_range[1]
        
    if b_range is None:
        b_start, b_end = 0, bvh_b.nframes - 1
    else:
        b_start, b_end = b_range[0], b_range[1]
    
    # 转换为numpy数组
    frames_a_str = bvh_a.frames[a_start:a_end+1]
    frames_b_str = bvh_b.frames[b_start:b_end+1]
    
    frames_a = np.array(frames_a_str, dtype=float)
    frames_b = np.array(frames_b_str, dtype=float)
    
    print(f"原始长度: A={len(frames_a)}帧, B={len(frames_b)}帧")
    
    # 对齐动画长度
    frames_a_aligned, frames_b_aligned = align_animation_length(
        frames_a, frames_b, mode=align_mode
    )
    
    print(f"对齐后长度: A={len(frames_a_aligned)}帧, B={len(frames_b_aligned)}帧")
    
    # 获取通道索引信息
    pos_idx, rot_info = get_channels(bvh_a)
    
    # 预分配混合结果
    num_frames = len(frames_a_aligned)
    blended_frames = np.zeros_like(frames_a_aligned)
    
    print(f"开始混合 {num_frames} 帧...")
    
    # 逐帧混合
    for frame_idx in range(num_frames):
        frame_a = frames_a_aligned[frame_idx]
        frame_b = frames_b_aligned[frame_idx]
        blended_frame = blended_frames[frame_idx]
        
        # 根节点位移线性混合
        blended_frame[pos_idx] = (
            weight_a * frame_a[pos_idx] + 
            weight_b * frame_b[pos_idx]
        )
        
        # 关节旋转球面插值混合
        for joint_name, (idxs, seq) in rot_info.items():
            # 提取欧拉角
            eul_a = frame_a[idxs]
            eul_b = frame_b[idxs]
            
            # 转换为四元数
            quat_a = R.from_euler(seq, eul_a, degrees=True)
            quat_b = R.from_euler(seq, eul_b, degrees=True)
            
            # 球面线性插值
            if weight_a == 1.0:
                blended_quat = quat_a
            elif weight_b == 1.0:
                blended_quat = quat_b
            else:
                # 使用Slerp进行四元数插值
                slerp = Slerp([0, 1], R.from_quat([quat_a.as_quat(), quat_b.as_quat()]))
                blended_quat = slerp([weight_b])[0]  # weight_b作为插值参数
            
            # 转换回欧拉角
            blended_euler = blended_quat.as_euler(seq, degrees=True)
            blended_frame[idxs] = blended_euler
        
        # 处理其他通道(如果有)
        all_processed_idx = set(pos_idx + [idx for idxs, _ in rot_info.values() for idx in idxs])
        for i in range(len(frame_a)):
            if i not in all_processed_idx:
                blended_frame[i] = weight_a * frame_a[i] + weight_b * frame_b[i]
    
    # 生成输出BVH文件
    header_parts = bvh_a.as_bvh().split('MOTION')
    motion_header = f"MOTION\nFrames: {num_frames}\nFrame Time: {bvh_a.frame_time}\n"
    
    # 格式化帧数据
    frames_text = "\n".join(
        " ".join(f"{v:.6f}" for v in frame) 
        for frame in blended_frames
    )
    
    bvh_text = header_parts[0] + motion_header + frames_text + "\n"
    
    # 写入文件
    Path(output_path).write_text(bvh_text, encoding='utf-8')
    
    print(f"✓ 混合完成！输出文件: {output_path}")
    print(f"  总帧数: {num_frames}")
    print(f"  时长: {num_frames * bvh_a.frame_time:.2f}秒")
    
    return blended_frames

def blend_with_varying_weights(
    anim_a_path,
    anim_b_path,
    output_path, 
    weight_curve=None,
    align_mode='loop',
    a_range=None,
    b_range=None
):
    """
    使用变化权重曲线混合动画
    
    参数:
        weight_curve: 权重曲线函数或数组，返回每帧的weight_a值 (0.0-1.0)
                     如果是函数，接收参数 (frame_index, total_frames)
                     如果是数组，长度应该等于目标帧数
    """
    
    # 加载和对齐动画 (复用之前的代码)
    bvh_a = get_bvh(anim_a_path)
    bvh_b = get_bvh(anim_b_path)
    
    # 检查骨架一致性
    if [j.name for j in bvh_a.get_joints()] != [j.name for j in bvh_b.get_joints()]:
        raise ValueError("骨架结构不一致")
    
    # 提取帧数据
    if a_range is None:
        frames_a = np.array(bvh_a.frames, dtype=float)
    else:
        frames_a = np.array(bvh_a.frames[a_range[0]:a_range[1]+1], dtype=float)
        
    if b_range is None:
        frames_b = np.array(bvh_b.frames, dtype=float)
    else:
        frames_b = np.array(bvh_b.frames[b_range[0]:b_range[1]+1], dtype=float)
    
    # 对齐长度
    frames_a_aligned, frames_b_aligned = align_animation_length(
        frames_a, frames_b, mode=align_mode
    )
    
    num_frames = len(frames_a_aligned)
    
    # 生成权重曲线
    if weight_curve is None:
        # 默认权重曲线：从1.0渐变到0.0再回到1.0 (先A后B再A)
        weights_a = np.concatenate([
            np.linspace(1.0, 0.0, num_frames//3),
            np.full(num_frames//3, 0.0), 
            np.linspace(0.0, 1.0, num_frames - 2*(num_frames//3))
        ])
    elif callable(weight_curve):
        # 权重曲线是函数
        weights_a = np.array([
            weight_curve(i, num_frames) for i in range(num_frames)
        ])
    else:
        # 权重曲线是数组
        weights_a = np.array(weight_curve)
        if len(weights_a) != num_frames:
            # 插值调整长度
            old_indices = np.linspace(0, len(weights_a)-1, len(weights_a))
            new_indices = np.linspace(0, len(weights_a)-1, num_frames)
            weights_a = np.interp(new_indices, old_indices, weights_a)
    
    # 确保权重在合理范围内
    weights_a = np.clip(weights_a, 0.0, 1.0)
    weights_b = 1.0 - weights_a
    
    print(f"使用变化权重混合 {num_frames} 帧")
    print(f"权重A范围: {weights_a.min():.3f} - {weights_a.max():.3f}")
    
    # 获取通道信息
    pos_idx, rot_info = get_channels(bvh_a)
    blended_frames = np.zeros_like(frames_a_aligned)
    
    # 逐帧混合（使用不同权重）
    for frame_idx in range(num_frames):
        weight_a = weights_a[frame_idx]
        weight_b = weights_b[frame_idx]
        
        frame_a = frames_a_aligned[frame_idx]
        frame_b = frames_b_aligned[frame_idx]
        blended_frame = blended_frames[frame_idx]
        
        # 位移线性混合
        blended_frame[pos_idx] = weight_a * frame_a[pos_idx] + weight_b * frame_b[pos_idx]
        
        # 旋转球面插值
        for joint_name, (idxs, seq) in rot_info.items():
            eul_a = frame_a[idxs]
            eul_b = frame_b[idxs]
            
            quat_a = R.from_euler(seq, eul_a, degrees=True)
            quat_b = R.from_euler(seq, eul_b, degrees=True)
            
            if weight_a >= 0.999:
                blended_quat = quat_a
            elif weight_b >= 0.999:
                blended_quat = quat_b
            else:
                slerp = Slerp([0, 1], R.from_quat([quat_a.as_quat(), quat_b.as_quat()]))
                blended_quat = slerp([weight_b])[0]
            
            blended_frame[idxs] = blended_quat.as_euler(seq, degrees=True)
        
        # 其他通道线性混合
        all_processed_idx = set(pos_idx + [idx for idxs, _ in rot_info.values() for idx in idxs])
        for i in range(len(frame_a)):
            if i not in all_processed_idx:
                blended_frame[i] = weight_a * frame_a[i] + weight_b * frame_b[i]
    
    # 输出文件
    header_parts = bvh_a.as_bvh().split('MOTION')
    motion_header = f"MOTION\nFrames: {num_frames}\nFrame Time: {bvh_a.frame_time}\n"
    frames_text = "\n".join(" ".join(f"{v:.6f}" for v in frame) for frame in blended_frames)
    bvh_text = header_parts[0] + motion_header + frames_text + "\n"
    
    Path(output_path).write_text(bvh_text, encoding='utf-8')
    print(f"✓ 变权重混合完成！输出: {output_path}")
    
    return blended_frames, weights_a

def load_res_json(path, csv_file):

    """加载BVH文件的JSON格式"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        motions = data.get('motions', [])
        for motion in motions:
            # 处理每个动作的细节
            match_file_name = motion.get('match_file_name')
            keyframe_start = motion.get('keyframe_start')
            keyframe_end = motion.get('keyframe_end')

            yield {
                'match_file_name': match_file_name,
                'keyframe_start': keyframe_start,
                'keyframe_end': keyframe_end
            }
def load_interval_csv(path):
    """加载BVH文件的CSV格式"""
    df = pd.read_csv(path)

# 使用示例
if __name__ == "__main__":
    # 示例1: 固定权重混合
    print("=== 示例1: 固定权重混合 ===")
    # test.wav
    # # 1.yulu1-002(01)
    # blend_animations_by_weight(
    #     anim_a_path='./motions/yulu1-002(01).bvh',
    #     anim_b_path='./motions/baohuo.bvh',  # idle动画
    #     output_path='./blended_yulu1-002(01).bvh',
    #     weight_a=0.850,  # 主动画权重70%
    #     weight_b=1-0.850,  # idle动画权重30%
    #     align_mode='loop',  # idle动画循环填充
    #     a_range=[0, 115]  # 主动画范围
    # )

    # 2.yuliao2_001_v03
    # blend_animations_by_weight(
    #     anim_a_path='./motions/yuliao2_001_v03.bvh',
    #     anim_b_path='./motions/baohuo.bvh',  # idle动画
    #     output_path='./debug/test_volume_-12db/blended_yuliao2_001_v03.bvh',
    #     weight_a=0.673,  # 主动画权重70%
    #     weight_b=1-0.673,  # idle动画权重30%
    #     align_mode='loop',  # idle动画循环填充
    #     a_range=[75, 105]  # 主动画范围
    # )

    # # 3.yulu14(1)
    # blend_animations_by_weight(
    #     anim_a_path='./motions/yulu14(1).bvh',
    #     anim_b_path='./motions/baohuo.bvh',  # idle动画
    #     output_path='./blended_yulu14(1).bvh',
    #     weight_a=0.9,  # 主动画权重70%
    #     weight_b=0.1,  # idle动画权重30%
    #     align_mode='loop',  # idle动画循环填充
    #     a_range=[30, 285]  # 主动画范围
    # )

    # # 4.yulu1-002(04)
    # blend_animations_by_weight(
    #     anim_a_path='./motions/yulu1-002(04).bvh',
    #     anim_b_path='./motions/baohuo.bvh',  # idle动画
    #     output_path='./debug/test_volume_-12db/blended_yulu1-002(04).bvh',
    #     weight_a=0.812,  # 主动画权重70%
    #     weight_b=1-0.812,  # idle动画权重30%
    #     align_mode='loop',  # idle动画循环填充
    #     a_range=[15, 180]  # 主动画范围
    # )


# -12db
    # 1.yulu1-002(01)
    blend_animations_by_weight(
        anim_a_path='./motions/yulu1-002(01).bvh',
        anim_b_path='./motions/baohuo.bvh',  # idle动画
        output_path='./blended_yulu1-002(01).bvh',
        weight_a=0.6,  # 主动画权重70%
        weight_b=1-0.6,  # idle动画权重30%
        align_mode='loop',  # idle动画循环填充
        a_range=[0, 115]  # 主动画范围
    )

    # 2.yuliao2_001_v03
    blend_animations_by_weight(
        anim_a_path='./motions/yuliao2_001_v03.bvh',
        anim_b_path='./motions/baohuo.bvh',  # idle动画
        output_path='./blended_yuliao2_001_v03.bvh',
        weight_a=0.514,  # 主动画权重70%
        weight_b=1-0.514,  # idle动画权重30%
        align_mode='loop',  # idle动画循环填充
        a_range=[75, 105]  # 主动画范围
    )

    # 3.yulu14(1)
    blend_animations_by_weight(
        anim_a_path='./motions/yulu14(1).bvh',
        anim_b_path='./motions/baohuo.bvh',  # idle动画
        output_path='./blended_yulu14(1).bvh',
        weight_a=0.6,  # 主动画权重70%
        weight_b=0.4,  # idle动画权重30%
        align_mode='loop',  # idle动画循环填充
        a_range=[30, 285]  # 主动画范围
    )

    # 4.yulu1-002(04)
    blend_animations_by_weight(
        anim_a_path='./motions/yulu1-002(04).bvh',
        anim_b_path='./motions/baohuo.bvh',  # idle动画
        output_path='./blended_yulu1-002(04).bvh',
        weight_a=0.6,  # 主动画权重70%
        weight_b=1-0.4,  # idle动画权重30%
        align_mode='loop',  # idle动画循环填充
        a_range=[15, 180]  # 主动画范围
    )

    # # 示例2: 变化权重混合
    # print("\n=== 示例2: 变化权重混合 ===")
    
    # # 定义权重曲线函数：从主动画渐变到idle再回到主动画
    # def weight_curve_func(frame_idx, total_frames):
    #     # 创建一个正弦波权重曲线
    #     t = frame_idx / total_frames
    #     return 0.5 + 0.5 * np.cos(2 * np.pi * t)
    
    # blend_with_varying_weights(
    #     anim_a_path='./motions/yulu1-002(01).bvh',
    #     anim_b_path='./motions/baohuo.bvh',
    #     output_path='./blended_varying_weight.bvh',
    #     weight_curve=weight_curve_func,
    #     align_mode='loop'
    # )
    
    # # 示例3: 自定义权重数组
    # print("\n=== 示例3: 自定义权重数组 ===")
    
    # # 创建一个渐变权重：开始100%主动画，结束100%idle动画
    # custom_weights = np.linspace(1.0, 0.0, 100)  # 100帧的渐变
    
    # blend_with_varying_weights(
    #     anim_a_path='./motions/yulu1-002(01).bvh',
    #     anim_b_path='./motions/idle.bvh', 
    #     output_path='./blended_custom_weight.bvh',
    #     weight_curve=custom_weights,
    #     align_mode='stretch'  # 拉伸对齐模式
    # )
