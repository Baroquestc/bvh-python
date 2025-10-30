# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from utils.bvh import Bvh

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

def smooth_interpolation_curve(t, curve_type='ease_in_out'):
    """应用不同的插值曲线"""
    if curve_type == 'ease_in_out':
        # 平滑的S曲线
        return 0.5 * (1 - np.cos(t * np.pi))
    elif curve_type == 'ease_out':
        # 快速开始，缓慢结束
        return 1 - (1 - t) ** 2
    elif curve_type == 'ease_in':
        # 缓慢开始，快速结束
        return t ** 2
    else:
        return t  # 线性

def calculate_gesture_motion_velocity(frames, rot_info, key_joint_patterns):
    """专门计算手势动作的运动速度（不包括根节点位置）"""
    if len(frames) < 2:
        return 0.0
    
    frame_count = len(frames) - 1
    
    # 只计算关键关节（手臂相关）的旋转速度
    gesture_velocities = []
    for name, (idxs, seq) in rot_info.items():
        is_gesture_joint = any(pattern in name for pattern in key_joint_patterns)
        if not is_gesture_joint:
            continue
            
        joint_velocities = []
        for i in range(frame_count):
            eul_a = frames[i][idxs]
            eul_b = frames[i+1][idxs]
            
            # 计算角度差异
            angle_diff = np.abs(eul_b - eul_a)
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            joint_velocity = np.sum(angle_diff)
            joint_velocities.append(joint_velocity)
        
        if joint_velocities:
            avg_joint_velocity = np.mean(joint_velocities)
            gesture_velocities.append(avg_joint_velocity)
            # print(f"  关节 {name}: 平均角速度 {avg_joint_velocity:.2f}°/帧")
    
    # 返回所有手势关节的平均角速度
    return np.mean(gesture_velocities) if gesture_velocities else 0.0

def calculate_velocity_optimized_blend_frames(primary_frames, secondary_frames, rot_info, pos_idx, fps=60):
    """专门针对数字人播报场景优化的混合帧数计算"""
    
    # 定义关键关节模式（主要是手臂相关）
    key_joint_patterns = ['pelvis', 'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05', 'clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r']

    # 计算前后动画的手势运动速度（专门针对手臂动作）
    primary_velocity = calculate_gesture_motion_velocity(primary_frames[-5:], rot_info, key_joint_patterns) if len(primary_frames) >= 5 else 0.0
    secondary_velocity = calculate_gesture_motion_velocity(secondary_frames[:10], rot_info, key_joint_patterns) if len(secondary_frames) >= 2 else 0.0
    
    print(f"主动画末尾速度: {primary_velocity:.3f}, 目标动画开头速度: {secondary_velocity:.3f}")
    
    # 计算姿态差异
    primary_last_frame = primary_frames[-1]
    secondary_start_frame = secondary_frames[0]
    
    # 计算关键关节（手势相关）的角度差异
    important_angle_diffs = []
    gesture_angle_diffs = []  # 专门记录手势关节的角度差异
    
    for name, (idxs, seq) in rot_info.items():
        is_key_joint = any(pattern in name for pattern in key_joint_patterns)
        
        eul_a = primary_last_frame[idxs]
        eul_b = secondary_start_frame[idxs]
        
        # 处理角度环绕
        angle_diff = np.abs(eul_b - eul_a)
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        joint_max_diff = np.max(angle_diff)
        
        if joint_max_diff > 3:  # 只考虑有意义的变化
            important_angle_diffs.append(joint_max_diff)
            
            # 如果是手势相关关节，单独记录
            if is_key_joint:
                gesture_angle_diffs.append(joint_max_diff)
    
    # 优先使用手势关节的角度差异，如果没有则使用所有关节
    if gesture_angle_diffs:
        avg_angle_diff = np.mean(gesture_angle_diffs)
        max_angle_diff = np.max(gesture_angle_diffs)
        print(f"手势关节差异: 平均{avg_angle_diff:.1f}°, 最大{max_angle_diff:.1f}°")
    else:
        avg_angle_diff = np.mean(important_angle_diffs) if important_angle_diffs else 0
        max_angle_diff = np.max(important_angle_diffs) if important_angle_diffs else 0
        print(f"所有关节差异: 平均{avg_angle_diff:.1f}°, 最大{max_angle_diff:.1f}°")
    
    # 基于手势动画速度和角度差异动态计算混合时间
    # 目标：让过渡速度与前后手势动画速度相匹配
    
    if primary_velocity == 0 and secondary_velocity == 0:
        # 两个都是静态姿态，基于角度差异计算理想过渡速度
        # 对于数字人播报，手势过渡应该相对较快以保持自然感
        ideal_gesture_speed = 80.0  # 度/秒，比一般过渡稍快
        blend_time = max_angle_diff / ideal_gesture_speed
        print(f"静态手势过渡: 角度差异{max_angle_diff:.1f}°, 理想速度{ideal_gesture_speed}°/s")
    else:
        # 动态计算：基于前后手势动画的实际速度
        if primary_velocity == 0:
            # 从静态开始到手势动作，需要快速启动但平滑
            target_speed = secondary_velocity * 0.8  # 目标速度的80%作为过渡速度
        elif secondary_velocity == 0:
            # 从手势动作到静态，需要平滑停止
            target_speed = primary_velocity * 0.6   # 当前速度的60%作为过渡速度
        else:
            # 两个都是手势动作，使用智能加权平均
            # 对于手势切换，倾向于使用较快的速度以保持连贯性
            min_speed = min(primary_velocity, secondary_velocity)
            max_speed = max(primary_velocity, secondary_velocity)
            target_speed = min_speed * 0.3 + max_speed * 0.7  # 偏向较快的动作
        
        # 确保手势过渡速度在合理范围内（针对播报场景优化）
        target_speed = max(20.0, min(target_speed, 150.0))  # 20-150度/秒，比一般动作稍快
        
        # 基于目标速度和角度差异计算时间
        blend_time = avg_angle_diff / target_speed
        
        # 根据速度差异调整时间 - 对于手势，速度差异大的情况需要更平滑的过渡
        speed_diff_ratio = abs(primary_velocity - secondary_velocity) / max(primary_velocity + secondary_velocity, 1.0)
        speed_diff_factor = 1.0 + speed_diff_ratio * 0.2  # 减少调整幅度，保持手势的敏捷性
        blend_time *= speed_diff_factor
        
        print(f"手势动态过渡: 主速度{primary_velocity:.1f}°/s, 目标速度{secondary_velocity:.1f}°/s")
        print(f"计算的过渡速度: {target_speed:.1f}°/s, 速度差异因子: {speed_diff_factor:.2f}")
    
    # 基于数字人播报场景的手势特征调整
    if avg_angle_diff > 45:  # 大幅度手势切换
        blend_time *= 1.2    # 适当延长以确保平滑，但不要太长
    elif avg_angle_diff > 20:  # 中等幅度手势
        blend_time *= 1.0    # 标准时间
    elif avg_angle_diff < 8:   # 微小手势调整
        blend_time *= 0.7    # 快速过渡，保持手势敏捷性
    
    # 额外的播报场景优化
    # 如果是从快速手势到慢速手势，需要更多时间来减速
    if primary_velocity > secondary_velocity + 10:
        blend_time *= 1.1
    
    # 转换为帧数
    n_blend = int(blend_time * fps)
    
    # 限制在合理范围内（数字人播报场景）
    n_blend = max(10, min(n_blend, 50))  # 0.17-0.83秒
    
    print(f"播报优化混合帧数: {n_blend} ({n_blend/fps:.2f}秒)")

    return n_blend

def blend_bvh(primary_bvh, secondary_bvh, out_path, t_sec=0.5, fps=60, adaptive_mode='gesture', curve_type='ease_in_out'):
    """BVH动作混合主函数 - 针对数字人播报场景优化"""
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
    

    if adaptive_mode == 'velocity':
        # 使用播报优化算法
        pos_idx, rot_info = get_channels(load_primary_bvh)
        n_blend = calculate_velocity_optimized_blend_frames(primary_animation_frames, secondary_animation_frames, rot_info, pos_idx, fps)
    else:
        n_blend = int(round(t_sec * fps))
    
    print(f"过渡帧数: {n_blend}, 过渡时间: {n_blend/fps:.2f}秒, 帧率: {fps}fps, 模式: {adaptive_mode}, 曲线类型: {curve_type}")
    
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
        # 应用平滑插值曲线
        smooth_t_values = np.array([smooth_interpolation_curve(t, curve_type) for t in t_values])
        
        pos_a = primary_last_frame[pos_idx]
        pos_b = secondary_start_frame[pos_idx]
        
        # 向量化根位移插值预计算 (n_blend x 3) - 使用平滑插值
        pos_interpolated = np.outer(1 - smooth_t_values, pos_a) + np.outer(smooth_t_values, pos_b)

        # 生成过渡帧（向量化优化版本）
        for i, (t, smooth_t) in enumerate(zip(t_values, smooth_t_values)):
            current_blend_frame_view = blend_frames[i]
            
            # 复制基础帧
            current_blend_frame_view[:] = primary_last_frame
            
            # 直接赋值预计算的位移插值
            current_blend_frame_view[pos_idx] = pos_interpolated[i]
            
            # 批量处理旋转插值 - 使用平滑插值
            for slerp_obj, joint_idxs, joint_seq in interpolators.values():
                current_blend_frame_view[joint_idxs] = slerp_obj(smooth_t).as_euler(joint_seq, degrees=True)

        return blend_frames, load_primary_bvh, primary_animation_frames, secondary_animation_frames

        # all_frames_np = np.vstack([frames_a, blend_frames, frames_b])


if __name__ == "__main__":
    idle_bvh = ('./tests/new_idle_cross.bvh', [0, 1])
    # target_bvh = ('./motions/yuliao2_001_v03.bvh', [75, 105])
    target_bvh = ('./motions_normalized/SJ_NV_MJSJ_6_001.bvh', [30, 105])
    # bvh = Bvh(Path(target_bvh[0]).read_text(encoding='utf-8'))
    # path = bvh.get_joint_path('hand_r')
    # print(f"关节路径: {path}")

    out_path = './debug/blend_adaptive_gesture_ease_in_out.bvh'
    t_sec = 0.7
    fps = 30

    # 使用播报优化模式，线性插值
    blend_frames, load_primary_bvh, primary_animation_frames, secondary_animation_frames = blend_bvh(
        idle_bvh, target_bvh, out_path, 
        t_sec=t_sec,
        fps=fps,
        adaptive_mode='fixed',  # 'velocity'速度优化, 'fixed'固定时间
        curve_type='linear'  # 'ease_in_out', 'ease_in', 'ease_out', 'linear'
    )
    
    print(f"自适应混合帧数: {blend_frames.shape[0]}")
    # 合成最终动画：idle + 过渡帧 + 目标动作
    all_frames_np = np.vstack([primary_animation_frames, blend_frames, secondary_animation_frames])
    write_bvh(load_primary_bvh, out_path, all_frames_np)
