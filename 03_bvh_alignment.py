# -*- coding: utf-8 -*-
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import copy
import math

class BVHAlignment:
    def __init__(self, ref_bvh: str, target_bvh: str):
        """
        初始化BVH对齐工具
        
        Args:
            ref_bvh: 参考文件
            target_bvh: 待对齐文件
        """
        self.ref_bvh = ref_bvh
        self.target_bvh = target_bvh
        
        # 加载两个BVH文件
        self.ref_bvh_data = self.load_bvh(ref_bvh)
        self.target_bvh_data = self.load_bvh(target_bvh)
        
        # 验证骨骼结构一致性
        self.verify_skeleton_consistency()
        
        print(f"ref_bvh: {len(self.ref_bvh_data['motion_data'])}帧, {self.ref_bvh_data['channels']}通道")
        print(f"target_bvh: {len(self.target_bvh_data['motion_data'])}帧, {self.target_bvh_data['channels']}通道")
    
    def load_bvh(self, file_path: str) -> Dict:
        """加载BVH文件"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        bvh_data = {
            'header': "",
            'motion_data': [],
            'frame_time': 0.033333,
            'frames': 0,
            'channels': 0,
            'joint_names': [],
            'channel_names': [],
            'joint_hierarchy': {}
        }
        
        motion_start = False
        header_lines = []
        current_joint = None
        joint_stack = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            if stripped_line == "MOTION":
                motion_start = True
                header_lines = lines[:i+1]
                continue
            
            if not motion_start:
                # 解析骨骼层次结构
                if stripped_line.startswith("ROOT") or stripped_line.startswith("JOINT"):
                    current_joint = stripped_line.split()[1]
                    bvh_data['joint_names'].append(current_joint)
                    joint_stack.append(current_joint)
                elif stripped_line.startswith("CHANNELS"):
                    parts = stripped_line.split()
                    channel_count = int(parts[1])
                    channels = parts[2:]
                    for channel in channels:
                        bvh_data['channel_names'].append(f"{current_joint}_{channel}")
                elif stripped_line == "{":
                    continue
                elif stripped_line == "}":
                    if joint_stack:
                        joint_stack.pop()
                    current_joint = joint_stack[-1] if joint_stack else None
            else:
                if stripped_line.startswith("Frames:"):
                    bvh_data['frames'] = int(stripped_line.split()[1])
                elif stripped_line.startswith("Frame Time:"):
                    bvh_data['frame_time'] = float(stripped_line.split()[2])
                else:
                    if stripped_line:
                        frame_data = [float(x) for x in stripped_line.split()]
                        bvh_data['motion_data'].append(frame_data)
        
        bvh_data['header'] = ''.join(header_lines)
        bvh_data['motion_data'] = np.array(bvh_data['motion_data'])
        bvh_data['channels'] = bvh_data['motion_data'].shape[1] if len(bvh_data['motion_data']) > 0 else 0
        
        return bvh_data
    
    def verify_skeleton_consistency(self):
        """验证两个BVH文件的骨骼结构是否一致"""
        if len(self.ref_bvh_data['joint_names']) != len(self.target_bvh_data['joint_names']):
            raise ValueError("两个BVH文件的关节数量不一致")
        
        if self.ref_bvh_data['channels'] != self.target_bvh_data['channels']:
            raise ValueError("两个BVH文件的通道数量不一致")
        
        for i, (joint1, joint2) in enumerate(zip(self.ref_bvh_data['joint_names'], self.target_bvh_data['joint_names'])):
            if joint1 != joint2:
                print(f"警告: 关节名称不一致 - {joint1} vs {joint2}")
        
        print("骨骼结构验证完成")
    
    def get_root_transform(self, bvh_data: Dict, frame_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取根节点的位置和旋转
        
        Args:
            bvh_data: BVH数据
            frame_idx: 帧索引
            
        Returns:
            (position, rotation) - 位置向量和旋转向量(欧拉角)
        """
        frame_data = bvh_data['motion_data'][frame_idx]
        
        # 通常前6个通道是根节点的位置(X,Y,Z)和旋转(X,Y,Z)
        position = frame_data[:3]  # X, Y, Z position
        rotation = frame_data[3:6]  # X, Y, Z rotation (euler angles)
        
        return position, rotation
    
    def spatial_alignment(self, 
                         reference_frame: int = 0,
                         target_frame: int = 0,
                         align_position: bool = True,
                         align_rotation: bool = True) -> np.ndarray:
        """
        空间对齐：将target_bvh对齐到ref_bvh的空间位置
        
        Args:
            reference_frame: 参考帧（ref_bvh）
            target_frame: 目标帧（target_bvh）
            align_position: 是否对齐位置
            align_rotation: 是否对齐旋转
            
        Returns:
            对齐后的target_bvh动画数据
        """
        # 获取参考位置和旋转
        ref_pos, ref_rot = self.get_root_transform(self.ref_bvh_data, reference_frame)
        target_pos, target_rot = self.get_root_transform(self.target_bvh_data, target_frame)
        
        # 计算对齐变换
        pos_offset = ref_pos - target_pos if align_position else np.zeros(3)
        rot_offset = ref_rot - target_rot if align_rotation else np.zeros(3)
        
        # 应用变换到所有帧
        aligned_data = self.target_bvh_data['motion_data'].copy()
        
        for frame_idx in range(len(aligned_data)):
            if align_position:
                aligned_data[frame_idx, :3] += pos_offset
            
            if align_rotation:
                aligned_data[frame_idx, 3:6] += rot_offset
        
        print(f"空间对齐完成:")
        print(f"  位置偏移: {pos_offset}")
        print(f"  旋转偏移: {rot_offset}")
        
        return aligned_data
    
    def temporal_alignment(self, 
                          target_frames: int = None,
                          alignment_method: str = "linear") -> np.ndarray:
        """
        时间对齐：调整target_bvh的时间长度以匹配ref_bvh
        
        Args:
            target_frames: 目标帧数，如果为None则使用ref_bvh的帧数
            alignment_method: 对齐方法 ("linear", "cubic", "preserve_keyframes")
            
        Returns:
            时间对齐后的target_bvh动画数据
        """
        if target_frames is None:
            target_frames = len(self.ref_bvh_data['motion_data'])
        
        original_frames = len(self.target_bvh_data['motion_data'])
        
        if original_frames == target_frames:
            print("时间长度已一致，无需调整")
            return self.target_bvh_data['motion_data'].copy()
        
        print(f"时间对齐: {original_frames}帧 -> {target_frames}帧")
        
        # 创建时间映射
        original_time = np.linspace(0, 1, original_frames)
        target_time = np.linspace(0, 1, target_frames)
        
        # 对每个通道进行插值
        aligned_data = np.zeros((target_frames, self.target_bvh_data['channels']))
        
        for channel in range(self.target_bvh_data['channels']):
            if alignment_method == "linear":
                interp_func = interp1d(original_time, self.target_bvh_data['motion_data'][:, channel], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
            elif alignment_method == "cubic":
                interp_func = interp1d(original_time, self.target_bvh_data['motion_data'][:, channel], 
                                     kind='cubic', bounds_error=False, fill_value='extrapolate')
            else:  # preserve_keyframes
                interp_func = interp1d(original_time, self.target_bvh_data['motion_data'][:, channel], 
                                     kind='quadratic', bounds_error=False, fill_value='extrapolate')
            
            aligned_data[:, channel] = interp_func(target_time)
        
        return aligned_data
    
    def pose_alignment(self, 
                      reference_frame: int = 0,
                      target_frame: int = 0,
                      align_all_joints: bool = False) -> np.ndarray:
        """
        姿态对齐：将target_bvh的特定帧姿态对齐到ref_bvh
        
        Args:
            reference_frame: 参考帧（ref_bvh）
            target_frame: 目标帧（target_bvh）
            align_all_joints: 是否对齐所有关节（否则只对齐根节点）
            
        Returns:
            姿态对齐后的target_bvh动画数据
        """
        ref_frame_data = self.ref_bvh_data['motion_data'][reference_frame]
        target_frame_data = self.target_bvh_data['motion_data'][target_frame]
        
        aligned_data = self.target_bvh_data['motion_data'].copy()
        
        if align_all_joints:
            # 对齐所有关节
            offset = ref_frame_data - target_frame_data
            
            # 应用偏移到所有帧
            for frame_idx in range(len(aligned_data)):
                aligned_data[frame_idx] += offset
        else:
            # 只对齐根节点（前6个通道）
            root_offset = ref_frame_data[:6] - target_frame_data[:6]
            thigh_offset = ref_frame_data[828:] - target_frame_data[828:]  # 假设828是其他关节的起始索引
            
            for frame_idx in range(len(aligned_data)):
                # aligned_data[frame_idx, :6] += root_offset
                aligned_data[frame_idx, 828:] += thigh_offset  # 保持其他关节不变
                aligned_data[frame_idx, :6] = ref_frame_data[:6]
                # aligned_data[frame_idx, 828:] = ref_frame_data[828:]  # 保持其他关节不变
        
        print(f"姿态对齐完成 (对齐{'所有关节' if align_all_joints else '根节点'})")
        
        return aligned_data
    
    def scale_alignment(self, 
                       scale_factor: float = None,
                       auto_scale: bool = True,
                       reference_joint_indices: List[int] = None) -> np.ndarray:
        """
        缩放对齐：调整target_bvh的动作幅度以匹配ref_bvh
        
        Args:
            scale_factor: 缩放因子，如果为None则自动计算
            auto_scale: 是否自动计算缩放因子
            reference_joint_indices: 用于计算缩放的参考关节索引
            
        Returns:
            缩放对齐后的target_bvh动画数据
        """
        if reference_joint_indices is None:
            # 使用根节点的位置通道
            reference_joint_indices = [0, 1, 2]
        
        if auto_scale or scale_factor is None:
            # 计算两个动画的平均运动幅度
            ref_bvh_range = np.ptp(self.ref_bvh_data['motion_data'][:, reference_joint_indices], axis=0)
            target_bvh_range = np.ptp(self.target_bvh_data['motion_data'][:, reference_joint_indices], axis=0)
            
            # 计算平均缩放因子
            valid_ratios = []
            for i in range(len(reference_joint_indices)):
                if target_bvh_range[i] != 0:
                    valid_ratios.append(ref_bvh_range[i] / target_bvh_range[i])
            
            if valid_ratios:
                scale_factor = np.mean(valid_ratios)
            else:
                scale_factor = 1.0
        
        print(f"缩放对齐: 缩放因子 = {scale_factor:.4f}")
        
        # 应用缩放
        aligned_data = self.target_bvh_data['motion_data'].copy()
        
        # 计算每个动画的中心点
        ref_bvh_center = np.mean(self.ref_bvh_data['motion_data'][:, reference_joint_indices], axis=0)
        target_bvh_center = np.mean(self.target_bvh_data['motion_data'][:, reference_joint_indices], axis=0)
        
        # 以中心点为基准进行缩放
        for frame_idx in range(len(aligned_data)):
            for joint_idx in reference_joint_indices:
                # 相对于中心点缩放
                offset = aligned_data[frame_idx, joint_idx] - target_bvh_center[joint_idx - reference_joint_indices[0]]
                aligned_data[frame_idx, joint_idx] = target_bvh_center[joint_idx - reference_joint_indices[0]] + offset * scale_factor
        
        return aligned_data
    
    def comprehensive_alignment(self, 
                              spatial_align: bool = True,
                              temporal_align: bool = True,
                              pose_align: bool = True,
                              scale_align: bool = False,
                              reference_frame: int = 0,
                              target_frame: int = 0) -> np.ndarray:
        """
        综合对齐：执行多种对齐方法
        
        Args:
            spatial_align: 是否进行空间对齐
            temporal_align: 是否进行时间对齐
            pose_align: 是否进行姿态对齐
            scale_align: 是否进行缩放对齐
            reference_frame: 参考帧
            target_frame: 目标帧
            
        Returns:
            综合对齐后的target_bvh动画数据
        """
        print("=== 开始综合对齐 ===")
        
        # 从原始数据开始
        aligned_data = self.target_bvh_data['motion_data'].copy()
        
        # 步骤1: 时间对齐（必须先做，因为会改变帧数）
        if temporal_align:
            print("\n1. 执行时间对齐...")
            # 临时更新target_bvh数据进行时间对齐
            target_motion_data = self.target_bvh_data.copy()
            target_motion_data['motion_data'] = aligned_data
            
            aligned_data = self.temporal_alignment()
            
            # 更新target_bvh数据以供后续步骤使用
            self.target_bvh_data['motion_data'] = aligned_data
        
        # 步骤2: 缩放对齐
        if scale_align:
            print("\n2. 执行缩放对齐...")
            target_motion_data = self.target_bvh_data.copy()
            target_motion_data['motion_data'] = aligned_data
            
            aligned_data = self.scale_alignment()
            self.target_bvh_data['motion_data'] = aligned_data
        
        # 步骤3: 姿态对齐
        if pose_align:
            print("\n3. 执行姿态对齐...")
            target_motion_data = self.target_bvh_data.copy()
            target_motion_data['motion_data'] = aligned_data
            
            # 调整参考帧索引（如果时间对齐改变了帧数）
            adjusted_target_frame = min(target_frame, len(aligned_data) - 1)
            
            aligned_data = self.pose_alignment(reference_frame, adjusted_target_frame)
            self.target_bvh_data['motion_data'] = aligned_data
        
        # 步骤4: 空间对齐（最后做，确保位置正确）
        if spatial_align:
            print("\n4. 执行空间对齐...")
            target_motion_data = self.target_bvh_data.copy()
            target_motion_data['motion_data'] = aligned_data
            
            adjusted_target_frame = min(target_frame, len(aligned_data) - 1)
            
            aligned_data = self.spatial_alignment(reference_frame, adjusted_target_frame)
        
        print("\n=== 综合对齐完成 ===")
        return aligned_data
    
    def save_aligned_bvh(self, aligned_data: np.ndarray, output_path: str):
        """保存对齐后的BVH文件"""
        with open(output_path, 'w') as f:
            # 使用target_bvh的头部信息
            f.write(self.target_bvh_data['header'])
            
            # 写入新的帧数信息
            f.write(f"Frames: {len(aligned_data)}\n")
            f.write(f"Frame Time: {self.target_bvh_data['frame_time']}\n")
            
            # 写入对齐后的动作数据
            for frame in aligned_data:
                frame_str = ' '.join([f"{x:.6f}" for x in frame])
                f.write(frame_str + '\n')
        
        print(f"对齐后的BVH文件已保存到: {output_path}")
    
    def analyze_alignment_quality(self, aligned_data: np.ndarray) -> Dict:
        """分析对齐质量"""
        analysis = {}
        
        # 时间长度比较
        analysis['frame_count'] = {
            'ref_bvh': len(self.ref_bvh_data['motion_data']),
            'target_bvh_original': len(self.target_bvh_data['motion_data']),
            'target_bvh_aligned': len(aligned_data)
        }
        
        # 空间范围比较
        ref_bvh_range = np.ptp(self.ref_bvh_data['motion_data'][:, :3], axis=0)
        target_bvh_range = np.ptp(aligned_data[:, :3], axis=0)
        
        analysis['spatial_range'] = {
            'ref_bvh': ref_bvh_range,
            'target_bvh_aligned': target_bvh_range,
            'ratio': target_bvh_range / (ref_bvh_range + 1e-8)
        }
        
        # 起始姿态比较
        ref_bvh_start = self.ref_bvh_data['motion_data'][0]
        target_bvh_start = aligned_data[0]
        
        analysis['pose_difference'] = {
            'root_position_diff': np.linalg.norm(ref_bvh_start[:3] - target_bvh_start[:3]),
            'root_rotation_diff': np.linalg.norm(ref_bvh_start[3:6] - target_bvh_start[3:6]),
            'total_diff': np.linalg.norm(ref_bvh_start - target_bvh_start)
        }
        
        return analysis

# 主要使用函数
def align_bvh_files(ref_bvh: str, 
                   target_bvh: str, 
                   output_path: str,
                   alignment_config: Dict = None) -> Dict:
    """
    对齐两个BVH文件的主函数
    
    Args:
        ref_bvh: 参考BVH文件路径
        target_bvh: 待对齐BVH文件路径
        output_path: 输出文件路径
        alignment_config: 对齐配置
        
    Returns:
        对齐质量分析结果
    """
    
    # 默认配置
    default_config = {
        'spatial_align': False,
        'temporal_align': False,
        'pose_align': True,
        'scale_align': False,
        'reference_frame': 0,
        'target_frame': 0
    }
    
    if alignment_config:
        default_config.update(alignment_config)
    
    # 创建对齐器
    aligner = BVHAlignment(ref_bvh, target_bvh)
    
    # 执行综合对齐
    aligned_data = aligner.comprehensive_alignment(**default_config)
    
    # 保存结果
    aligner.save_aligned_bvh(aligned_data, output_path)
    
    # 分析对齐质量
    quality_analysis = aligner.analyze_alignment_quality(aligned_data)
    
    print("\n=== 对齐质量分析 ===")
    print(f"帧数: ref_bvh={quality_analysis['frame_count']['ref_bvh']}, "
          f"target_bvh原始={quality_analysis['frame_count']['target_bvh_original']}, "
          f"target_bvh对齐后={quality_analysis['frame_count']['target_bvh_aligned']}")
    print(f"根节点位置差异: {quality_analysis['pose_difference']['root_position_diff']:.4f}")
    print(f"根节点旋转差异: {quality_analysis['pose_difference']['root_rotation_diff']:.4f}")
    print(f"总体姿态差异: {quality_analysis['pose_difference']['total_diff']:.4f}")
    
    return quality_analysis

# 使用示例
if __name__ == "__main__":
    # 示例1: 基本对齐
    # align_bvh_files(
    #     ref_bvh="./motions/baohuo_0702.bvh",      # 参考文件
    #     target_bvh="./motions/yulu1-002(04).bvh",         # 待对齐文件
    #     output_path="aligned_output.bvh"
    # )
    
    # 示例2: 自定义对齐配置
    # custom_config = {
    #     'spatial_align': True,     # 空间对齐
    #     'temporal_align': True,    # 时间对齐
    #     'pose_align': True,        # 姿态对齐
    #     'scale_align': True,       # 缩放对齐
    #     'reference_frame': 0,      # 参考帧
    #     'target_frame': 0          # 目标帧
    # }
    
    # align_bvh_files(
    #     ref_bvh="reference.bvh",
    #     target_bvh="target.bvh",
    #     output_path="custom_aligned.bvh",
    #     alignment_config=custom_config
    # )
    
    # 示例3: 单独使用特定对齐功能
    # aligner = BVHAlignment("./motions_normalized/baohuo_0703_02.bvh", "./motions_normalized/yulu1-002(04).bvh")
    
    # 只进行空间对齐
    # spatial_aligned = aligner.spatial_alignment(reference_frame=0, target_frame=0)
    # aligner.save_aligned_bvh(spatial_aligned, "spatial_aligned.bvh")
    
    # # 只进行时间对齐
    # temporal_aligned = aligner.temporal_alignment(target_frames=120)
    # aligner.save_aligned_bvh(temporal_aligned, "temporal_aligned.bvh")

    # 只进行姿态对齐
    # pose_aligned = aligner.pose_alignment(reference_frame=0, target_frame=0)
    # aligner.save_aligned_bvh(pose_aligned, "pose_aligned.bvh")

    # 示例4: 批量对齐
    motions_normalized = "./motions_normalized"
    bvh_files = list(Path(motions_normalized).glob('*.bvh'))

    if not bvh_files:
        print("没有找到BVH文件，请检查路径")
    else:
        for bvh_file in bvh_files:
            try:
                print(f"正在处理文件: {bvh_file.name}")
                align_bvh_files(
                    ref_bvh="./motions_normalized/baohuo_0703_02.bvh",  # 参考文件
                    target_bvh=str(bvh_file),  # 待对齐文件
                    output_path=f"./align_all_joints/{bvh_file.stem}.bvh"
                )
            except Exception as e:
                print(f"处理 {bvh_file.name} 时出错: {e}")
