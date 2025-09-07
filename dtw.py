import utils.bvh as bvh
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from dtaidistance import dtw

# --- 1. 辅助函数与姿态处理器 ---

class PoseHelper:
    """姿态处理辅助类，用于转换和计算"""

    @staticmethod
    def to_pose_vector(anim, frame_idx):
        """将指定帧转换为一个姿态向量 (根位置 + 四元数)"""
        pose_vector = []
        joint_names = anim.get_joints_names()

        # 添加根关节的位置
        root = anim.get_joint(joint_names[0])
        print(root) 
        pos = anim.frame_joint_channels(frame_idx, root.name, ['Xposition', 'Yposition', 'Zposition'])
        pose_vector.extend(pos)

        # 添加所有关节的旋转四元数
        for joint_name in joint_names:
            joint = anim.get_joint(joint_name)
            # bvh-python默认使用ZXY欧拉角顺序
            rot_euler = anim.frame_joint_channels(frame_idx, joint.name, ['Zrotation', 'Xrotation', 'Yrotation'])
            quat = R.from_euler('zxy', rot_euler, degrees=True).as_quat()
            pose_vector.extend(quat)
            
        return np.array(pose_vector)

    @staticmethod
    def to_pose_vectors(anim, start_frame, end_frame):
        """将一个动画片段转换为姿态向量序列"""
        return np.array([PoseHelper.to_pose_vector(anim, i) for i in range(start_frame, end_frame)])

    @staticmethod
    def pose_distance(pose1, pose2):
        # pose1, pose2: shape=(D,)
        # 例如只对前3维（根节点位置）做距离
        return np.linalg.norm(pose1[:3] - pose2[:3])


# --- 2. DTW动画融合器 ---

class BVH_DTW_Blender:
    """使用DTW进行BVH动画融合"""

    def __init__(self, anim_A, anim_B):
        """
        Args:
            anim_A (bvh.Bvh): 源动画对象
            anim_B (bvh.Bvh): 目标动画对象
        """
        if anim_A.get_joints_names() != anim_B.get_joints_names():
            raise ValueError("两个动画的骨骼结构必须完全相同!")
        
        self.anim_A = anim_A
        self.anim_B = anim_B
        self.joint_names = self.anim_A.get_joints_names()
        # 确保帧率一致，否则需要重采样
        if abs(anim_A.frame_time - anim_B.frame_time) > 1e-6:
             print("警告: 动画帧率不匹配，融合效果可能受影响。")
        self.frame_time = anim_A.frame_time

    def blend(self, start_A_frame, start_B_frame, end_B_frame, blend_window_A=30, blend_window_B=30):
        """
        执行DTW融合。

        Args:
            start_A_frame (int): 截取源动画的起始帧。
            end_B_frame (int): 截取目标动画的结束帧。
            blend_window_A (int): 用于DTW分析的源动画窗口大小（从末尾取）。
            blend_window_B (int): 用于DTW分析的目标动画窗口大小（从开头取）。

        Returns:
            bvh.Bvh: 一个包含融合后完整动画的新bvh对象。
        """
        print("开始DTW动画融合...")
        
        # --- 步骤 1: 提取用于DTW分析的姿态序列 ---
        # 从源动画的末尾和目标动画的开头提取姿态向量
        import time
        start_time = time.time()
        seq_A = PoseHelper.to_pose_vectors(self.anim_A, start_A_frame, start_A_frame + blend_window_A)
        seq_B = PoseHelper.to_pose_vectors(self.anim_B, start_B_frame, start_B_frame + blend_window_B)
        end_time = time.time()
        print(f"提取姿态向量时间: {end_time - start_time} 秒")
        
        # --- 步骤 2: 计算DTW对齐路径 ---
        print(f"计算 {blend_window_A} 帧与 {blend_window_B} 帧之间的DTW路径...")
        # dtaidistance 使用自定义距离函数
        dtw_dist, dtw_path = dtw_custom(seq_A, seq_B, dist_func=PoseHelper.pose_distance)
        end_time = time.time()
        print(f"计算DTW路径时间: {end_time - start_time} 秒")
        # --- 步骤 3: 根据DTW路径生成融合帧 ---
        blended_frames_data = self._generate_blended_frames(dtw_path, seq_A, seq_B)
        num_blended_frames = len(blended_frames_data)
        print(f"生成了 {num_blended_frames} 帧过渡动画。")

        # # --- 步骤 4: 拼接动画片段 ---
        # # 1. 源动画的主要部分
        # final_frames = [self.anim_A.frame_to_list(i) for i in range(start_A_frame)]
        # # 2. 新生成的融合部分
        # final_frames.extend(blended_frames_data)
        # # 3. 目标动画的剩余部分
        # final_frames.extend([list(self.anim_B.frames[i]) for i in range(blend_window_B, end_B_frame)])

        # # --- 步骤 5: 创建并返回新的BVH对象 ---
        # # 复制骨架结构
        # new_anim = bvh.Bvh(self.anim_A.as_bvh()) 
        # new_anim.frames = final_frames
        
        # print("融合完成！")
        # return new_anim

    def _generate_blended_frames(self, path, seq_A, seq_B):
        """根据DTW路径插值生成过渡帧"""
        blended_frames = []
        
        # `path` 是一个 (index_A, index_B) 的元组列表
        num_blend_steps = len(path)
        
        for i in range(num_blend_steps):
            # 获取对齐的源/目标姿态向量
            idx_A, idx_B = path[i]
            pose_A = seq_A[idx_A]
            pose_B = seq_B[idx_B]

            # 计算混合权重 alpha，从0平滑过渡到1
            alpha = i / (num_blend_steps - 1)
            alpha = 0.5 * (1 - np.cos(alpha * np.pi)) # 使用平滑曲线 (Smoothstep)
            
            # 插值姿态
            blended_pose = self._interpolate_poses(pose_A, pose_B, alpha)
            
            # 将插值后的姿态向量转换回bvh帧数据格式
            blended_frames.append(self._pose_vector_to_frame_data(blended_pose))
            
        return blended_frames

    def _interpolate_poses(self, pose_A, pose_B, alpha):
        """使用SLERP和线性插值混合两个姿态向量"""
        # 线性插值根位置
        pos_A, pos_B = pose_A[:3], pose_B[:3]
        interp_pos = (1 - alpha) * pos_A + alpha * pos_B
        
        interpolated_vector = list(interp_pos)
        
        # SLERP插值所有关节的旋转
        num_joints = (len(pose_A) - 3) // 4
        for i in range(num_joints):
            start_idx = 3 + i * 4
            end_idx = start_idx + 4
            
            quat_A = pose_A[start_idx:end_idx]
            quat_B = pose_B[start_idx:end_idx]
            
            # 使用Scipy的Rotation进行SLERP
            key_rots = R.from_quat([quat_A, quat_B])
            slerp = Slerp([0, 1], key_rots)
            interp_quat = slerp([alpha]).as_quat()[0]
            
            interpolated_vector.extend(interp_quat)
            
        return np.array(interpolated_vector)

    def _pose_vector_to_frame_data(self, pose_vector):
        """将姿态向量转换回BVH的单帧列表数据"""
        frame_data = []
        
        # 根位置
        frame_data.extend(pose_vector[:3])
        
        # 关节旋转 (从四元数转回欧拉角)
        num_joints = (len(pose_vector) - 3) // 4
        quat_idx = 3
        
        for joint_name in self.joint_names:
            joint = self.anim_A.get_joint(joint_name)
            channels = self.anim_A.joint_channels(joint.name)
            
            # 找到旋转通道
            rot_channels = [c for c in channels if 'rotation' in c.lower()]
            if not rot_channels:
                continue
                
            quat = pose_vector[quat_idx : quat_idx + 4]
            quat_idx += 4
            
            # 转换回欧拉角
            rot_euler = R.from_quat(quat).as_euler('zxy', degrees=True)
            
            # 按照BVH文件中的通道顺序添加数据
            euler_map = {'Zrotation': rot_euler[0], 'Xrotation': rot_euler[1], 'Yrotation': rot_euler[2]}
            
            # 注意：这里我们只处理了旋转，如果有关节也有位置通道，需要额外处理
            # 简单的示例假设非根关节只有旋转
            for channel in channels:
                 if 'position' not in channel.lower(): # 跳过根关节已处理的位置
                    if channel in euler_map:
                         frame_data.append(euler_map[channel])

        return frame_data

def dtw_custom(seq1, seq2, dist_func):
    """
    计算两个序列的DTW距离和最优路径，支持自定义距离函数
    :param seq1: shape=(N, D) 的numpy数组
    :param seq2: shape=(M, D) 的numpy数组
    :param dist_func: 距离函数，dist_func(a, b) -> float
    :return: dtw距离, 路径list[(i, j)]
    """
    N, M = len(seq1), len(seq2)
    cost = np.zeros((N, M))
    # 计算距离矩阵
    for i in range(N):
        for j in range(M):
            cost[i, j] = dist_func(seq1[i], seq2[j])
    # 动态规划填表
    acc_cost = np.zeros((N, M))
    acc_cost[0, 0] = cost[0, 0]
    for i in range(1, N):
        acc_cost[i, 0] = cost[i, 0] + acc_cost[i-1, 0]
    for j in range(1, M):
        acc_cost[0, j] = cost[0, j] + acc_cost[0, j-1]
    for i in range(1, N):
        for j in range(1, M):
            acc_cost[i, j] = cost[i, j] + min(acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1])
    # 回溯最优路径
    path = []
    i, j = N-1, M-1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            idx = np.argmin([acc_cost[i-1, j-1], acc_cost[i-1, j], acc_cost[i, j-1]])
            if idx == 0:
                i -= 1
                j -= 1
            elif idx == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return acc_cost[-1, -1], path

# --- 3. 主函数与示例 ---
def main():
    """
    示例: 将一个"行走"动画平滑地融合到一个"跑步"动画。
    你需要准备两个bvh文件: walk.bvh 和 run.bvh
    """
    try:
        # 加载动画文件
        with open("./tests/idle.bvh") as f:
            anim_walk = bvh.Bvh(f.read())
        with open("./tests/LJ_NV_DBJJ_1_001.bvh") as f:
            anim_run = bvh.Bvh(f.read())
    except FileNotFoundError:
        print("错误: 请确保 'walk.bvh' 和 'run.bvh' 文件在当前目录下。")
        print("你可以从任何BVH资源网站下载这两个文件用于测试。")
        return

    # 创建DTW融合器
    blender = BVH_DTW_Blender(anim_walk, anim_run)

    # 执行融合
    # - 从 walk.bvh 的第0帧开始，取到完整动画
    # - 过渡到 run.bvh 的完整动画
    # - 使用 walk 的最后30帧和 run 的最前30帧进行DTW分析
    blended_animation = blender.blend(
        start_A_frame=0, 
        start_B_frame=70,
        end_B_frame=anim_run.nframes,
        blend_window_A=1,
        blend_window_B=50
    )

    # 保存结果
    output_filename = "blended_walk_to_run.bvh"
    with open(output_filename, 'w') as f:
        f.write(blended_animation.as_bvh())
    
    print(f"\n成功创建融合动画: {output_filename}")
    print(f"原始 'walk' 帧数: {anim_walk.nframes}")
    print(f"原始 'run' 帧数: {anim_run.nframes}")
    print(f"融合后总帧数: {blended_animation.nframes}")

if __name__ == '__main__':
    main()
