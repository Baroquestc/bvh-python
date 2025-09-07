from utils.bvh import Bvh
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    # Example usage
    # 1. Load a BVH file
    idle_data = Path('./tests/idle.bvh').read_text(encoding='utf-8')
    bvh = Bvh(idle_data)
    print("Number of frames:", bvh.nframes)

    # 保存为新的BVH文件
    output_path = Path('./output.bvh')
    output_path.write_text(bvh.as_bvh(), encoding='utf-8')
    
    joint_names = bvh.get_joints_names()
    print("length of joint names:", len(joint_names))

    # 2. Access joint data
    joint = bvh.get_joint("thigh_r")
    print("Joint data:", joint.value)
    print("Joint name:", joint.name)

    # 3. Access frames
    frames = bvh.frames
    frames = np.array(frames)
    print("Frames shape:", frames.shape)

    # 4. Access joint channels
    channels = bvh.joint_channels("pelvis")
    print("Joint channels:", channels)

    # 5. Access joint name index
    joint_index = bvh.get_joint_index("thigh_r")
    print("Joint index:", joint_index)

    # 6. Access joint channels start index
    joint_channel_start_index = bvh.get_joint_channels_index("thigh_r")
    channel_index = bvh.get_joint_channel_index("thigh_r", "Xrotation")
    print("Joint channels index:", joint_channel_start_index)
    print("Channel index for 'thigh_r.Xrotation':", channel_index)

    # 7. Access joint frames
    frames_joint = bvh.frames_joint("thigh_r")
    frames_joint = np.array(frames_joint)
    print("Frames joint shape:", frames_joint.shape)

    # frames_joint = bvh.frames_joint("thigh_r", channels=["Xrotation", "Yrotation"])
    # frames_joint = np.array(frames_joint)
    # print("Frames joint shape:", frames_joint.shape)

    # 8. Access joint global channel index
    # get_joint_global_index = bvh.get_joint_global_channel_index("thigh_r", ["Xrotation", "Yrotation", "Zrotation"])
    # print("Global channel index for 'thigh_r':", get_joint_global_index)

    get_joint_global_index = bvh.get_joint_global_channel_index("thigh_r")
    print("Global channel index for 'thigh_r':", get_joint_global_index)

    print(bvh.get_joint_index("spine_01"))
    print(bvh.get_joint_index("thigh_r"))
    print(bvh.get_joint_index("thigh_l"))

    # 9. Get joint position at a specific frame
    # 计算foot_l和ROOT的距离
    root_coord = bvh.get_joint_position('pelvis', 0)
    foot_l_coord = bvh.get_joint_position('foot_l', 0)
    foot_r_coord = bvh.get_joint_position('foot_r', 0)
    print(f"root_coord: {root_coord}, foot_l_coord: {foot_l_coord}, foot_r_coord: {foot_r_coord}")
    # 计算foot_l和foot_r的中点
    mid_coord = (foot_l_coord + foot_r_coord) / 2
    print(f"mid_coord: {mid_coord}")
    # 计算foot_l和ROOT的距离
    distance = np.linalg.norm(root_coord - foot_l_coord)
    print("foot_l和ROOT的距离:", distance)
    # 计算foot_l和mid_coord的距离
    distance = np.linalg.norm(foot_l_coord - mid_coord)
    print("foot_l和mid_coord的距离:", distance)
    # 计算foot_r和mid_coord的距离
    distance = np.linalg.norm(foot_r_coord - mid_coord)
    print("foot_r和mid_coord的距离:", distance)
