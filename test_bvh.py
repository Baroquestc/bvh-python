from bvh import Bvh
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    # Example usage
    # 1. Load a BVH file
    idle_data = Path('./tests/LJ_NV_DBJJ_1_001.bvh').read_text(encoding='utf-8')
    bvh = Bvh(idle_data)
    print("Number of frames:", bvh.nframes)

    # 保存为新的BVH文件
    # output_path = Path('./output.bvh')
    # output_path.write_text(bvh.as_bvh(), encoding='utf-8')
    
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
