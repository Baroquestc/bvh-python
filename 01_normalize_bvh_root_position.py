import os  

def normalize_bvh_root_position(bvh_data, root_initial):  
    lines = bvh_data.splitlines()  
    hierarchy_end_idx = next(i for i, line in enumerate(lines) if 'MOTION' in line)  
    
    # 提取层级结构  
    hierarchy = lines[:hierarchy_end_idx]  
    
    # 寻找动作帧数据起始位置  
    frame_count_idx = hierarchy_end_idx + 1  
    frame_time_idx = hierarchy_end_idx + 2  
    motion_start_idx = hierarchy_end_idx + 3  
    
    # 处理帧数据  
    motion_lines = lines[motion_start_idx:]  
    processed_motion_lines = []  
    
    for idx, motion_line in enumerate(motion_lines):  
        if not motion_line.strip():  
            continue  
            
        parts = motion_line.split()  
        values = list(map(float, parts))  
        
        if idx == 0:
            # 计算初始帧的根节点偏移
            offset = [values[0] - root_initial[0], values[1] - root_initial[1], values[2] - root_initial[2]]
        # 所有帧都减去初始帧偏移，使根节点对齐到 root_initial
        values[0] -= offset[0]
        values[1] -= offset[1]
        values[2] -= offset[2]
        processed_motion_lines.append(' '.join(map(str, values)))
    
    # 重构BVH文件内容  
    result = lines[:motion_start_idx] + processed_motion_lines  
    return '\n'.join(result)  

def process_bvh_files(input_folder_path, output_folder_path, root_initial):  
    # 确保输出文件夹存在  
    if not os.path.exists(output_folder_path):  
        os.makedirs(output_folder_path)  
    
    for filename in os.listdir(input_folder_path):  
        if filename.endswith('.bvh'):  
            input_file_path = os.path.join(input_folder_path, filename)  
            output_file_path = os.path.join(output_folder_path, filename)  
            
            try:  
                with open(input_file_path, 'r') as file:  
                    bvh_data = file.read()  
                
                normalized_data = normalize_bvh_root_position(bvh_data, root_initial)  
                
                # 保存处理后的文件到输出文件夹，保持原文件名  
                with open(output_file_path, 'w') as file:  
                    file.write(normalized_data)  
                    
                print(f"已处理: {filename}")  
            except Exception as e:  
                print(f"处理文件 {filename} 时出错: {str(e)}")  

# 使用示例  
process_bvh_files('./motions/', './motions_normalized/', [-0.516278, 84.9455, 0.855116])  
