from joint_mapping import JointMapping
import torch

if __name__ == "__main__":
    policy_joint_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
                            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
                            'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        
    unitree_joint_order = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
                            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 
                            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 
                            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
    
    joint_map = JointMapping(policy_joint_order, unitree_joint_order, device="cpu")

    joints_offset = torch.tensor(
            [ 0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  1.0000,  1.0000, -1.5000, -1.5000, -1.5000, -1.5000], 
            device="cpu", dtype=torch.float32)
    
    joint_scale = 0.5

    unitree_joint_position = torch.tensor(
        [0.07165517,  0.74026716, -1.4568543,
        -0.12187503,  0.73048484, -1.4397367,
        0.07167274,  0.6996284,  -1.3835967,  
        -0.11966448,  0.68990105, -1.3674877], device="cpu")
    
    transformed_position = joint_map.policy_to_unitree(
        joint_map.unitree_to_policy(
            unitree_joint_position, 
            scale=joint_scale, 
            offset=joints_offset), 
        scale=joint_scale,
        offset=joints_offset)
    
    print(f"transformed: {transformed_position}")
    print(f"original: {unitree_joint_position}")