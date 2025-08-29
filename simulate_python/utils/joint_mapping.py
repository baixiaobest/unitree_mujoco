import torch

class JointMapping:
    def __init__(self, policy_joint_order, unitree_joint_order, device="cpu"):
        self.policy_joint_order = policy_joint_order
        self.unitree_joint_order = unitree_joint_order
        self.mapping= self._construct_mapping()
        self.device = device

    def _construct_mapping(self):
        """Constructs a mapping from policy joint order to Unitree joint order."""
        mapping = {}
        for i, joint in enumerate(self.policy_joint_order):
            unitree_idx = self.unitree_joint_order.index(joint)
            mapping[i] = unitree_idx
        return mapping
    
    def policy_to_unitree(self, policy_action: torch.Tensor, scale: float=1.0, offset: torch.Tensor|None=None):
        """
        Reorders joint values from policy joint order to Unitree joint order.
        
        Args:
            policy_action: Tensor of shape (12,) in policy joint order
            device: Device to place the resulting tensor on
            
        Returns:
            Tensor of shape (12,) in Unitree joint order
        """
        scaled_and_offset = scale * policy_action + (0 if offset is None else offset)
        unitree_action = torch.zeros_like(scaled_and_offset, device=self.device)
        for policy_idx, unitree_idx in self.mapping.items():
            unitree_action[unitree_idx] = scaled_and_offset[policy_idx]
        
        return unitree_action
    
    def unitree_to_policy(self, unitree_action: torch.Tensor, scale: float=1.0, offset: torch.Tensor|None=None):
        """
        Reorders joint values from Unitree joint order to policy joint order.
        
        Args:
            unitree_action: Tensor of shape (12,) in Unitree joint order
            device: Device to place the resulting tensor on
            
        Returns:
            Tensor of shape (12,) in policy joint order
        """
        policy_action = torch.zeros_like(unitree_action, device=self.device)
        for policy_idx, unitree_idx in self.mapping.items():
            policy_action[policy_idx] = unitree_action[unitree_idx]

        policy_action = (policy_action - (0 if offset is None else offset)) / scale
        
        return policy_action