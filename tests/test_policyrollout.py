import numpy as np
import torch
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.transform.obs_buffer import ObsBufferTransform
def create_test_observation_dict():
    # Set horizon length N
    N = 1  # Match the expected sequence length directly
    
    # Create fake image data (random values between 0-255 for RGB images)
    image_shape = (480, 640, 3)
    fake_head_image = np.random.randint(0, 256, (N, *image_shape), dtype=np.uint8)
    fake_left_wrist_image = np.random.randint(0, 256, (N, *image_shape), dtype=np.uint8)
    fake_right_wrist_image = np.random.randint(0, 256, (N, *image_shape), dtype=np.uint8)
    
    # Create fake state data
    fake_ee_pose_left = np.random.uniform(-1, 1, (N, 9)).astype(np.float32)  # position + rotation6d
    fake_ee_pose_right = np.random.uniform(-1, 1, (N, 9)).astype(np.float32)  # position + rotation6d
    
    fake_joint_angle_left = np.random.uniform(-np.pi, np.pi, (N, 7)).astype(np.float32)
    fake_joint_angle_right = np.random.uniform(-np.pi, np.pi, (N, 7)).astype(np.float32)
    
    fake_gripper_left = np.random.uniform(0, 1, (N, 1)).astype(np.float32)
    fake_gripper_right = np.random.uniform(0, 1, (N, 1)).astype(np.float32)
    
    # Create the observation dictionary
    observation_dict = {
        "observation_images_head": fake_head_image,
        "observation_images_left_wrist": fake_left_wrist_image,
        "observation_images_right_wrist": fake_right_wrist_image,
        "observation_states_ee_pose_left": fake_ee_pose_left,
        "observation_states_joint_angle_left": fake_joint_angle_left,
        "observation_states_gripper_position_left": fake_gripper_left,
        "observation_states_ee_pose_right": fake_ee_pose_right,
        "observation_states_joint_angle_right": fake_joint_angle_right,
        "observation_states_gripper_position_right": fake_gripper_right,
    }
    
    return observation_dict

def test_policy_rollout():
    # Configuration
    model_path = "/home/zhexin/ckpt/gr00t_pour1000_joint/checkpoint-20000"
    embodiment_tag = "new_embodiment"
    data_config = DATA_CONFIG_MAP["new_embodiment_joint"]

    # Initialize policy with deployment_transform()
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.deployment_transform(),
        embodiment_tag=embodiment_tag,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create single observation with proper sequence length
    obs = create_test_observation_dict()

    # for key in obs.keys():
    #     print(key, obs[key].shape)
    #obs = ObsBufferTransform(apply_to=["observation_images_head", "observation_images_left_wrist", "observation_images_right_wrist", "observation_states_ee_pose_left", "observation_states_joint_angle_left", "observation_states_gripper_position_left", "observation_states_ee_pose_right", "observation_states_joint_angle_right", "observation_states_gripper_position_right"]).apply(obs)
    
    # Get action from policy
    action = policy.get_action(obs)
    
    # Print action shape and values
    print("\nAction outputs:")
    for key, value in action.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, min={value.min():.3f}, max={value.max():.3f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_policy_rollout()