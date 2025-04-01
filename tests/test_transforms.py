import numpy as np
from pathlib import Path
from gr00t.data.dataset import ModalityConfig, DatasetMetadata
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.utils.misc import any_describe
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.dataset import LeRobotSingleDataset
def create_test_observation_dict():
    # Set horizon length N
    N = 1  # Using 10 timesteps as an example
    
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

def test_dataset_with_transforms():
    # 1. Create fake observation dictionary
    obs_dict = create_test_observation_dict()
    
    # 2. Set up configs similar to test_dataset.py
    embodiment_tag = EmbodimentTag("new_embodiment")
    dataset_path = "/home/zhexin/data/xarm_dual/pour_1000"
    
    # 3. Get the data config and modality configs from DATA_CONFIG_MAP
    data_config_cls = DATA_CONFIG_MAP["new_embodiment_joint"]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.deployment_transform()
    
    # Create dataset to get metadata
    train_dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=None,
        embodiment_tag=embodiment_tag,
        video_backend="decord",
    )
    
    # Set metadata from the dataset
    transforms.set_metadata(train_dataset.metadata)
    obs_dict = transforms(obs_dict)
    
    # 5. Print the transformed observation dictionary
    for key, value in obs_dict.items():
        try:
            print(key, value.shape, value.min(), value.max(),'\n')
        except:
            print(key, value, value,'\n')

if __name__ == "__main__":
    test_dataset_with_transforms()