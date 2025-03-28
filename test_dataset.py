from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

dataset_path = "data/xarm_dual/pour_1000"

# get the data config
embodiment_tag = EmbodimentTag("new_embodiment")

# 1.1 modality configs and transforms
data_config_cls = DATA_CONFIG_MAP["new_embodiment"]
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()

# 1.2 data loader
train_dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    transforms=transforms,
    embodiment_tag=embodiment_tag,
    video_backend="decord",
)

batch = train_dataset[0]

for key, value in batch.items():
    try:
        print(key, value.shape, value.min(), value.max())
    except:
        print(key, value)
# print(batch["state.left_arm"])
# print(batch["action.left_arm"])

# imgs = batch["pixel_values"]
# import matplotlib.pyplot as plt
# import torch
# for i in range(imgs.shape[0]):
#     img_tensor = imgs[i]
#     img_tensor = img_tensor.permute(1, 2, 0)
#     img_tensor = img_tensor.cpu().type(torch.float32).numpy()
#     plt.imshow(img_tensor)
#     plt.savefig(f"eval/img_{i}.png")
#     plt.close()
