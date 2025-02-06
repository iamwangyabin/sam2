from torch.utils.data import Dataset
from .database import CoreDataset, FileDataset
import yaml

class IncrementalDataManager(Dataset):

    def __init__(self, yaml_file_path):
        super().__init__()
        self.session_datasets = {}
        self._current_session = None
        
        # 1. 读取 YAML，然后分离 common 和 session_configs
        with open(yaml_file_path, 'r') as f:
            full_config = yaml.safe_load(f)
            if not isinstance(full_config, dict):
                raise ValueError("YAML 文件格式错误：顶层应为 dict。")

        common_cfg = full_config.get("common", {})
        session_configs = full_config.get("session_configs", {})
        if not isinstance(session_configs, dict) or len(session_configs) == 0:
            raise ValueError("session_configs 不存在或不是 dict。")

        # 2. 遍历 session_configs，将 common 参数合并/覆盖到每个 session 里
        for session_id, cfg in session_configs.items():
            if "datasets_list" not in cfg:
                raise ValueError(f"Session {session_id} 缺少 'datasets_list' 配置。")

            # 将 common_cfg 和 session_cfg 合并
            merged_cfg = dict(common_cfg)  # 复制公共参数
            merged_cfg.update(cfg)         # 让 session 覆盖公共字段

            # 3. 构建 ds_list 并初始化 CoreDataset
            ds_list = []
            for ds_cfg in merged_cfg["datasets_list"]:
                ds_list.append(FileDataset(
                    root_path=ds_cfg["root_path"],
                    im_list_file=ds_cfg["im_list_file"]
                ))

            self.session_datasets[session_id] = CoreDataset(
                datasets_list=ds_list,
                mode=merged_cfg.get("mode", "train"),
                imsize=merged_cfg.get("imsize", 1024),
                augment_type=merged_cfg.get("augment_type", 0),
                num_pairs=merged_cfg.get("num_pairs", 1),
                pp_type=merged_cfg.get("pp_type", None),
                pp_param=merged_cfg.get("pp_param", None),
                resize_mode=merged_cfg.get("resize_mode", None),
                crop_prob=merged_cfg.get("crop_prob", 0)
            )

        # 4. 默认设置为第一个 session
        self._current_session = list(self.session_datasets.keys())[0]

    def set_session(self, session_id):
        if session_id not in self.session_datasets:
            raise KeyError(f"Session {session_id} 不存在。可用session: {list(self.session_datasets.keys())}")
        self._current_session = session_id

    def get_current_session_id(self):
        return self._current_session

    def __getitem__(self, idx):
        return self.session_datasets[self._current_session][idx]

    def __len__(self):
        return len(self.session_datasets[self._current_session])

