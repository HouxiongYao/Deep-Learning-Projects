
class Config:
    def __init__(self):
        self.data_dir = "./Dataset"
        self.save_path = "model.ckpt"
        self.batch_size = 64
        self.n_workers = 8
        self.valid_steps = 2000
        self.warmup_steps = 1000
        self.save_steps = 10000
        self.total_steps = 700

    def __repr__(self):
        return f"Config(data_dir={self.data_dir}, save_path={self.save_path}, batch_size={self.batch_size}, " \
               f"n_workers={self.n_workers}, valid_steps={self.valid_steps}, warmup_steps={self.warmup_steps}, " \
               f"save_steps={self.save_steps}, total_steps={self.total_steps})"