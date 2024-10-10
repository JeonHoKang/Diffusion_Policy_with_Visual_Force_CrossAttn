import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='config', config_name='config')
def my_app(cfg: DictConfig):
    print(f"Database host: {cfg.database.host}")
    print(f"Database port: {cfg.database.port}")
    print(f"User name: {cfg.user.name}")
    print(f"User age: {cfg.user.age}")

if __name__ == "__main__":
    my_app()