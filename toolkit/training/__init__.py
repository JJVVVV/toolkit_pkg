# from .components import Optimizer, Scaler, Scheduler
# from .dataloader import get_dataloader, gradient_accumulate
# from .loss_functions import cos_loss, kl_loss, mse_loss


def lazy_import_Trainer():
    from .trainer import Trainer

    globals()["Trainer"] = Trainer


def lazy_import_initialize():
    from .initializer import initialize

    globals()["initialize"] = initialize


def lazy_import_Evaluator():
    from .evaluator import Evaluator

    globals()["Evaluator"] = Evaluator


def lazy_import_WatchDog():
    from .watchdog import WatchDog

    globals()["WatchDog"] = WatchDog


def lazy_import_ckptManager():
    from .checkpoint_manager import CheckpointManager

    globals()["CheckpointManager"] = CheckpointManager


# __getattr__ 是Python 3.7及以上的特性，用于延迟加载模块
def __getattr__(name):
    match name:
        case "Trainer":
            lazy_import_Trainer()
            return globals()[name]
        case "initialize":
            lazy_import_initialize()
            return globals()[name]
        case "Evaluator":
            lazy_import_Evaluator()
            return globals()[name]
        case "WatchDog":
            lazy_import_WatchDog()
            return globals()[name]
        case "CheckpointManager":
            lazy_import_ckptManager()
            return globals()[name]
        case _:
            raise AttributeError(f"module {__name__} has no attribute {name}")
