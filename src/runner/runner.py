import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.core import LightningModule
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from src.runner.metric import cross_entropy


class Runner(LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig):
        super().__init__()
        self.model = model
        self.hparams.update({"dataset": f"{config.dataset.type}"})
        self.hparams.update({"model": f"{config.model.type}"})
        self.hparams.update(config.model.params)
        self.hparams.update(config.runner.dataloader.params)
        self.hparams.update({"optimizer": f"{config.runner.optimizer.params.type}"})
        self.hparams.update(config.runner.optimizer.params)
        self.hparams.update({"scheduler": f"{config.runner.scheduler.type}"})
        self.hparams.update({"scheduler_gamma": f"{config.runner.scheduler.params.gamma}"})
        self.hparams.update(config.runner.trainer.params)
        print(self.hparams)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = SGD(params=self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = MultiStepLR(
            opt, milestones=[self.hparams.max_epochs], gamma=self.hparams.scheduler_gamma
        )
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        acc = (y == prediction).float().mean()

        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()
        tqdm_dict = {"train_acc": avg_acc, "train_loss": avg_loss}
        return {**tqdm_dict, "progress_bar": tqdm_dict}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        number_of_correct_pred = torch.sum(y == prediction).item()
        return {"val_loss": loss, "n_correct_pred": number_of_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        total_count = sum([x["n_pred"] for x in outputs])
        total_n_correct_pred = sum([x["n_correct_pred"] for x in outputs])
        total_loss = torch.stack([x["val_loss"] * x["n_pred"] for x in outputs]).sum()
        val_loss = total_loss / total_count
        val_acc = total_n_correct_pred / total_count
        tqdm_dict = {"val_acc": val_acc, "val_loss": val_loss}
        return {**tqdm_dict, "progress_bar": tqdm_dict}

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        number_of_correct_pred = torch.sum(y == prediction).item()
        return {"loss": loss, "n_correct_pred": number_of_correct_pred, "n_pred": len(x)}

    def test_epoch_end(self, outputs):
        total_count = sum([x["n_pred"] for x in outputs])
        total_n_correct_pred = sum([x["n_correct_pred"] for x in outputs])
        total_loss = torch.stack([x["loss"] * x["n_pred"] for x in outputs]).sum().item()
        test_loss = total_loss / total_count
        test_acc = total_n_correct_pred / total_count
        return {"loss": test_loss, "acc": test_acc}


if __name__ == "__main__":
    # Runner Test
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from pytorch_lightning import Trainer

    from src.dataset.dataset_registry import DataSetRegistry
    from src.model.net import DeepHash

    torch.backends.cudnn.deterministic = True

    # Config
    model_params = {
        "type": "DeepHash",
        "params": {"width": 227, "height": 227, "channels": 3, "hash_bits": 48},
    }

    dataset_params = {
        "type": "CIFAR10",
        "params": {"path": {"train": "../../data/cifar10", "test": "../../data/cifar10"}},
    }

    runner_params = {
        "dataloader": {"type": "DataLoader", "params": {"num_workers": 48, "batch_size": 256}},
        "optimizer": {"type": "SGD", "params": {"learning_rate": 1e-2, "momentum": 0.9}},
        "scheduler": {"type": "MultiStepLR", "params": {"gamma": 0.1}},
        "trainer": {
            "type": "Trainer",
            "params": {"max_epochs": 128, "gpus": -1, "distributed_backend": "ddp"},
        },
    }

    model_config = OmegaConf.create(model_params)
    dataset_config = OmegaConf.create(dataset_params)
    runner_config = OmegaConf.create(runner_params)

    config = OmegaConf.create()
    config.update(model=model_config, dataset=dataset_config, runner=runner_config)
    print(config.pretty())

    # Dataset
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(227),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset = DataSetRegistry.get(config.dataset.type)
    train_dataset = dataset(
        root=config.dataset.params.path.train, train=True, transform=train_transform, download=True,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.runner.dataloader.params.batch_size,
        num_workers=config.runner.dataloader.params.num_workers,
        drop_last=True,
        shuffle=True,
    )

    test_dataset = dataset(
        root=config.dataset.params.path.test, train=False, transform=test_transform, download=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.runner.dataloader.params.batch_size,
        num_workers=config.runner.dataloader.params.num_workers,
        drop_last=False,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = DeepHash(config=config)
    model.summary()
    runner = Runner(model=model, config=config)

    # # Vanilla Loop
    # optimizer_, _ = runner.configure_optimizers()
    # optimizer = optimizer_[0]
    # runner = runner.to(device)

    # epoch_losses = []
    # for _ in range(0, 1):
    #     for j, batch in enumerate(train_dataloader):
    #         if j > 5:
    #             break
    #         batch = [x.to(device) for x in batch]
    #         loss_dict = runner.training_step(batch, j)
    #         loss = loss_dict["loss"]
    #         print(loss_dict)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     epoch_losses.append(loss.item())
    # print(epoch_losses)

    # lightning_loop
    trainer = Trainer(
        max_epochs=2,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        gpus=-1,
        early_stop_callback=False,
        checkpoint_callback=False,
        deterministic=True,
        logger=False,
        replace_sampler_ddp=False,
    )

    trainer.fit(model=runner, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)
    final_loss = trainer.running_loss.last().item()
    print(final_loss)

