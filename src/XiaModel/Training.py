from src.XiaModel.Dataset import AECDataset
from src.XiaModel.PreProcess import OnlineFDNormalizer, STFTLogScaler

from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset
import random


from tqdm.auto import tqdm, trange

class Trainer:
    def __init__(self, 
                 data_path, 
                 model,
                 lr = 1e-3,
                 seed=None,
                 n=-1,
                 hop = 160,
                 epochs=10,
                 batch_size=32,
                 window_l = 320,
                 hidden_size=322,
                 dft_size=320,
                 train_fraction=0.8,
                 device=None,
                 checkpoint_dir=None,
                 ):
        self.dataset = AECDataset(data_path, seed)

        self.train_dataset, self.test_dataset = self.split_by_guid(
            self.dataset,
            train_fraction=train_fraction,
            seed=seed
        )
        
        self.model = model
        n_features = self.dataset.crop_n

        self.FDNorm = OnlineFDNormalizer(n_features)
        
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.dtf_size = dft_size
        self.hop = hop
        self.window_l = window_l
        
        self.device = device
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        
        
        self.loss_function = MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

    def preprocess(self, batch):
        mic = torch.stack([x["mic"] for x in batch])
        lpb = torch.stack([x["lpb"] for x in batch])
        target = torch.stack([x["target"] for x in batch])

        return STFTLogScaler(
            far_mic=mic,
            far_lpb=lpb,
            near_mic=target,
            n_fft=self.dtf_size,
            hop=self.hop,
            window_l=self.window_l,
        )
        
    def target_f(self, batch_T):
        l = batch_T.size(1) / 2
        return batch_T[:,l:,:]
    
    def lpb_f(self, batch_T):
        l = batch_T.size(1) / 2
        return batch_T[:,:l,:]
 
    def train(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.preprocess,
            shuffle=True
        )

        self.model.train()

        epoch_bar = trange(
            self.epochs,
            desc="Training",
            position=0
        )
        epoch_losses = []

        for epoch_n in epoch_bar:

            batch_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch_n + 1}/{self.epochs}",
                leave=False,
                position=1
            )

            epoch_loss = 0.0

            for features, mic_mag, near_mag in batch_bar:

                features = features.to(self.device)
                mic_mag = mic_mag.to(self.device)
                near_mag = near_mag.to(self.device)

                B = features.size(0)
                T = features.size(2)

                h01, h02 = self._initial_hidden(B)

                predictions = []

                self.optimizer.zero_grad()

                for t in range(T):
                    x_t = features[:, :, t].unsqueeze(1)

                    pred_t, h01, h02 = self.model(
                        x_t,
                        h01,
                        h02
                    )

                    predictions.append(
                        pred_t.squeeze(1)
                    )

                mask = torch.stack(
                    predictions,
                    dim=2
                )

                enhanced_mag = mask * mic_mag

                loss = torch.nn.functional.mse_loss(
                    enhanced_mag,
                    near_mag
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optimizer.step()

                epoch_loss += loss.item()

                batch_bar.set_postfix(
                    loss=f"{loss.item():.5f}"
                )

            mean_loss = epoch_loss / len(train_dataloader)
            epoch_losses.append(mean_loss)

            epoch_bar.set_postfix(
                loss=f"{mean_loss:.5f}"
            )

            if self.checkpoint_dir is not None:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir / f"epoch_{epoch_n + 1}.pt"
                )

        return epoch_losses

    def split_by_guid(self, dataset, train_fraction=0.8, seed=None):
        df = dataset.df
        df = df[df["supervised"]]

        guids = list(df["guid"].unique())

        rng = random.Random(seed)
        rng.shuffle(guids)

        cut = int(len(guids) * train_fraction)

        train_guids = set(guids[:cut])
        val_guids = set(guids[cut:])

        train_idx = df.index[
            df["guid"].isin(train_guids)
        ].tolist()

        val_idx = df.index[
            df["guid"].isin(val_guids)
        ].tolist()

        return (
            Subset(dataset, train_idx),
            Subset(dataset, val_idx),
        )

    def _initial_hidden(self, batch_size): 
        h01 = torch.zeros( 1, batch_size, self.hidden_size, device=self.device, ) 
        h02 = torch.zeros( 1, batch_size, self.hidden_size, device=self.device, ) 
        return h01, h02 

    def eval(
        self,
        batch_size=None,
    ):
        if batch_size is None:
            batch_size = self.batch_size

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            collate_fn=self.preprocess,
            shuffle=False
        )

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        batch_bar = tqdm(
            test_dataloader,
            desc="Evaluating",
            leave=False
        )

        with torch.inference_mode():

            for features, mic_mag, near_mag in batch_bar:

                features = features.to(self.device)
                mic_mag = mic_mag.to(self.device)
                near_mag = near_mag.to(self.device)

                B = features.size(0)
                T = features.size(2)

                h01, h02 = self._initial_hidden(B)

                predictions = []

                for t in range(T):

                    x_t = features[:, :, t].unsqueeze(1)

                    pred_t, h01, h02 = self.model(
                        x_t,
                        h01,
                        h02
                    )

                    predictions.append(
                        pred_t.squeeze(1)
                    )

                mask = torch.stack(
                    predictions,
                    dim=2
                )

                enhanced_mag = mask * mic_mag

                loss = torch.nn.functional.mse_loss(
                    enhanced_mag,
                    near_mag
                )

                total_loss += loss.item() * B
                total_samples += B

                batch_bar.set_postfix(
                    loss=f"{loss.item():.5f}"
                )

        mean_loss = total_loss / total_samples

        self.model.train()

        return mean_loss
