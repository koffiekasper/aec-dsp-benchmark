from src.XiaModel.Dataset import FarEndSingleTalkDataset
from src.XiaModel.PreProcess import OnlineFDNormalizer, STFTLogScaler, InverseSTFT

from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split


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
        self.dataset = FarEndSingleTalkDataset(data_path, seed, n)

        n_train = int(len(self.dataset) * train_fraction)
        n_test = len(self.dataset) - n_train
        if seed:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        self.train_dataset, self.test_dataset = random_split(
            self.dataset,
            [n_train, n_test] ,
            generator=generator
        )
        
        self.model = model
        n_features = self.dataset[0]['length']
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
        far_lpb = torch.stack([
            d["farend_lpb"] for d in batch
        ])

        far_mic = torch.stack([
            d["farend_mic"] for d in batch
        ])

        near_mic = torch.stack([
            d["nearend_mic"] for d in batch
        ])

        B = far_mic.size(0)

        ser_db = torch.empty(B, 1).uniform_(-10, 10)

        near_rms = near_mic.square().mean(
            dim=1,
            keepdim=True
        ).sqrt()

        echo_rms = far_mic.square().mean(
            dim=1,
            keepdim=True
        ).sqrt()

        alpha = (
            near_rms
            / (
                echo_rms
                * 10 ** (ser_db / 20)
                + 1e-8
            )
        )

        mixed_mic = (
            near_mic
            + alpha * far_mic
        )

        return STFTLogScaler(
            far_mic=mixed_mic,
            far_lpb=far_lpb,
            near_mic=near_mic,
            n_fft=self.dtf_size,
            hop=self.hop,
            window_l=self.window_l
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
