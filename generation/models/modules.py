import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .GAN3D import Discriminator, Generator


class GAN3DModule(pl.LightningModule):
    def __init__(self, latent_dim=100, channels=32, lr=2e-4, betas=(0.5, 0.999), block_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(latent_dim, channels)
        self.discriminator = Discriminator(channels)
    

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx):
        real_images = batch[0]
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim, 1, 1, 1, device=self.device)

        opt_g, opt_d = self.optimizers()

        # train generator
        self.toggle_optimizer(opt_g)
        fake_images = self(z)
        y = torch.ones(batch_size, device=self.device)
        g_loss = self.adversarial_loss(self.discriminator(fake_images), y)
        self.log('g_loss', g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # train discriminator
        self.toggle_optimizer(opt_d)
        valid = torch.ones(batch_size, device=self.device)
        real_loss = self.adversarial_loss(self.discriminator(real_images), valid)
        fake = torch.zeros(batch_size, device=self.device)
        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        self.log('d_loss', d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = (self.hparams.beta1, self.hparams.beta2)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return [opt_g, opt_d], []
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []
    
    def on_epoch_end(self):
        z = torch.randn(16, self.hparams.latent_dim, 1, 1, 1, device=self.device)
        fake_objects = self(z)
        self.save_objects(fake_objects, f"generated_objects_epoch_{self.current_epoch}.binvox")

    def save_objects(self, objects, file_path):
        with open(file_path, "wb") as f:
            for obj in objects:
                f.write(obj.cpu().numpy().tobytes())

    def show_3d_objects(self, objects: np.array):
        # display 3D voxels with matplotlib
        fig = plt.figure(figsize=(8, 8))
        for i, ob in enumerate(objects):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            ax.voxels(ob, edgecolor="k")
            print(f"Plotting object {i + 1} of {len(objects)}")
        plt.show()



if __name__ == "__main__":
    model = GAN3DModule.load_from_checkpoint("lightning_logs/version_12/checkpoints/epoch=651-step=71720.ckpt")
    model.eval()

    z = torch.randn(4, model.hparams.latent_dim, 1, 1, 1, device=model.device)
    fake_objects = model(z)
    print(fake_objects.shape)
    model.show_3d_objects(torch.round(fake_objects).detach().cpu().numpy())