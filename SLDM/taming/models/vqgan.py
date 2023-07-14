import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from taming.models.vqvae import VQSub
from omegaconf import OmegaConf
import numpy as np
import os
import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="pixel_values",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.learning_rate = 1e-4
        self.loss = instantiate_from_config(lossconfig)
        # convert omegaconf to python serilizeable obj
        ddconfig = OmegaConf.to_object(ddconfig)
        self.disc_conditional = OmegaConf.to_object(lossconfig)["params"]["disc_conditional"]
        
        self.vqvae = VQSub(**ddconfig)
        self.automatic_optimization = False

        self.frequency = 1
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.use_SPADE = ddconfig["use_SPADE"]
        # print("use spade:", self.use_SPADE)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, input, segmap=None):
        dec, diff = self.vqvae(input, segmap)
        return dec, diff

    def get_input(self, batch, k):
        x = batch["pixel_values"]
        if self.use_SPADE:
            y = batch["segmap"]
            y = self.preprocess_input(y, 34)
            y = y.float()
        else:
            y = None

        if len(x.shape) == 3:
            x = x[..., None]
        x = x.float()
        #x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, y)
        opt_ae, opt_disc = self.optimizers()

        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), cond=y if self.disc_conditional else None, split="train")

        #self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()
        
        # discriminator
        discloss, log_dict_disc = self.loss(qloss, x.detach(), xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), cond=y if self.disc_conditional else None, split="train")
        #self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, y)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(),cond=y if self.disc_conditional else None, split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(),cond=y if self.disc_conditional else None, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #            prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss,
        #            prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(self.vqvae.parameters(),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.vqvae.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        #assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def preprocess_input(self, data, num_classes):
        # move to GPU and change data types
        data['label'] = data['label'].long()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        input_label = torch.FloatTensor(bs, num_classes, h, w).zero_().to(data['label'].device)
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if 'instance' in data:
            # inst_map = data['instance']
            # instance_edge_map = self.get_edges(inst_map)
            # input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)
            b_msk = (data['instance'] != 0)
            # binary mask to gpu-device and turn type to float
            instance_edge_map = b_msk.to(data['instance'].device).float()
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_().to(t.device)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def on_train_epoch_end(self):
        """Tries to save current checkpoint at the end of each train epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = self.current_epoch # type: ignore
        if epoch % self.frequency == 0:
            #self.log_images()
            self.vqvae.save_pretrained(os.path.join(f"./SPADE_VQ_model_V2/{epoch}ep", "vqvae"))