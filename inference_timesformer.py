import os
import random
from datetime import datetime

import numpy as np
import scipy.stats as st
import cv2
import gc
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import glob
import PIL.Image
from warmup_scheduler import GradualWarmupScheduler
from tqdm.auto import tqdm
from tap import Tap

PIL.Image.MAX_IMAGE_PIXELS = 933120000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class InferenceArgumentParser(Tap):
    segment_id: list[str] = ['20231210121321']
    segment_path: str = './scrolls'
    model_path: str = './checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'
    out_path: str = "./predictions"
    stride: int = 8
    start: int = 32
    num_layers: int = 5
    workers: int = 2
    batch_size: int = 64
    size: int = 64
    reverse: int = 0
    device: str = 'cuda'


# Parse arguments
args = InferenceArgumentParser().parse_args()


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


class CFG:
    # ============== File structure =============
    comp_name = 'vesuvius'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = './'
    exp_name = 'pretraining_all'

    # ==============  Model config  =============
    num_layers = args.num_layers
    encoder_depth = 5

    # ==============  Training config  =============
    size = 64
    tile_size = 64
    stride = tile_size // 3
    train_batch_size = 256
    valid_batch_size = 256
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 3
    # Best results between 3-5 epochs, too many and you will start over-fitting

    # AdamW warmup
    warmup_factor = 10
    lr = 1e-4 / warmup_factor
    min_lr = 1e-6
    num_workers = 16
    seed = 42

    # ============== Augmentation =============
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean = [0] * num_layers,
            std = [1] * num_layers
        ),
        ToTensorV2(transpose_mask=True),
    ]


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def cfg_init(cfg, mode='val'):
    set_seed(cfg.seed)


def read_images(fragment_id,start_idx,end_idx,rotation=0):
    layer_format = 'tif'
    valid_formats = ['tif', 'jpg', 'png']
    detected_format = None
    for ext in valid_formats:
        test_path = f"{args.segment_path}/{fragment_id}/layers/{start_idx:02}.{ext}"
        if os.path.exists(test_path):
            detected_format = ext
            break
    if detected_format is None:
        raise FileNotFoundError("No valid .tif, .jpg, or .png files found.")
    layer_format = detected_format
    
    images = []
    idxs = range(start_idx, end_idx)

    print()
    for i in idxs:
        print(f"Loading image {(i+1) - start_idx} of {end_idx - start_idx}...")
        print(f"{args.segment_path}/{fragment_id}/layers/{i:02}.{layer_format}")
        image = cv2.imread(f"{args.segment_path}/{fragment_id}/layers/{i:02}.{layer_format}", 0)
        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image,0,200)
        images.append(image)
    print(f"Successfully loaded {end_idx - start_idx} images.")

    images = np.stack(images, axis=2)

    if args.reverse != 0 or fragment_id in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:
        print("Reverse Segment")
        images = images[:,:,::-1]
    
    fragment_mask = None
    wildcard_path_mask = f"{args.segment_path}/{fragment_id}/*_mask.png"
    if os.path.exists(f"{args.segment_path}/{fragment_id}/{fragment_id}_mask.png"):
        fragment_mask = cv2.imread(CFG.comp_dataset_path + f"{args.segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    elif len(glob.glob(wildcard_path_mask)) > 0:
        # any *mask.png exists
        mask_path = glob.glob(wildcard_path_mask)[0]
        fragment_mask = cv2.imread(mask_path, 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        # White mask
        fragment_mask = np.ones_like(images[:,:,0]) * 255

    return images, fragment_mask


def get_img_splits(fragment_id,start_idx,end_idx,rotation=0):
    images = []
    xyxys = []
    image, fragment_mask = read_images(fragment_id,start_idx,end_idx,rotation)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2] == 0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])
    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean = [0] * CFG.num_layers,
            std = [1] * CFG.num_layers
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(
        test_dataset,
        batch_size = CFG.valid_batch_size,
        shuffle = False,
        num_workers = CFG.num_workers, pin_memory=False, drop_last=False,
    )

    return test_loader, np.stack(xyxys),(image.shape[0],image.shape[1]),fragment_mask


def get_transforms(data, cfg):
    aug = None
    if data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, cfg, transform=None):
        self.images = images
        self.xyxys = xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        return image, xy


class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=64,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters()
        self.mask_prediction = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.ones(self.hparams.pred_shape)
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        self.backbone=TimeSformer(
                dim = 512,
                image_size = 64,
                patch_size = 16,
                num_frames = 5, # useless? 
                num_classes = 16,
                channels=1,
                depth = 8,
                heads = 6,
                dim_head =  64,
                attn_dropout = 0.1,
                ff_dropout = 0.1
            )
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:,None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        x = self.backbone(torch.permute(x, (0, 2, 1,3,4)))
        x = x.view(-1,1,4,4)
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/Arcface_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys = batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_prediction[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/MSE_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=CFG.lr)
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer],[scheduler]


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=1e-6
    )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine
    )
    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)


def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_prediction = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    kernel = gkern(CFG.size,1)
    kernel = kernel / kernel.max()
    model.eval()

    for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_prediction[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy(),kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    # adding a small epsilon value prevents divide by zero errors
    mask_count[mask_count == 0] = np.finfo(float).eps

    mask_prediction /= mask_count
    return mask_prediction


if __name__ == "__main__":
    print()
    print("Initialising...")

    # Initialise config variables
    cfg_init(CFG)

    # CUDA checks
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device type: {device}")
        print(f"Device count: {torch.cuda.device_count()}")
        #torch.print_cuda_info()
    else:
        device = torch.device('cpu')
        print(f"Device type: {device}")
        print("WARNING: CUDA not available.")
    print()

    model = RegressionPLModel.load_from_checkpoint(args.model_path, strict=False)
    model.cuda()
    model.eval()
    wandb.init(
        project="Vesuvius", 
        name=f"ALL_scrolls_tta", 
    )

    for fragment_id in args.segment_id:
        if glob.glob(f"{args.segment_path}/{fragment_id}/layers/*.*"):
            predictions = []
            for r in [0]:
                for i in [args.start]:
                    start_f = i
                    end_f = start_f + CFG.num_layers
                    test_loader, test_xyxz, test_shape, fragment_mask = get_img_splits(fragment_id, start_f, end_f, r)
                    mask_prediction = predict_fn(test_loader, model, device, test_xyxz,test_shape)
                    mask_prediction = np.clip(np.nan_to_num(mask_prediction), a_min=0, a_max=1)
                    mask_prediction /= mask_prediction.max()

                    predictions.append(mask_prediction)

            img = wandb.Image(
                predictions[0],
                caption=f"{fragment_id}"
            )
            wandb.log({'predictions': img})
            gc.collect()

            if len(args.out_path) > 0:
                # CV2 image
                image_cv = (mask_prediction * 255).astype(np.uint8)
                try:
                    os.makedirs(f"{args.out_path}/{fragment_id}", exist_ok=True)
                except:
                    pass
                print()
                print("Saving predictions...")
                # Generate a timestamp for this inference
                now = datetime.now()
                current_time = now.strftime("%Y%m%d%H%M%S")
                cv2.imwrite(os.path.join(
                    f"{args.out_path}/{fragment_id}",
                    f"{fragment_id}_prediction_n{args.num_layers}s{start_f}e{end_f-1}_{current_time}.png"),
                    image_cv
                )
                print("Done.")
                print()
                #output_path = f"{args.out_path}/{fragment_id}"
                #output_path = os.path.realpath(output_path)
                #os.startfile
        else:
            print("ERROR: Could not find a valid layer file to run inference on.")
    del mask_prediction, test_loader, model
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()
    print()
