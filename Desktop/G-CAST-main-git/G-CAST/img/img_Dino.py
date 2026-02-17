import os, random, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import RandomSolarize

from PIL import Image,  ImageOps
import numpy as np
from tqdm import tqdm
import copy


import wandb

from .utils import EarlyStopping

# --------------------------
# DINO-specific augmentations
# --------------------------

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DINOTransform:
    def __init__(self, global_crop_size=224, local_crop_size=96, local_crops_number=6):
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.local_crops_number = local_crops_number

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        #
        # Global crops
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=(0.4, 1.), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=(0.4, 1.), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            RandomSolarize(threshold=128, p=0.2),
            normalize,
        ])
        # Local crops
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=(0.05, 0.4), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5)),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        crops += [self.local_transfo(image) for _ in range(self.local_crops_number)]
        return crops


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None, mode=0):
        self.root_dir = root_dir
        self.transform = transform
        if mode:
            self.image_list = sorted(
                [x for x in os.listdir(root_dir) if x.endswith('.png')],
                key=lambda x: int(x.split('.')[0])
            )
        else:
            self.image_list = sorted(
                [x for x in os.listdir(root_dir) if x.endswith('.png')],
                key=lambda x: x.split('.')[0])
        if max_samples is not None:
            self.image_list = self.image_list[:max_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            crops = self.transform(img)
            return crops
        return img


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False,hidden_dim = 4096, norm_last_layer=True):
        super().__init__()
        # hidden_dim = 4096
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.requires_grad = False if norm_last_layer else True

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


#
def update_teacher(student, teacher, momentum):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data = momentum * param_t.data + (1. - momentum) * param_s.data


@torch.no_grad()
def teacher_forward(teacher_backbone, teacher_head, views):
    # return [teacher_head(teacher_backbone(v.unsqueeze(0))) for v in views]
    return [teacher_head(teacher_backbone(v)) for v in views]


def process_images(dataset_name, slide,
                   epoch_num=10,model_path="./CLIP/DLPFC/",
                   model_name="DlPfc.pth", embed_name="clip.npy"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


    # path = f"./Dataset/{dataset_name}/clip_img"
    # train_path = f"./Dataset/clip_img"
    train_path = f"./Dataset/{dataset_name}/{slide}/clip"
    transform = DINOTransform()
    dataset = CustomDataset(train_path, transform=transform, mode=1)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
                            # , num_workers=4)

    # Evaluation phase
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    eval_path =  f"./Dataset/{dataset_name}/{slide}/clip"
    # eval_dataset = CustomDataset(eval_path, transform=eval_transform, mode=1)

    # Backbone and DINO head
    student_backbone = models.resnet50(pretrained=False)
    student_backbone.fc = nn.Identity()
    teacher_backbone = copy.deepcopy(student_backbone)

    student_head = DINOHead(2048, 8192)
    teacher_head = type(student_head)(2048, 8192)
    teacher_head.load_state_dict(student_head.state_dict())
    teacher_head.to(device)

    student_backbone = student_backbone.to(device)
    teacher_backbone = teacher_backbone.to(device)
    student_head = student_head.to(device)
    teacher_head = teacher_head.to(device)

    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(student_backbone.parameters(), lr=3e-4)
    # 初始化 EarlyStopping
    path = os.path.join(model_path, model_name)
    early_stopping = EarlyStopping(patience=10, verbose=False, path=path)

    print("Start DINO Training")
    wandb.init(project="DINO", name=f"{dataset_name}", mode="offline")
    for epoch in range(epoch_num):
        epoch_loss = 0
        for crops in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            views = [crop.to(device) for crop in crops ]

            student_out = [student_head(student_backbone(v)) for v in views]
            teacher_out = teacher_forward(teacher_backbone, teacher_head, views[:2])
            teacher_out = [t.detach() for t in teacher_out]

            total_loss = 0
            temp = 0.1

            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        continue
                    logits = student_out[v] @ q.T / temp
                    labels = torch.arange(logits.shape[0]).to(device)
                    loss = nn.functional.cross_entropy(logits, labels)
                    total_loss += loss

            total_loss /= (len(teacher_out) * (len(student_out) - 1))

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            # EMA update
            momentum = 0.996
            update_teacher(student_backbone, teacher_backbone, momentum)
            update_teacher(student_head, teacher_head, momentum)

            epoch_loss += total_loss.item()
        wandb.log({"loss": epoch_loss}, step=epoch)
        early_stopping(epoch_loss, student_backbone)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        save_path = os.path.join(model_path, f"{slide}_{epoch}.pth")
        torch.save(student_backbone.state_dict(), save_path)


    print("Saving model and embeddings")
    # save_path = os.pa
    torch.save(student_backbone.state_dict(), model_name)

    # Evaluation phase
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    eval_path =  f"./Dataset/{dataset_name}/{slide}/clip"
    eval_dataset = CustomDataset(eval_path, transform=eval_transform, mode=1)
    embeddings = []
    with torch.no_grad():
        student_backbone.eval()
        for i in tqdm(range(len(eval_dataset)), desc='Eval'):
            img = eval_dataset[i].unsqueeze(0).to(device)
            feat = student_backbone(img).cpu().numpy()
            embeddings.append(feat)

    embeddings = np.vstack(embeddings)
    np.save(f'./Dataset/{dataset_name}/{slide}/{embed_name}', embeddings)
    return embeddings


def eval_dino(dataset_name, slide,model_path="./CLIP/Mouse/", model_name="DLPFC.pth", embed_name="clip.npy"):
    # path = f"./Dataset/{dataset_name}/{slide}/patch"
    eval_path = f"./Dataset/{dataset_name}/{slide}/clip"
    # path = f"./Dataset/{dataset_name}/{slide}/patch_img"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = os.path.join(model_path, model_name)
    student_backbone = models.resnet50(pretrained=True)
    student_backbone.fc = nn.Identity()
    student_backbone.load_state_dict(torch.load(path ))
    student_backbone = student_backbone.to(device)

    # Evaluation phase
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    eval_dataset = CustomDataset(eval_path, transform=eval_transform, mode=1)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    embeddings = []
    with torch.no_grad():
        student_backbone.eval()
        # for i in tqdm(range(len(eval_dataset)), desc='Eval'):
        #     img = eval_dataset[i].unsqueeze(0).to(device)
        #     feat = student_backbone(img).cpu().numpy()
        #     embeddings.append(feat)
        for batch in tqdm(eval_dataloader, desc='Eval'):
            imgs = batch.to(device)
            feats = student_backbone(imgs).cpu().numpy()
            embeddings.append(feats)

    embeddings = np.vstack(embeddings)
    np.save(f'./Dataset/{dataset_name}/{slide}/{embed_name}', embeddings)
    print(f"eval {slide}...")
    return embeddings
