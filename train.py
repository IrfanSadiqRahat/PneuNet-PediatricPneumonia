"""
Train PneuNet for pediatric pneumonia detection.
Usage: python train.py --data_dir data/chest_xray
High sensitivity optimization: weighted loss for Pneumonia class.
"""
import argparse, torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import PneuNet

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/chest_xray")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tfm = {
        "train": transforms.Compose([
            transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10, translate=(0.05,0.05)),
            transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)]),
        "val": transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)]),
    }
    loaders = {s: DataLoader(datasets.ImageFolder(f"{args.data_dir}/{s}", tfm[s]),
               args.batch_size, shuffle=(s=="train"), num_workers=4, pin_memory=True)
               for s in ("train","val")}

    model = PneuNet(num_classes=2).to(device)
    # Weighted CE: up-weight Pneumonia (class 1) for high sensitivity
    weights = torch.tensor([1.0, 2.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_sensitivity = 0.0
    for epoch in range(1, args.epochs+1):
        for phase in ("train","val"):
            model.train() if phase=="train" else model.eval()
            tp = fp = tn = fn = 0
            for imgs, labels in loaders[phase]:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.set_grad_enabled(phase=="train"):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                    if phase=="train":
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                preds = out.argmax(1)
                tp += ((preds==1)&(labels==1)).sum().item()
                fp += ((preds==1)&(labels==0)).sum().item()
                tn += ((preds==0)&(labels==0)).sum().item()
                fn += ((preds==0)&(labels==1)).sum().item()
            if phase=="val":
                sensitivity = tp/(tp+fn+1e-8)
                specificity = tn/(tn+fp+1e-8)
                accuracy    = (tp+tn)/(tp+tn+fp+fn+1e-8)
                print(f"Epoch {epoch:3d} | acc={accuracy:.4f} "
                      f"sens={sensitivity:.4f} spec={specificity:.4f}")
                if sensitivity > best_sensitivity:
                    best_sensitivity = sensitivity
                    torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
                    print(f"  ✅ Best sensitivity={best_sensitivity:.4f}")
        scheduler.step()

if __name__=="__main__": main()
