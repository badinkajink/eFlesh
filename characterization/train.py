import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class SensorForceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        states_csv: str,
        sensor_csv: str,
        target_col: int = 4,
        x_mean=None, x_std=None,
        y_mean=None, y_std=None,
        normalize_x: bool = True,
        normalize_y: bool = True,
    ):
        states_time, states_val = [], []
        with open(states_csv, "r") as f:
            for row in csv.reader(f):
                vals = [float(x) for x in row]
                if len(vals) <= target_col:
                    continue
                if vals[target_col] != -1:
                    states_time.append(vals[0])
                    states_val.append(vals[target_col])
        states_time = np.asarray(states_time, dtype=np.float64)
        states_val = np.asarray(states_val, dtype=np.float64)

        sensor_time, sensor_data = [], []
        with open(sensor_csv, "r") as f:
            for row in csv.reader(f):
                vals = [float(x) for x in row]
                sensor_time.append(vals[0])
                sensor_data.append(vals[1:])
        sensor_time = np.asarray(sensor_time, dtype=np.float64)
        sensor_data = np.asarray(sensor_data, dtype=np.float64)

        matched_sens, matched_y = [], []
        for i in range(len(sensor_time)):
            t = sensor_time[i]
            idx = np.argmin(np.abs(states_time - t))
            matched_sens.append(sensor_data[i])
            matched_y.append(states_val[idx])

        self.X = np.asarray(matched_sens, dtype=np.float32)
        self.Y = np.asarray(matched_y, dtype=np.float32).reshape(-1, 1)

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        if self.normalize_x:
            if x_mean is None or x_std is None:
                x_mean = self.X.mean(axis=0)
                x_std = self.X.std(axis=0)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            self.x_mean = x_mean.astype(np.float32)
            self.x_std = x_std.astype(np.float32)
            self.X = (self.X - self.x_mean) / self.x_std
        else:
            self.x_mean = np.zeros(self.X.shape[1], dtype=np.float32)
            self.x_std = np.ones(self.X.shape[1], dtype=np.float32)

        if self.normalize_y:
            if y_mean is None or y_std is None:
                y_mean = self.Y.mean(axis=0)
                y_std = self.Y.std(axis=0)
            y_std = np.where(y_std < 1e-8, 1.0, y_std)
            self.y_mean = y_mean.astype(np.float32)
            self.y_std = y_std.astype(np.float32)
            self.Y = (self.Y - self.y_mean) / self.y_std
        else:
            self.y_mean = np.zeros(1, dtype=np.float32)
            self.y_std = np.ones(1, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y):
        if isinstance(y, torch.Tensor):
            return y * torch.tensor(self.y_std, device=y.device) + torch.tensor(self.y_mean, device=y.device)
        return y * self.y_std + self.y_mean

class SensorSpatialDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        states_csv: str,
        sensor_csv: str,
        z_thresh: float = 145.1,
        x_mean=None, x_std=None,
        y_mean=None, y_std=None,
        normalize_x: bool = True,
        normalize_y: bool = True,
    ):
        states_time, states_xyz = [], []
        with open(states_csv, "r") as f:
            for row in csv.reader(f):
                row = [float(v) for v in row]
                states_time.append(row[0])
                states_xyz.append(row[1:4])
        states_time = np.asarray(states_time, dtype=np.float64)
        states_xyz = np.asarray(states_xyz, dtype=np.float64)

        sensor_time, sensor_data = [], []
        with open(sensor_csv, "r") as f:
            for row in csv.reader(f):
                row = [float(v) for v in row]
                sensor_time.append(row[0])
                sensor_data.append(row[1:])
        sensor_time = np.asarray(sensor_time, dtype=np.float64)
        sensor_data = np.asarray(sensor_data, dtype=np.float64)

        matched_xyz, matched_sens, matched_mask = [], [], []
        for i in range(len(sensor_time)):
            st = sensor_time[i]
            idx = np.argmin(np.abs(states_time - st))
            xyz = states_xyz[idx]
            matched_xyz.append(xyz)
            matched_sens.append(sensor_data[i])
            matched_mask.append(xyz[2] < z_thresh)

        matched_xyz = np.asarray(matched_xyz, dtype=np.float64)
        matched_sens = np.asarray(matched_sens, dtype=np.float64)
        matched_mask = np.asarray(matched_mask, dtype=bool)

        self.X = matched_sens[matched_mask].astype(np.float32)
        self.Y = matched_xyz[matched_mask].astype(np.float32)

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        if self.normalize_x:
            if x_mean is None or x_std is None:
                x_mean = self.X.mean(axis=0)
                x_std = self.X.std(axis=0)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            self.x_mean = x_mean.astype(np.float32)
            self.x_std = x_std.astype(np.float32)
            self.X = (self.X - self.x_mean) / self.x_std
        else:
            self.x_mean = np.zeros(self.X.shape[1], dtype=np.float32)
            self.x_std = np.ones(self.X.shape[1], dtype=np.float32)

        if self.normalize_y:
            if y_mean is None or y_std is None:
                y_mean = self.Y.mean(axis=0)
                y_std = self.Y.std(axis=0)
            y_std = np.where(y_std < 1e-8, 1.0, y_std)
            self.y_mean = y_mean.astype(np.float32)
            self.y_std = y_std.astype(np.float32)
            self.Y = (self.Y - self.y_mean) / self.y_std
        else:
            self.y_mean = np.zeros(3, dtype=np.float32)
            self.y_std = np.ones(3, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y):
        if isinstance(y, torch.Tensor):
            return y * torch.tensor(self.y_std, device=y.device) + torch.tensor(self.y_mean, device=y.device)
        return y * self.y_std + self.y_mean

def fit(
    dataset_full,
    out_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(dataset_full.X)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    split = int(0.8 * n)
    train_idx, val_idx = idxs[:split], idxs[split:]

    x_mean = dataset_full.X[train_idx].mean(axis=0)
    x_std = dataset_full.X[train_idx].std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    y_mean = dataset_full.Y[train_idx].mean(axis=0)
    y_std = dataset_full.Y[train_idx].std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    if out_dim == 1:
        dataset = SensorForceDataset(
            states_csv=dataset_full._states_csv,
            sensor_csv=dataset_full._sensor_csv,
            target_col=dataset_full._target_col,
            x_mean=x_mean, x_std=x_std,
            y_mean=y_mean, y_std=y_std,
            normalize_x=True, normalize_y=True,
        )
    else:
        dataset = SensorSpatialDataset(
            states_csv=dataset_full._states_csv,
            sensor_csv=dataset_full._sensor_csv,
            z_thresh=dataset_full._z_thresh,
            x_mean=x_mean, x_std=x_std,
            y_mean=y_mean, y_std=y_std,
            normalize_x=True, normalize_y=True,
        )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False
    )

    model = MLP(in_dim=dataset.X.shape[1], out_dim=out_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    pbar = tqdm(range(1, epochs + 1), desc="Training", ncols=150)
    for e in pbar:
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for Xb, Yb in train_loader:
            Xb = Xb.float().to(device)
            Yb = Yb.float().to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * Xb.size(0)
            train_count += Xb.size(0)
        train_mse = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum, val_count = 0.0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb = Xb.float().to(device)
                Yb = Yb.float().to(device)
                pred = model(Xb)
                loss = criterion(pred, Yb)
                val_loss_sum += loss.item() * Xb.size(0)
                val_count += Xb.size(0)
                all_pred.append(pred.cpu())
                all_true.append(Yb.cpu())
        val_mse = val_loss_sum / max(1, val_count)

        pred = torch.cat(all_pred, dim=0)
        true = torch.cat(all_true, dim=0)

        pred_real = torch.from_numpy(dataset.unnormalize_y(pred.numpy()))
        true_real = torch.from_numpy(dataset.unnormalize_y(true.numpy()))
        d = pred_real - true_real

        if out_dim == 1:
            rmse = torch.sqrt(torch.mean(d[:, 0] ** 2)).item()
            pbar.set_postfix({
                'Train MSE': f'{train_mse:.4f}',
                'Val MSE': f'{val_mse:.4f}',
                'RMSE': f'{rmse:.3f}g'
            })
        else:
            per_axis_rmse = torch.sqrt(torch.mean(d ** 2, dim=0))
            euclid_rmse = torch.sqrt(torch.mean(torch.sum(d ** 2, dim=1)))
            rx, ry, rz = (per_axis_rmse[0].item(), per_axis_rmse[1].item(), per_axis_rmse[2].item())
            pbar.set_postfix({
                'RMSE_x': f'{rx:.2f}mm',
                'RMSE_y': f'{ry:.2f}mm', 
                'RMSE_z': f'{rz:.2f}mm',
                'Net': f'{euclid_rmse:.2f}mm'
            })

    return model, (x_mean, x_std, y_mean, y_std)

def main():
    parser = argparse.ArgumentParser(description="eFlesh characterization training")
    parser.add_argument("--mode", choices=["spatial", "normal", "shear"], required=True)
    parser.add_argument("--folder", type=str, required=True, help="Path to a dataset folder in characterization/datasets/")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target_col", type=int, default=4, help="Force column in states.csv (use if normal/shear columns differ)")
    parser.add_argument("--z_thresh", type=float, default=145.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    states_csv = os.path.join(args.folder, "states.csv")
    sensor_csv = os.path.join(args.folder, "sensor_post_baselines.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "spatial":
        full = SensorSpatialDataset(
            states_csv, sensor_csv, z_thresh=args.z_thresh,
            normalize_x=False, normalize_y=False,
        )
        full._states_csv = states_csv
        full._sensor_csv = sensor_csv
        full._z_thresh = args.z_thresh
        out_dim = 3
    else:
        full = SensorForceDataset(
            states_csv, sensor_csv, target_col=args.target_col,
            normalize_x=False, normalize_y=False,
        )
        full._states_csv = states_csv
        full._sensor_csv = sensor_csv
        full._target_col = args.target_col
        out_dim = 1

    model, stats = fit(
        dataset_full=full,
        out_dim=out_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    os.makedirs(os.path.join(args.folder, "artifacts"), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mode": args.mode,
            "out_dim": out_dim,
            "x_mean": stats[0],
            "x_std": stats[1],
            "y_mean": stats[2],
            "y_std": stats[3],
        },
        os.path.join(args.folder, "artifacts", f"eflesh_{args.mode}_mlp128.pt"),
    )

if __name__ == "__main__":
    main()
