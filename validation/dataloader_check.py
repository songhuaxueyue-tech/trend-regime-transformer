import torch
from torch.utils.data import DataLoader
from scripts.dataset import OHLCVWindowDataset

DATA_PATH = "../data_out/BTC_USDT_USDT-1h-futures.feather"

def main():
    dataset = OHLCVWindowDataset(
        data_path=DATA_PATH,
        window=48,
        normalize=True
    )

    dl = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    x, y = next(iter(dl))
    print("[OK] DataLoader works")
    print("x shape:", x.shape)   # (32, 48, feature_dim)
    print("y shape:", y.shape)   # (32,)

if __name__ == "__main__":
    main()




