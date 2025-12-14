from dataset import OHLCVWindowDataset

ds = OHLCVWindowDataset(
    data_path="../data_out/BTC_USDT_USDT-1h-futures.feather",
    window=48
)

print(len(ds))           # > 1000
x, y = ds[0]
print(x.shape)           # torch.Size([48, feature_dim])
print(x.dtype, y)        # torch.float32, int


