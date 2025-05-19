import numpy as np
import xarray as xr
import tensorflow as tf

class CustomDataset(tf.keras.utils.Sequence):
    # def __init__(self, modis, improve, indices, batch_size=128):
    #     self.modis = modis[indices]
    #     self.improve = improve[indices]
    #     self.batch_size = batch_size

    # def __len__(self):
    #     return int(np.ceil(len(self.modis) / self.batch_size))

    # def __getitem__(self, idx):
    #     batch_modis = self.modis[idx * self.batch_size : (idx + 1) * self.batch_size]
    #     batch_improve = self.improve[idx * self.batch_size : (idx + 1) * self.batch_size]
    #     return batch_modis, batch_improve
    def __init__(self, modis, indices, batch_size=128):
        self.data = modis[indices]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_data


def load_IMPROVE():
    file = "../../data/improve_monthly_gridded.nc"
    ds = xr.open_dataset(file)
    df = ds.to_dataframe().reset_index()
    print(df.head(), df.columns)

    feature_vars = [v for v in ds.data_vars if all(dim in ds[v].dims for dim in ['lat_bin', 'lon_bin'])]
    print(f"Loading features: {feature_vars}")

    feature_arrays = []
    for var in feature_vars:
        var_data = ds[var]
        if 'year_ts' in var_data.dims and 'month_ts' in var_data.dims:
            var_data = var_data.stack(time=("year_ts", "month_ts"))
            var_data = var_data.transpose("time", "lat_bin", "lon_bin")
        elif 'time' in var_data.dims:
            var_data = var_data.transpose("time", "lat_bin", "lon_bin")
        else:
            raise ValueError(f"Unexpected time dimensions in variable {var}")
        feature_arrays.append(var_data.values)

    data = np.stack(feature_arrays, axis=-1)  # shape (T, H, W, C)
    print(f"Loaded IMPROVE data shape: {data.shape}")
    return data


def load_MODIS():
    file = "../../data/AOD_Dark_Target_Deep_Blue_Mean_Mean.npy"
    data = np.load(file)
    print(f"Loaded MODIS data shape: {data.shape}")
    return data


def get_dataloaders(batch_size=128, shuffle=True, split=0.8):
    modis = load_MODIS()
    # improve = load_IMPROVE()

    # Reshape MODIS to match (T, H, W, 1) if needed
    if modis.ndim == 3:
        modis = modis[..., np.newaxis]

    n_samples = modis.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(split * n_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # train_dataset = CustomDataset(modis, train_indices, batch_size=batch_size)
    # val_dataset = CustomDataset(modis, val_indices, batch_size=batch_size)

    return modis[train_indices], modis[val_indices]
