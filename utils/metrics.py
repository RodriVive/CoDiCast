import numpy as np


def lat_weighted_rmse_one_var(Y_true, Y_pred, var_idx, resolution):
    """
    Y_pred: (N, H, W, C) 
    Y_true: (N, H, W, C) 
    var_idx: int, pick one variable --> (N, H, W)
    resolution: int
    """
    if resolution == 5.625:
        latitudes = np.array([-87.1875, -81.5625, -75.9375, -70.3125, -64.6875, -59.0625, -53.4375,
                              -47.8125, -42.1875, -36.5625, -30.9375, -25.3125, -19.6875, -14.0625,
                              -8.4375,  -2.8125,   2.8125,   8.4375,  14.0625,  19.6875,  25.3125,
                              30.9375,  36.5625,  42.1875,  47.8125,  53.4375,  59.0625,  64.6875,
                              70.3125,  75.9375,  81.5625,  87.1875
                             ]
                            )
    
    if resolution == 2.8125:
        latitudes = np.array([-88.59375, -85.78125, -82.96875, -80.15625, -77.34375, -74.53125,
                              -71.71875, -68.90625, -66.09375, -63.28125, -60.46875, -57.65625,
                              -54.84375, -52.03125, -49.21875, -46.40625, -43.59375, -40.78125,
                              -37.96875, -35.15625, -32.34375, -29.53125, -26.71875, -23.90625,
                              -21.09375, -18.28125, -15.46875, -12.65625, -9.84375, -7.03125,
                              -4.21875, -1.40625, 1.40625, 4.21875, 7.03125, 9.84375,
                              12.65625, 15.46875, 18.28125, 21.09375, 23.90625, 26.71875,
                              29.53125, 32.34375, 35.15625, 37.96875, 40.78125, 43.59375,
                              46.40625, 49.21875, 52.03125, 54.84375, 57.65625, 60.46875,
                              63.28125, 66.09375, 68.90625, 71.71875, 74.53125, 77.34375,
                              80.15625, 82.96875, 85.78125, 88.59375
                             ]
                            )
    if resolution == 1.40625:
        latitudes = np.array([-89.296875, -87.890625, -86.484375, -85.078125, -83.671875, -82.265625,
                             -80.859375, -79.453125, -78.046875, -76.640625, -75.234375, -73.828125,
                             -72.421875, -71.015625, -69.609375, -68.203125, -66.796875, -65.390625,
                             -63.984375, -62.578125, -61.171875, -59.765625, -58.359375, -56.953125,
                             -55.546875, -54.140625, -52.734375, -51.328125, -49.921875, -48.515625,
                             -47.109375, -45.703125, -44.296875, -42.890625, -41.484375, -40.078125,
                             -38.671875, -37.265625, -35.859375, -34.453125, -33.046875, -31.640625,
                             -30.234375, -28.828125, -27.421875, -26.015625, -24.609375, -23.203125,
                             -21.796875, -20.390625, -18.984375, -17.578125, -16.171875, -14.765625,
                             -13.359375, -11.953125, -10.546875,  -9.140625,  -7.734375,  -6.328125,
                             -4.921875,  -3.515625,  -2.109375,  -0.703125,   0.703125,   2.109375,
                             3.515625,   4.921875,   6.328125,   7.734375,   9.140625,  10.546875,
                             11.953125,  13.359375,  14.765625,  16.171875,  17.578125,  18.984375,
                             20.390625,  21.796875,  23.203125,  24.609375,  26.015625,  27.421875,
                             28.828125,  30.234375,  31.640625,  33.046875,  34.453125,  35.859375,
                             37.265625,  38.671875,  40.078125,  41.484375,  42.890625,  44.296875,
                             45.703125,  47.109375,  48.515625,  49.921875,  51.328125,  52.734375,
                             54.140625,  55.546875,  56.953125,  58.359375,  59.765625,  61.171875,
                             62.578125,  63.984375,  65.390625,  66.796875,  68.203125,  69.609375,
                             71.015625,  72.421875,  73.828125,  75.234375,  76.640625,  78.046875,
                             79.453125,  80.859375,  82.265625,  83.671875,  85.078125,  86.484375,
                             87.890625,  89.296875
                             ]
                            )
        
    # Convert latitudes to radians for cosine calculation  (H,)
    lat_radians = np.radians(latitudes)
    cosine_lat_weights = np.cos(lat_radians)

    # Normalize weights
    cosine_lat_weights /= cosine_lat_weights.mean()  # len(cosine_lat): divided by H

    # Calculate weighted squared differences  (N, H, W)
    squared_diffs = (Y_pred[:, :, :, var_idx] - Y_true[:, :, :, var_idx]) ** 2
    weighted_squared_diffs = squared_diffs * cosine_lat_weights[np.newaxis, :, np.newaxis] # (1, H, 1)
    
    # Calculate mean squared error for each sample (N, )
    mse_per_sample = np.mean(weighted_squared_diffs, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample) 
    
    # Calculate overall RMSE by averaging across all samples (1, )
    rmse = rmse_per_sample.mean()

    return rmse


def lat_weighted_acc_one_var(Y_true, Y_pred, var_idx, resolution, clim):
    """
    Computes Anomaly Correlation Coefficient (ACC)
    
    Y_pred, Y_true: (N, H, W, C)
    clim: (H, W)
    """
    if resolution == 5.625:
        latitudes = np.array([...])  # same as before
    elif resolution == 2.8125:
        latitudes = np.array([...])  # same as before
    elif resolution == 1.40625:
        latitudes = np.array([...])  # same as before
    else:
        raise ValueError("Unsupported resolution")

    # Cosine latitude weights (H,)
    lat_radians = np.radians(latitudes)
    cosine_lat_weights = np.cos(lat_radians)
    cosine_lat_weights /= cosine_lat_weights.mean()  # normalize

    # Extract anomalies
    pred = Y_pred[:, :, :, var_idx]  # (N, H, W)
    true = Y_true[:, :, :, var_idx]
    clim = clim[np.newaxis, :, :]  # (1, H, W)

    pred_anom = pred - clim
    true_anom = true - clim

    # Weighted means across spatial dims
    def weighted_mean(x):
        return np.sum(x * cosine_lat_weights[np.newaxis, :, np.newaxis], axis=(1, 2)) / np.sum(cosine_lat_weights)

    pred_mean = weighted_mean(pred_anom)
    true_mean = weighted_mean(true_anom)

    # Subtract weighted means
    pred_anom -= pred_mean[:, np.newaxis, np.newaxis]
    true_anom -= true_mean[:, np.newaxis, np.newaxis]

    # Weighted covariance numerator
    numerator = np.sum(pred_anom * true_anom * cosine_lat_weights[np.newaxis, :, np.newaxis], axis=(1, 2))

    # Weighted variances
    pred_var = np.sum(pred_anom ** 2 * cosine_lat_weights[np.newaxis, :, np.newaxis], axis=(1, 2))
    true_var = np.sum(true_anom ** 2 * cosine_lat_weights[np.newaxis, :, np.newaxis], axis=(1, 2))

    # Denominator
    denominator = np.sqrt(pred_var * true_var)

    acc_per_sample = numerator / (denominator + 1e-8)  # prevent division by 0
    return np.mean(acc_per_sample)
