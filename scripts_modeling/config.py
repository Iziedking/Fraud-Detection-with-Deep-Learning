CONFIG = {
    'data': {
        'train_X': 'data_files/X_train_resampled.parquet',
        'train_y': 'data_files/y_train_resampled.parquet',
        'val_X': 'data_files/X_val_scaled.parquet',
        'val_y': 'data_files/y_val.parquet',
        'test_X': 'data_files/X_test_scaled.parquet',
        'test_y': 'data_files/y_test.parquet',
        'feature_groups': 'data_files/feature_groups.json'
    },
    'model': {
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dense_hidden': [256, 128, 64],
        'dropout': 0.3,
        'fusion_dim': 256
    },
    'training': {
        'batch_size': 512,
        'learning_rate': 0.001,
        'epochs': 100,
        'early_stop_patience': 10
    },
    'paths': {
        'models': 'models',
        'logs': 'logs'
    }
}