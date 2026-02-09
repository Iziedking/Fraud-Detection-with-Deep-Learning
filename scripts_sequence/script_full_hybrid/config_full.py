CONFIG = {
    'data': {
        'train_temp': 'data_files/X_temp_train_full.npy',
        'train_rel': 'data_files/X_rel_train_full.npy',
        'train_beh': 'data_files/X_beh_train_full.npy',
        'train_y': 'data_files/y_train_full.npy',
        'val_temp': 'data_files/X_temp_val_full.npy',
        'val_rel': 'data_files/X_rel_val_full.npy',
        'val_beh': 'data_files/X_beh_val_full.npy',
        'val_y': 'data_files/y_val_full.npy',
        'test_temp': 'data_files/X_temp_test_full.npy',
        'test_rel': 'data_files/X_rel_test_full.npy',
        'test_beh': 'data_files/X_beh_test_full.npy',
        'test_y': 'data_files/y_test_full.npy'
    },
    'model': {
        'temporal_dim': 30,
        'relational_dim': 52,
        'behavioral_dim': 353,
        'lstm_hidden': 64,
        'lstm_layers': 2,
        'gnn_hidden': 64,
        'gnn_heads': 4,
        'dense_hidden': [256, 128],
        'fusion_hidden': 256,
        'dropout': 0.3
    },
    'training': {
        'batch_size': 512,
        'learning_rate': 0.001,
        'epochs': 100,
        'early_stop_patience': 10
    },
    'paths': {
        'models': 'models',
        'logs': 'logs',
        'results': 'results_sequence/result_hybrid_full'
    }
}