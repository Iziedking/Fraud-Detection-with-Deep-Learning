CONFIG = {
    'data': {
        'train_temp': 'data_files/X_temp_train_fusion.npy',
        'train_rel': 'data_files/X_rel_train_fusion.npy',
        'train_beh': 'data_files/X_beh_train_fusion.npy',
        'train_y': 'data_files/y_train_fusion.npy',
        'val_temp': 'data_files/X_temp_val_fusion.npy',
        'val_rel': 'data_files/X_rel_val_fusion.npy',
        'val_beh': 'data_files/X_beh_val_fusion.npy',
        'val_y': 'data_files/y_val_fusion.npy',
        'test_temp': 'data_files/X_temp_test_fusion.npy',
        'test_rel': 'data_files/X_rel_test_fusion.npy',
        'test_beh': 'data_files/X_beh_test_fusion.npy',
        'test_y': 'data_files/y_test_fusion.npy'
    },
    'model': {
        'temporal_dim': 30,
        'relational_dim': 38,
        'behavioral_dim': 184,
        'lstm_hidden': 64,
        'lstm_layers': 2,
        'gnn_hidden': 64,
        'gnn_heads': 4,
        'dense_hidden': [128, 64],
        'fusion_hidden': 128,
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
        'results': 'results_sequence/result_hybrid'
    }
}