CONFIG = {
    'data': {
        'train_X': 'data_files/X_train_gnn.npy',
        'train_y': 'data_files/y_train_gnn.npy',
        'val_X': 'data_files/X_val_gnn.npy',
        'val_y': 'data_files/y_val_gnn.npy',
        'test_X': 'data_files/X_test_gnn.npy',
        'test_y': 'data_files/y_test_gnn.npy'
    },
    'model': {
        'input_dim': 38,
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
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
        'results': 'results_sequence/result_gnn'
    }
}