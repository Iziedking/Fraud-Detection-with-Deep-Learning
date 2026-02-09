CONFIG = {
    'data': {
        'train_X': 'data_files/X_train_seq_smote.npy',
        'train_y': 'data_files/y_train_seq_smote.npy',
        'val_X': 'data_files/X_val_seq.npy',
        'val_y': 'data_files/y_val_seq.npy',
        'test_X': 'data_files/X_test_seq.npy',
        'test_y': 'data_files/y_test_seq.npy'
    },
    'model': {
        'input_dim': 249,
        'seq_length': 5,
        'lstm_hidden': 128,
        'lstm_layers': 2,
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
        'results': 'results_sequence/result_lstm'
    }
}