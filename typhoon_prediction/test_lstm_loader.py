"""Test LSTM data loader"""
from data_loaders.lstm_loader import create_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=r'D:\typhoon_aligned\organized_by_typhoon',
    sequence_length=10,
    prediction_horizon=1,
    batch_size=16,
    target_type='wind'
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')
print(f'Test batches: {len(test_loader)}')

# Test loading
seq, tgt, meta = next(iter(train_loader))
print(f'\nSequence shape: {seq.shape}')
print(f'Target shape: {tgt.shape}')
print(f'Sample typhoon: {meta["typhoon_id"][0]}')
print(f'Start time: {meta["start_time"][0]}')
print(f'Target time: {meta["target_time"][0]}')

print('\nLSTM loader test successful!')

