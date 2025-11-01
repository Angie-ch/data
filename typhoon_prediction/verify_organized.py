"""Verify organized data structure"""
import numpy as np
import json
from pathlib import Path

folder = Path(r'D:\typhoon_aligned\organized_by_typhoon\2018039N08151')
feat = np.load(folder / 'features.npy')
with open(folder / 'metadata.json', 'r') as f:
    meta = json.load(f)

print(f'Features shape: {feat.shape}')
print(f'Metadata length: {len(meta)}')
print(f'First timestep: {meta[0]["time"]}')
print(f'Last timestep: {meta[-1]["time"]}')
print(f'Typhoon: {meta[0]["typhoon_name"]}')
print(f'\nSample metadata entries:')
for i in [0, len(meta)//2, len(meta)-1]:
    print(f'  Timestep {i}: time={meta[i]["time"]}, lat={meta[i]["lat"]}, lon={meta[i]["lon"]}, wind={meta[i]["wind"]}')

