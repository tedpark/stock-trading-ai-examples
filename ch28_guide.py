"""Chapter 28: view the learning path and chapter files without dependencies."""
import json
from pathlib import Path

print('First run: python run.py 00')
print('Foundations: 01 → 02 → 03 → 09 → 16 → 17 → 18')
print('Regimes and pairs: 04~12 → 19~22')
print('Reinforcement learning and validation: 13~15 → 23~27')
for row in json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8')):
    print(row['chapter'], row['file'])
