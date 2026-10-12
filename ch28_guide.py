"""28장: 설치 없이 학습 경로와 장별 파일 목록 확인."""
import json
from pathlib import Path

print('첫 실행: python run.py 00')
print('기초: 01 → 02 → 03 → 09 → 16 → 17 → 18')
print('레짐/페어: 04~12 → 19~22')
print('강화학습/검증: 13~15 → 23~27')
for row in json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8')):
    print(row['chapter'], row['file'])
