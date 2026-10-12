"""python run.py 01 | python run.py all | python run.py --list"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='더미 데이터 기반 책 실습 실행')
    parser.add_argument('chapter', nargs='?', default='00', help='00~28, core, all')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    rows = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    if args.list:
        for row in rows:
            print(row['chapter'], row['file'], row['kind'])
        return 0
    chapter = args.chapter.zfill(2)
    selected = [r for r in rows if args.chapter == 'all'
                or (args.chapter == 'core' and r['chapter'] not in ('14', '23', '27'))
                or r['chapter'] == chapter]
    if not selected:
        parser.error('장 번호는 00~28 또는 core/all입니다.')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    failures = []
    for row in selected:
        print('\n>>> ' + row['file'], flush=True)
        try:
            result = subprocess.run([sys.executable, str(ROOT / row['file'])],
                                    cwd=ROOT, env=env, timeout=240)
            if result.returncode:
                failures.append(row['file'])
        except subprocess.TimeoutExpired:
            failures.append(row['file'])
            print('240초 실행 제한 초과')
    print(f'\nPASS {len(selected)-len(failures)}/{len(selected)}', flush=True)
    if failures:
        print('FAIL:', ', '.join(failures))
        print('의존성 오류라면: python -m pip install -r requirements-full.txt')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
