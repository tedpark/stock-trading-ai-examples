"""네트워크 연결을 차단하고 모든 예제와 테스트를 실행: python verify.py"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'tests' / 'offline'),
               PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    # 다른 작업의 MLflow 설정/자동 추적·실험 ID를 상속하지 않는다.
    env = {k: v for k, v in env.items() if not k.startswith('MLFLOW_')}
    env['MLFLOW_ENABLE_TELEMETRY'] = 'false'
    commands = [['run.py', 'all'], ['-m', 'unittest', 'discover', '-s', 'tests', '-v']]
    for command in commands:
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    print('All examples and tests passed with outbound connections blocked.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
