"""Run all examples and tests with network connections blocked: python verify.py"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'tests' / 'offline'),
               PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    # Do not inherit MLflow settings, automatic tracking, or experiment IDs from other tasks.
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
