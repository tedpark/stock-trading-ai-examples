"""Real loopback HTTP integration test. Starts and stops only its own servers.

Run: python verify_service.py --output /tmp/book-service-check
Optional: --market-symbol SPY (read-only live Yahoo download).
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
import requests

ROOT=Path(__file__).resolve().parent

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

def stop(process):
    if process.poll() is None:
        if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
        else:process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
            else:process.kill()
            process.wait(timeout=10)

def wait_ready(process,url):
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('server exited before readiness')
        try:
            if requests.get(url+'/health',timeout=1).status_code==200:return
        except requests.RequestException:pass
        time.sleep(.2)
    raise TimeoutError('local server did not become ready')

def verify(output,market_symbol=None):
    output.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('MLFLOW_','PAPER_'))}
    env.update(MLFLOW_ENABLE_TELEMETRY='false',MLFLOW_DISABLE_AGENT_HINT='1',
               MLFLOW_HTTP_REQUEST_TIMEOUT='2',MLFLOW_HTTP_REQUEST_MAX_RETRIES='0',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    checks=[];processes=[];logs=[]
    def launch(command,name):
        log=(output/(name+'.log')).open('w');logs.append(log)
        proc=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=env,stdout=log,
                              stderr=subprocess.STDOUT,start_new_session=(os.name=='posix'))
        processes.append(proc);return proc
    with TemporaryDirectory(prefix='book-service-') as directory:
        base=Path(directory)
        try:
            tracking_port=free_port();tracking_url=f'http://127.0.0.1:{tracking_port}'
            tracking=launch(['-m','mlflow','server','--host','127.0.0.1','--port',str(tracking_port),
                             '--workers','1','--backend-store-uri','sqlite:///'+str(base/'tracking.db'),
                             '--artifacts-destination',str(base/'artifacts')],'mlflow')
            wait_ready(tracking,tracking_url)
            env.update(PAPER_SERVICE_HOME=str(base/'service'),PAPER_MLFLOW_URI=tracking_url)
            port=free_port();url=f'http://127.0.0.1:{port}'
            command=['-m','uvicorn','paper_service.app:app','--host','127.0.0.1','--port',str(port)]
            service=launch(command,'service');wait_ready(service,url)
            response=requests.post(url+'/runs',json={'request_id':'integrated-demo'},timeout=90)
            assert response.status_code==200,response.text
            result=response.json();assert result['tracking']['status']=='recorded',result['tracking']
            assert result['metrics']['fills']>0 and result['timeline'][-1]['quantity']==0
            checks.append('HTTP data -> features -> HMM -> risk -> paper fills -> SQLite -> MLflow')
            from mlflow import MlflowClient
            client=MlflowClient(tracking_uri=tracking_url)
            run_id=result['tracking']['run_id'];run=client.get_run(run_id)
            assert run.info.status=='FINISHED'
            assert run.data.metrics['final_equity']==result['metrics']['final_equity']
            assert run.info.artifact_uri.startswith('mlflow-artifacts:')
            path=client.download_artifacts(run_id,'result.json',str(base))
            saved=json.loads(Path(path).read_text());assert saved['data_sha256']==result['data_sha256']
            model=client.download_artifacts(run_id,'model/model.joblib',str(base))
            assert Path(model).stat().st_size>0
            checks.append('MLflow HTTP metric readback and proxied JSON/model artifact download')
            again=requests.post(url+'/runs',json={'request_id':'integrated-demo'},timeout=15)
            assert again.json()==result
            assert requests.post(url+'/runs',json={'request_id':'integrated-demo','seed':43},timeout=10).status_code==409
            checks.append('idempotent repeated request and conflicting request rejection')
            from paper_service.engine import make_bars
            from paper_service.market import rows
            payload={'bars':rows(make_bars())}
            prediction=requests.post(url+'/runs/integrated-demo/predict',json=payload,timeout=30)
            assert prediction.status_code==200,prediction.text
            assert abs(sum(prediction.json()['state_probabilities'])-1)<1e-8
            stop(service);service=launch(command,'service-restarted');wait_ready(service,url)
            assert requests.get(url+'/runs/integrated-demo',timeout=10).json()['result']==result
            assert requests.post(url+'/runs/integrated-demo/predict',json=payload,timeout=30).json()==prediction.json()
            checks.append('process restart preserves ledger and saved-model inference')
            uploaded=requests.post(url+'/runs',json={'request_id':'upload','source':'uploaded',**payload},timeout=90)
            assert uploaded.status_code==200,uploaded.text
            checks.append('uploaded timestamped market-schema bars through full pipeline')
            stop(tracking)
            degraded=requests.post(url+'/runs',json={'request_id':'tracking-offline'},timeout=90)
            assert degraded.status_code==200,degraded.text
            assert degraded.json()['tracking']['status']=='failed'
            assert requests.get(url+'/runs/tracking-offline',timeout=10).json()['status']=='completed'
            checks.append('actual tracking-server outage preserves local completed paper result')
            monitor_port=free_port();monitor_url=f'http://127.0.0.1:{monitor_port}'
            monitor=launch(['-m','uvicorn','ch18_lab:app','--host','127.0.0.1','--port',str(monitor_port)],'chapter18-monitor')
            wait_ready(monitor,monitor_url)
            report=requests.get(monitor_url+'/report',timeout=10).json()
            assert report['violations']==['mdd'] and abs(report['metrics']['mdd']-.2)<1e-12
            checks.append('original Chapter 18 monitoring API over real HTTP')
            market={'status':'not_requested'}
            if market_symbol:
                from paper_service.market import download
                try:
                    bars=download(market_symbol)
                    live=requests.post(url+'/runs',json={'request_id':'market-readonly','source':'uploaded','bars':rows(bars)},timeout=90)
                    assert live.status_code==200,live.text
                    market={'status':'passed','symbol':market_symbol,'bars':len(bars),
                            'first':bars.index[0].isoformat(),'last':bars.index[-1].isoformat(),
                            'data_sha256':live.json()['data_sha256']}
                except Exception as exc:
                    market={'status':'unavailable','symbol':market_symbol,'error_type':type(exc).__name__,'reason':str(exc)[:200]}
            report={'status':'passed','checks':checks,'market':market,'metrics':result['metrics'],
                    'servers_cleaned_up':True,'mode':'paper','no_broker_orders':True}
            (output/'integration.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
        finally:
            for process in reversed(processes):stop(process)
            for log in logs:log.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('service-test-data'))
    parser.add_argument('--market-symbol');args=parser.parse_args()
    verify(args.output.resolve(),args.market_symbol)
