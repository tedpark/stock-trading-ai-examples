"""Optional MLflow recording; the local ledger remains authoritative."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

def record(uri,identifier,result,model_path):
    if not uri:return {'status':'disabled','run_id':None}
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT','2')
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES','0')
    run_id=None
    try:
        from mlflow import MlflowClient
        client=MlflowClient(tracking_uri=uri)
        name='book-paper-service'
        experiment=client.get_experiment_by_name(name)
        if experiment is None:
            try:experiment_id=client.create_experiment(name)
            except Exception:
                experiment=client.get_experiment_by_name(name)
                if experiment is None:raise
                experiment_id=experiment.experiment_id
        else:experiment_id=experiment.experiment_id
        run_id=client.create_run(experiment_id,tags={'request_id':identifier,'mode':'paper'}).info.run_id
        client.log_param(run_id,'data_sha256',result['data_sha256'])
        client.log_param(run_id,'trained_through',result['training']['trained_through'])
        for key,value in result['metrics'].items():client.log_metric(run_id,key,float(value))
        with TemporaryDirectory() as directory:
            path=Path(directory)/'result.json'
            path.write_text(json.dumps(result,allow_nan=False))
            client.log_artifact(run_id,str(path))
        client.log_artifact(run_id,str(model_path),'model')
        client.set_terminated(run_id)
        return {'status':'recorded','run_id':run_id}
    except Exception as exc:
        if run_id:
            try:client.set_terminated(run_id,status='FAILED')
            except Exception:pass
        # Avoid returning credential-bearing URLs or raw server messages.
        return {'status':'failed','run_id':run_id,'error_type':type(exc).__name__}
