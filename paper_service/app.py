"""Run: python -m uvicorn paper_service.app:app --host 127.0.0.1 --port 8766."""
import hashlib
import json
import os
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from .engine import Risk, execute, make_bars, signals, validate_bars
from .store import Store, Conflict
from .tracking import record

class Bar(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    timestamp:str
    open:float=Field(gt=0)
    close:float=Field(gt=0)
    volume:float=Field(gt=0)

class RunRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    request_id:str=Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    source:Literal['synthetic','uploaded']='synthetic'
    count:int=Field(default=360,ge=160,le=2000)
    seed:int=Field(default=42,ge=0,le=2147483647)
    train_fraction:float=Field(default=.6,ge=.5,le=.8)
    initial_cash:float=Field(default=10000,gt=0,le=1e9)
    max_exposure:float=Field(default=.5,gt=0,le=1)
    max_drawdown:float=Field(default=.1,gt=0,lt=1)
    fee_bps:float=Field(default=10,ge=0,le=100)
    slippage_bps:float=Field(default=5,ge=0,le=100)
    bars:list[Bar]|None=Field(default=None,max_length=2000)

class PredictRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    bars:list[Bar]=Field(min_length=100,max_length=2000)

def frame(rows):
    data=pd.DataFrame([row.model_dump() for row in rows])
    data.index=pd.to_datetime(data.pop('timestamp'),utc=True,errors='raise')
    return validate_bars(data)


def create_app(home=None,tracking_uri=None):
    home=Path(home or os.environ.get('PAPER_SERVICE_HOME','paper-data')).resolve()
    home.mkdir(parents=True,exist_ok=True)
    store=Store(home/'runs.sqlite')
    uri=tracking_uri if tracking_uri is not None else os.environ.get('PAPER_MLFLOW_URI','')
    api=FastAPI(title='Book Paper Trading',version='1.0',description='Local educational replay. No live broker orders.')
    @api.get('/',include_in_schema=False)
    def root():return RedirectResponse('/docs')
    @api.get('/health')
    def health():
        store.list()
        return {'status':'ok','mode':'paper','external_tracking':bool(uri)}
    @api.get('/runs')
    def runs():return store.list()
    @api.get('/runs/{identifier}')
    def get_run(identifier:str):
        found=store.get(identifier)
        if found is None:raise HTTPException(404,'run not found')
        return found
    @api.post('/runs')
    def run(request:RunRequest):
        payload=request.model_dump(mode='json')
        fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        try:
            if request.source=='uploaded':
                if not request.bars:raise ValueError('uploaded source requires bars')
                bars=frame(request.bars)
            else:
                if request.bars is not None:raise ValueError('synthetic source does not accept uploaded bars')
                bars=make_bars(request.count,request.seed)
            cached=store.claim(request.request_id,fingerprint)
            if cached is not None:return cached
        except Conflict as exc:raise HTTPException(409,str(exc)) from exc
        except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
        try:
            risk=Risk(**{key:payload[key] for key in Risk.__dataclass_fields__})
            result,bundle=execute(bars,request.train_fraction,risk)
            result.update({'request_id':request.request_id,'mode':'paper','configuration':payload})
            directory=home/'models'/request.request_id
            directory.mkdir(parents=True,exist_ok=True)
            temporary=directory/'model.tmp';model_path=directory/'model.joblib'
            joblib.dump(bundle,temporary);temporary.replace(model_path)
            result['tracking']=record(uri,request.request_id,result,model_path)
            store.finish(request.request_id,result=result)
            return result
        except ValueError as exc:
            store.finish(request.request_id,error=type(exc).__name__)
            raise HTTPException(422,str(exc)) from exc
        except Exception as exc:
            store.finish(request.request_id,error=type(exc).__name__)
            raise HTTPException(500,'run failed; inspect persisted status') from exc
    @api.post('/runs/{identifier}/predict')
    def predict(identifier:str,request:PredictRequest):
        found=store.get(identifier)
        if found is None:raise HTTPException(404,'run not found')
        if found['status']!='completed':raise HTTPException(409,'model is not available')
        # Identifier is resolved only from the database, never used as an arbitrary path.
        path=home/'models'/found['id']/'model.joblib'
        bundle=joblib.load(path)  # Only load files produced by this service.
        try:
            bars=frame(request.bars)
            feature_frame,targets,forecast,probability=signals(bundle,bars)
            return {'as_of':feature_frame.index[-1].isoformat(),
                    'trained_through':bundle['trained_through'],
                    'target_fraction':float(targets[-1]),'next_return_mean':float(forecast[-1]),
                    'forecast_units':'log_return_per_bar',
                    'state_probabilities':probability[-1].tolist(),
                    'contract':'raw HMM state order; fraction of configured cap, not an order'}
        except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
    return api

app=create_app()
