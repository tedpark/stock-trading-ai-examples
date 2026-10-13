"""Causal HMM signals and a cash/share ledger with next-open fills."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd
from hmm_demo import dummy_bars, features, fit_model, forward

@dataclass(frozen=True)
class Risk:
    initial_cash: float = 10000.
    max_exposure: float = .5
    max_drawdown: float = .1
    fee_bps: float = 10.
    slippage_bps: float = 5.

    def __post_init__(self):
        if not np.isfinite(list(self.__dict__.values())).all():
            raise ValueError('risk settings must be finite')
        if (self.initial_cash <= 0 or not 0 < self.max_exposure <= 1
                or not 0 < self.max_drawdown < 1
                or not 0 <= self.fee_bps <= 100 or not 0 <= self.slippage_bps <= 100):
            raise ValueError('invalid risk settings')


def make_bars(count=360, seed=42):
    """Synthetic opens equal the previous close; no external market service."""
    bars = dummy_bars(count, seed)
    bars['open'] = bars.close.shift(1).fillna(bars.close.iloc[0])
    return bars[['open', 'close', 'volume']]


def validate_bars(bars, min_rows=100):
    if (not isinstance(bars, pd.DataFrame) or bars.columns.has_duplicates
            or not isinstance(bars.index, pd.DatetimeIndex)):
        raise ValueError('DatetimeIndex and unique bar columns required')
    if not {'open','close','volume'} <= set(bars.columns):
        raise ValueError('open, close, volume columns required')
    if (len(bars) < min_rows or len(bars) > 2000 or bars.index.hasnans
            or bars.index.has_duplicates or not bars.index.is_monotonic_increasing):
        raise ValueError(f'{min_rows}..2000 unique chronological bars required')
    values=bars[['open','close','volume']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('positive finite prices and volume required')
    return bars[['open','close','volume']].copy()


def train(bars, fraction=.6):
    """Fit only a chronological prefix; the model never sees held-out prices."""
    bars=validate_bars(bars)
    if not np.isfinite(fraction) or not .5 <= fraction <= .8:
        raise ValueError("train fraction must be finite and between .5 and .8")
    frame=features(bars)
    split=int(len(frame)*fraction)
    if split < 60 or len(frame)-split < 20:
        raise ValueError('at least 60 training and 20 held-out feature rows required')
    model,scaler,order=fit_model(frame.to_numpy(),split)
    # Forecast the next state's unstandardized log-return mean, not the current mixture mean.
    bundle={'model':model,'scaler':scaler,'order':order,'columns':list(frame.columns),
            'trained_through':frame.index[split-1].isoformat(),'schema_version':1}
    return bundle,frame,split


def signals(bundle, bars):
    """Return long-only target fraction of the configured exposure cap.

    A positive one-step HMM return forecast proposes full cap; otherwise cash.
    This is an illustrative decision rule, not a calibrated probability of profit.
    """
    frame=features(bars)
    if list(frame.columns)!=bundle['columns'] or frame.empty:
        raise ValueError('feature schema mismatch or insufficient history')
    model,scaler=bundle['model'],bundle['scaler']
    alpha=forward(model,scaler.transform(frame.to_numpy()))
    means=model.means_[:,0]*scaler.scale_[0]+scaler.mean_[0]
    forecast=(alpha @ model.transmat_) @ means
    return frame,np.where(forecast>0,1.,0.),forecast,alpha


def simulate(bars, targets, risk):
    """targets[t] is known at close[t]; fill at open[t+1]. Long-only cash account.

    Drawdown detected at a close latches a halt and exits at the next open.
    Gaps may exceed the loss threshold. Last close liquidates all remaining shares.
    Fees/slippage are charged on each fill; fractional shares simplify arithmetic.
    """
    bars=validate_bars(bars, min_rows=2)
    targets=np.asarray(targets, dtype=float)
    if targets.ndim != 1 or len(bars)!=len(targets):
        raise ValueError('aligned bars and signals required')
    if not np.isfinite(targets).all() or np.any((np.asarray(targets)<0)|(np.asarray(targets)>1)):
        raise ValueError('target fractions must be in [0,1]')
    cash,quantity,peak=risk.initial_cash,0.,risk.initial_cash
    halted=False
    fills=[];timeline=[];equity=[risk.initial_cash]
    fee,slip=risk.fee_bps/10000,risk.slippage_bps/10000
    def fill(delta,price,timestamp,reason):
        nonlocal cash,quantity
        if abs(delta)<1e-10:return
        execution=price*(1+np.sign(delta)*slip)
        cost=abs(delta)*execution*fee
        cash-=delta*execution+cost;quantity+=delta
        if abs(quantity)<1e-10:quantity=0.
        fills.append({'timestamp':timestamp,'quantity':float(delta),'price':float(execution),
                      'fee':float(cost),'reason':reason})
    for i,(timestamp,row) in enumerate(bars.iterrows()):
        stamp=timestamp.isoformat()
        before=cash+quantity*row.open
        weight=0. if halted or i==0 else float(targets[i-1])*risk.max_exposure
        desired=weight*before/row.open
        # Reserve purchase fees and slippage: cash must never become negative.
        desired=min(desired,quantity+max(cash,0)/(row.open*(1+slip)*(1+fee)))
        fill(desired-quantity,row.open,stamp,'risk_halt' if halted else 'previous_close_signal')
        if i==len(bars)-1:fill(-quantity,row.close,stamp,'final_liquidation')
        value=float(cash+quantity*row.close)
        peak=max(peak,value);drawdown=1-value/peak
        halted=halted or drawdown>=risk.max_drawdown
        if cash < -1e-7 or quantity < -1e-7 or not np.isfinite(value) or value<=0:
            raise ValueError('invalid paper account state')
        equity.append(value)
        timeline.append({'timestamp':stamp,'cash':float(cash),'quantity':float(quantity),
                         'close':float(row.close),'equity':value,'drawdown':float(drawdown),
                         'halted':bool(halted),'signal':float(targets[i]),'target_weight':weight})
    metrics={'initial_equity':risk.initial_cash,'final_equity':equity[-1],
             'net_return':equity[-1]/risk.initial_cash-1,
             'max_drawdown':max(r['drawdown'] for r in timeline),
             'fees':sum(f['fee'] for f in fills),'fills':len(fills),'halted':bool(halted)}
    return {'metrics':metrics,'timeline':timeline,'fills':fills}


def execute(bars, fraction, risk):
    bundle,frame,split=train(bars,fraction)
    _,targets,forecast,alpha=signals(bundle,bars)
    # First OOS row is observed before any OOS order is placed.
    out=simulate(bars.loc[frame.index[split:]],targets[split:],risk)
    out['training']={'rows':split,'held_out_rows':len(frame)-split,
                     'trained_through':bundle['trained_through'],
                     'feature_columns':bundle['columns'],
                     'converged':bool(bundle['model'].monitor_.converged)}
    out['data_sha256']=hashlib.sha256(bars.to_csv().encode()).hexdigest()
    out['forecast_units']='log_return_per_bar'
    out['decision_rule']='long if next-state HMM log-return mean is positive; otherwise cash'
    return out,bundle
