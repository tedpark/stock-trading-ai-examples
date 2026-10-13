"""Optional read-only Yahoo downloader; never places orders or requires credentials."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .engine import validate_bars

def download(symbol,period='2y'):
    import yfinance as yf
    if not symbol or len(symbol)>20:raise ValueError('one ticker required')
    bars=yf.download(symbol,period=period,interval='1d',auto_adjust=True,
                     multi_level_index=False,progress=False,threads=False,timeout=15)
    if bars.empty:raise ValueError('market provider returned no data')
    if isinstance(bars.columns,pd.MultiIndex):
        if bars.columns.nlevels!=2 or list(bars.columns.get_level_values(1).unique())!=[symbol]:
            raise ValueError('expected one ticker')
        bars.columns=bars.columns.get_level_values(0)
    bars.columns=[str(x).lower() for x in bars.columns]
    if bars.columns.duplicated().any():raise ValueError('duplicate price columns')
    bars.index=pd.to_datetime(bars.index,utc=True)
    return validate_bars(bars)

def rows(bars):
    return [{'timestamp':t.isoformat(),'open':float(r.open),'close':float(r.close),'volume':float(r.volume)}
            for t,r in bars.iterrows()]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('symbol');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload={'request_id':'market-demo','source':'uploaded','bars':rows(download(args.symbol))}
    args.output.write_text(json.dumps(payload,allow_nan=False))
    print(f'Wrote {len(payload["bars"])} adjusted daily bars to {args.output}')
if __name__=='__main__':main()
