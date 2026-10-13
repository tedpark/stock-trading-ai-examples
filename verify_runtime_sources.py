"""Measure which implementation callables were exercised by the regression suite."""
import ast
import json
from pathlib import Path
import sys
import unittest
from sync_book import ROOT, contract, source_text

def main():
    entries=[e for e in contract() if e.get('symbol')]
    paths={str((ROOT/e['source']).resolve()) for e in entries}
    called=set()
    def profile(frame,event,arg):
        if event=='call' and frame.f_code.co_filename in paths:
            called.add((frame.f_code.co_filename,frame.f_code.co_qualname))
    sys.setprofile(profile)
    try:
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        result=unittest.TextTestRunner(verbosity=1).run(suite)
    finally:
        sys.setprofile(None)
    report=[]
    for e in entries:
        code=source_text(e);tree=ast.parse(code)
        symbols=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
        if e['symbol'].startswith('@'):
            exercised=all((str((ROOT/e['source']).resolve()),name) in called for name in symbols)
        else:
            symbol=e['symbol'];path=str((ROOT/e['source']).resolve())
            exercised=any(p==path and (name==symbol or name.startswith(symbol+'.')) for p,name in called)
        report.append(dict(source_id=e['id'],source=e['source'],symbol=e['symbol'],exercised=exercised))
    destination=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'RUNTIME_VALIDATION.json'
    destination.write_text(json.dumps(dict(tests_run=result.testsRun,successful=result.wasSuccessful(),excerpts=report),indent=2)+'\n')
    missing=[r['source_id'] for r in report if not r['exercised']]
    print('Unexercised implementation excerpts:',missing)
    return 0 if result.wasSuccessful() and not missing else 1
if __name__=='__main__':raise SystemExit(main())
