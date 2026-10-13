"""Verify exact archived book code and runnable-source mappings; never execute fragments."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def tree(source):
    parsed=ast.parse(source)
    if (parsed.body and isinstance(parsed.body[0],ast.Expr)
            and isinstance(parsed.body[0].value,ast.Constant)
            and isinstance(parsed.body[0].value.value,str)):
        parsed.body.pop(0)
    return ast.dump(parsed)

def verify():
    rows=json.loads((ROOT/'book_sources/index.json').read_text())
    seen=set();counts={}
    for row in rows:
        key=(row['edition'],row['chapter'],row['block'])
        assert key not in seen,key
        seen.add(key)
        path=(ROOT/row['artifact']).resolve()
        assert path.is_relative_to(ROOT/'book_sources'),path
        code=path.read_text()
        assert hashlib.sha256(code.encode()).hexdigest()==row['sha256'],key
        if row['language']=='python':
            ast.parse(code)
            error=None
            try:compile(code,str(path),'exec')
            except SyntaxError as exc:error=exc.msg
            assert error==row['compile_error'],key
            if row['runnable_file']:
                assert error is None,key
                assert tree(code)==tree((ROOT/row['runnable_file']).read_text()),key
            counts.setdefault(row['edition'],dict(python=0,runnable=0,context=0))
            counts[row['edition']]['python']+=1
            counts[row['edition']]['runnable']+=bool(row['runnable_file'])
            counts[row['edition']]['context']+=bool(error)
    for edition in ['ko','en','ja']:
        assert counts[edition]==dict(python=107,runnable=34,context=5),counts
    return dict(fenced_blocks=len(rows),editions=counts,
                scope='Integrity, syntax/context classification and runnable AST equivalence; not full execution of fragments')

if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
