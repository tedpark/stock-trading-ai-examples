"""Repository -> manuscript sync. The manuscript is never used to update source.

python sync_book.py --book-root /path/to/stock-trading-ai --check
python sync_book.py --book-root /path/to/stock-trading-ai --write --refresh-index
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
EDITIONS={'ko':'manuscript','en':'manuscript-en','ja':'manuscript-ja'}
FENCE=re.compile(r'^```([^\n`]*)\n(.*?)^```[ \t]*$',re.M|re.S)
PYTHON=re.compile(r'^```python\n(.*?)^```[ \t]*$',re.M|re.S)

def contract(root=ROOT):
    entries=json.loads((root/'book_contract.json').read_text())
    assert len(entries)==107 and len({e['id'] for e in entries})==107
    assert len({(e['chapter'],e['python_block']) for e in entries})==107
    return entries

def source_text(entry,root=ROOT):
    path=(root/entry['source']).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('source outside repository')
    code=path.read_text()
    if entry['strip_module_docstring']:
        tree=ast.parse(code)
        if (tree.body and isinstance(tree.body[0],ast.Expr)
                and isinstance(tree.body[0].value,ast.Constant)
                and isinstance(tree.body[0].value.value,str)):
            code=''.join(code.splitlines(keepends=True)[tree.body[0].end_lineno:])
    return code.strip()+'\n'

def synchronize(book_root,write=False,root=ROOT):
    entries=contract(root);by_key={(e['chapter'],e['python_block']):e for e in entries}
    planned=[];differences=[];seen=set()
    for edition,folder in EDITIONS.items():
        directory=book_root/folder
        if not directory.is_dir():raise ValueError(f'missing edition: {directory}')
        for path in sorted(directory.glob('*.md')):
            text=path.read_text();changes=[]
            for number,match in enumerate(PYTHON.finditer(text),1):
                key=(path.name,number)
                if key not in by_key:raise ValueError(f'unmapped Python block: {edition}/{key}')
                entry=by_key[key];seen.add((edition,*key))
                # Explicit anchors prevent insertion/reordering from silently changing mappings.
                prefix=text[:match.start()]
                prior=re.search(r'<!-- repository-source: ([^\n]+) -->\n(?:<!-- book-example: [^\n]+ -->\n)?$',prefix)
                marker=f"<!-- repository-source: {entry['id']} -->\n"
                if prior and prior[1]!=entry['id']:raise ValueError(f'wrong source anchor: {path}:{number}')
                expected=source_text(entry,root)
                if match[1]!=expected:changes.append((match.start(1),match.end(1),expected))
                if not prior:
                    legacy=re.search(r'<!-- book-example: [^\n]+ -->\n$',prefix)
                    position=legacy.start() if legacy else match.start()
                    changes.append((position,position,marker))
                if not prior or match[1]!=expected:differences.append(f'{edition}/{entry["id"]}')
            for start,end,value in sorted(changes,reverse=True):text=text[:start]+value+text[end:]
            if changes:planned.append((path,text))
    expected={(edition,e['chapter'],e['python_block']) for edition in EDITIONS for e in entries}
    if seen!=expected:raise ValueError(f'missing source slots: {sorted(expected-seen)}')
    # Validate every edition before writing any file.
    if write:
        for path,text in planned:path.write_text(text)
    return differences

def refresh_index(book_root,root=ROOT):
    if synchronize(book_root,root=root):raise ValueError('sync manuscript from repository before refreshing index')
    entries={(e['chapter'],e['python_block']):e for e in contract(root)}
    records=[];contents={};old_path=root/'book_sources/index.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else []
    for edition,folder in EDITIONS.items():
        for path in sorted((book_root/folder).glob('*.md')):
            text=path.read_text();python_index=0
            for number,match in enumerate(FENCE.finditer(text),1):
                language,code=match.groups();language=language.strip() or 'text'
                entry=None;error=None
                if language=='python':
                    python_index+=1;entry=entries[(path.name,python_index)]
                    assert code==source_text(entry,root)
                    ast.parse(code)
                    try:compile(code,str(path),'exec')
                    except SyntaxError as exc:error=exc.msg
                sha=hashlib.sha256(code.encode()).hexdigest()
                same=next((r for r in records if r['chapter']==path.name and r['block']==number and r['sha256']==sha),None)
                artifact=same['artifact'] if same else f'book_sources/{edition}/{path.stem}/block-{number:02}.{language}.txt'
                contents[artifact]=code
                records.append(dict(edition=edition,chapter=path.name,block=number,
                    python_block=python_index if entry else None,line=text[:match.start()].count('\n')+1,
                    language=language,kind=entry['kind'] if entry else 'reference_fragment',
                    artifact=artifact,runnable_file=entry['runnable_file'] if entry else None,
                    repository_source=entry['source'] if entry else None,
                    source_id=entry['id'] if entry else None,sha256=sha,compile_error=error,
                    execution='chapter_suite' if entry and entry['runnable_file'] else 'not_a_standalone_program'))
    for relative,code in contents.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(code)
    for relative in {r['artifact'] for r in old}-set(contents):
        path=(root/relative).resolve()
        if not path.is_relative_to((root/'book_sources').resolve()):raise ValueError('bad old artifact path')
        path.unlink(missing_ok=True)
    old_path.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    return records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--book-root',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--write',action='store_true')
    parser.add_argument('--refresh-index',action='store_true')
    args=parser.parse_args();diff=synchronize(args.book_root.resolve(),write=args.write)
    if diff and not args.write:
        print('Repository/manuscript drift:\n'+'\n'.join(diff));return 1
    if args.refresh_index:refresh_index(args.book_root.resolve())
    print(f'{len(diff)} source slots updated' if args.write else '321 Python slots match repository sources')
    return 0
if __name__=='__main__':raise SystemExit(main())
