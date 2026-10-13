"""Repository -> manuscript sync. The manuscript is never used to update source.

python sync_book.py --book-root /path/to/stock-trading-ai --check
python sync_book.py --book-root /path/to/stock-trading-ai --write --refresh-index
"""
import argparse
from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap

ROOT=Path(__file__).resolve().parent
EDITIONS={'ko':'manuscript','en':'manuscript-en','ja':'manuscript-ja'}
@dataclass(frozen=True)
class Fence:
    language: str
    code: str
    start: int
    body_start: int
    body_end: int


def fences(text):
    """Scan complete fences so nested examples cannot become executable listings.

    Python listings use one canonical spelling. Alternate Markdown spellings
    are rejected instead of silently escaping the repository mapping check.
    """
    active = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if active is None:
            match = re.match(r'^ {0,3}(`{3,}|~{3,})([^\n]*)\n?$', line)
            if match:
                delimiter, info = match.groups()
                language = info.strip() or 'text'
                is_python = (re.match(r'^(python[23]?|py)(\s|$)', language, re.I)
                             or re.search(r'(^|\s|\{)\.(python[23]?|py)(\s|\}|$)', language, re.I))
                if is_python and line != '```python\n':
                    raise ValueError('Use an anchored canonical ```python listing for repository code.')
                active = (delimiter, language, offset, offset + len(line))
        else:
            delimiter, language, start, body_start = active
            closing = r'^ {0,3}' + re.escape(delimiter[0]) + '{' + str(len(delimiter)) + r',}[ \t]*$'
            if re.match(closing, line.rstrip('\n')):
                yield Fence(language, text[body_start:offset], start, body_start, offset)
                active = None
        offset += len(line)
    if active is not None:
        raise ValueError('Unclosed Markdown fence; source inventory would be incomplete.')

def contract(root=ROOT):
    entries=json.loads((root/'book_contract.json').read_text())
    assert len(entries)==107 and len({e['id'] for e in entries})==107
    assert len({(e['chapter'],e['python_block']) for e in entries})==107
    return entries

def source_text(entry,root=ROOT):
    path=(root/entry['source']).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('source outside repository')
    if path.suffix != '.py' or 'book_snippets' in path.parts or 'book_sources' in path.parts:
        raise ValueError('Book Python must originate in executable .py source')
    code=path.read_text()
    tree=ast.parse(code)
    compile(tree,str(path),'exec')
    if any(isinstance(n,ast.Constant) and n.value is Ellipsis for n in ast.walk(tree)):
        raise ValueError('Ellipsis placeholder in implementation source')
    symbol=entry.get('symbol')
    if symbol:
        if symbol.startswith('@'):
            start=f'# source-region: {symbol[1:]}'
            end=f'# end-source-region: {symbol[1:]}'
            if code.count(start)!=1 or code.count(end)!=1:
                raise ValueError('missing or ambiguous source region')
            code=code.split(start,1)[1].split(end,1)[0]
        else:
            body=tree.body
            for component in symbol.split('.'):
                matches=[n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==component]
                if len(matches)!=1:raise ValueError(f'missing/ambiguous symbol: {symbol}')
                node=matches[0];body=node.body
            start=min([node.lineno]+[n.lineno for n in getattr(node,'decorator_list',[])])-1
            code=''.join(code.splitlines(keepends=True)[start:node.end_lineno])
        return textwrap.dedent(code).strip()+'\n'
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
            for number,match in enumerate((f for f in fences(text) if f.language == 'python'),1):
                key=(path.name,number)
                if key not in by_key:raise ValueError(f'unmapped Python block: {edition}/{key}')
                entry=by_key[key];seen.add((edition,*key))
                # Explicit anchors prevent insertion/reordering from silently changing mappings.
                prefix=text[:match.start]
                prior=re.search(r'<!-- repository-source: ([^\n]+) -->\n(?:<!-- book-example: [^\n]+ -->\n)?$',prefix)
                marker=f"<!-- repository-source: {entry['id']} -->\n"
                label=(f"<!-- source-link:start -->[Source: `{entry['source']}`"
                       + (f" · `{entry['symbol']}`" if entry.get('symbol') else '')
                       + f"](https://github.com/tedpark/stock-trading-ai-examples/blob/runtime-book-2026-10-13/{entry['source']})<!-- source-link:end -->\n\n")
                link=re.search(r'<!-- source-link:start -->.*?<!-- source-link:end -->\n\n$', prefix[:prior.start()] if prior else prefix)
                if prior:
                    if link:
                        if link[0]!=label:changes.append((link.start(),link.end(),label))
                    else:changes.append((prior.start(),prior.start(),label))

                if prior and prior[1]!=entry['id']:raise ValueError(f'wrong source anchor: {path}:{number}')
                expected=source_text(entry,root)
                if match.code!=expected:changes.append((match.body_start,match.body_end,expected))
                if not prior:
                    legacy=re.search(r'<!-- book-example: [^\n]+ -->\n$',prefix)
                    position=legacy.start() if legacy else match.start
                    changes.append((position,position,label+marker))
                if not prior or match.code!=expected:differences.append(f'{edition}/{entry["id"]}')
            for start,end,value in sorted(changes,reverse=True):text=text[:start]+value+text[end:]
            if changes:planned.append((path,text))
    expected={(edition,e['chapter'],e['python_block']) for edition in EDITIONS for e in entries}
    if seen!=expected:raise ValueError(f'missing source slots: {sorted(expected-seen)}')
    # Validate every edition before writing any file.
    if write:
        for path,text in planned:path.write_text(text)
    return differences

def synchronize_commands(book_root, write=False, root=ROOT):
    entries=json.loads((root/'command_contract.json').read_text())
    differences=[];plans=[]
    for edition,folder in EDITIONS.items():
        for path in sorted((book_root/folder).glob('*.md')):
            text=path.read_text()
            blocks=[f for f in fences(text) if f.language not in ('python','text')]
            expected=[e for e in entries if e['edition']==edition and e['chapter']==path.name]
            if len(blocks)!=len(expected):raise ValueError(f'unmapped command/configuration: {path}')
            changes=[]
            for block,entry in zip(blocks,expected):
                source=(root/entry['source']).resolve()
                if not source.is_relative_to(root.resolve()) or source.suffix not in ('.sh','.json'):
                    raise ValueError('invalid command source')
                if block.language!=entry['language']:raise ValueError('command language drift')
                code=source.read_text()
                label=f"<!-- command-source:start -->[Source: `{entry['source']}`](https://github.com/tedpark/stock-trading-ai-examples/blob/runtime-book-2026-10-13/{entry['source']})<!-- command-source:end -->\n\n"
                prefix=text[:block.start]
                prior=re.search(r'<!-- command-source:start -->.*?<!-- command-source:end -->\n\n$',prefix)
                if not prior:changes.append((block.start,block.start,label))
                elif prior[0]!=label:changes.append((prior.start(),prior.end(),label))
                if block.code!=code:
                    changes.append((block.body_start,block.body_end,code))
                    differences.append(f"{edition}/{path.name}/{entry['source']}")
            for a,b,value in sorted(changes,reverse=True):text=text[:a]+value+text[b:]
            if changes:plans.append((path,text))
    if write:
        for path,text in plans:path.write_text(text)
    return differences

def refresh_index(book_root,root=ROOT):
    if synchronize(book_root,root=root):raise ValueError('sync manuscript from repository before refreshing index')
    if synchronize_commands(book_root,root=root):raise ValueError('command source drift')
    command_entries=json.loads((root/'command_contract.json').read_text())
    entries={(e['chapter'],e['python_block']):e for e in contract(root)}
    records=[];contents={};old_path=root/'book_sources/index.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else []
    for edition,folder in EDITIONS.items():
        for path in sorted((book_root/folder).glob('*.md')):
            text=path.read_text();python_index=0;command_index=0
            commands=[e for e in command_entries if e['edition']==edition and e['chapter']==path.name]
            for number,match in enumerate(fences(text),1):
                language,code=match.language,match.code
                entry=None;error=None
                if language=='python':
                    python_index+=1;entry=entries[(path.name,python_index)]
                    assert code==source_text(entry,root)
                    ast.parse(code)
                    try:compile(code,str(path),'exec')
                    except SyntaxError as exc:error=exc.msg
                command_source=None
                if language not in ('python','text'):
                    command_source=commands[command_index]['source'];command_index+=1
                sha=hashlib.sha256(code.encode()).hexdigest()
                same=next((r for r in records if r['chapter']==path.name and r['block']==number and r['sha256']==sha),None)
                artifact=same['artifact'] if same else f'book_sources/{edition}/{path.stem}/block-{number:02}.{language}.txt'
                contents[artifact]=code
                records.append(dict(edition=edition,chapter=path.name,block=number,
                    python_block=python_index if entry else None,line=text[:match.start].count('\n')+1,
                    language=language,kind=entry['kind'] if entry else 'reference_fragment',
                    artifact=artifact,runnable_file=entry['runnable_file'] if entry else None,
                    repository_source=entry['source'] if entry else command_source,
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
    command_diff=synchronize_commands(args.book_root.resolve(),write=args.write)
    if command_diff and not args.write:
        print('Command/configuration source drift:',command_diff);return 1
    if args.refresh_index:refresh_index(args.book_root.resolve())
    print(f'{len(diff)} source slots updated' if args.write else '321 Python slots match repository sources')
    return 0
if __name__=='__main__':raise SystemExit(main())
