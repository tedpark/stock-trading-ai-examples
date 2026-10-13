"""Repository source changes flow into all editions; book edits cannot rewrite source."""
from pathlib import Path
import shutil
import hashlib
import json
from unittest.mock import patch
import verify_book_sources
import sys
from tempfile import TemporaryDirectory
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sync_book

class RepositoryFirstTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        base=Path(self.temp.name).resolve();self.repo=base/'repo';self.book=base/'book';self.repo.mkdir()
        self.entries=sync_book.contract();shutil.copy(sync_book.ROOT/'book_contract.json',self.repo/'book_contract.json')
        for entry in self.entries:
            path=self.repo/entry['source'];path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(sync_book.ROOT/entry['source'],path)
        for folder in sync_book.EDITIONS.values():
            (self.book/folder).mkdir(parents=True)
            for chapter in {e['chapter'] for e in self.entries}:
                text=''.join(f"<!-- repository-source: {e['id']} -->\n```python\n{sync_book.source_text(e,self.repo)}```\n" for e in self.entries if e['chapter']==chapter)
                (self.book/folder/chapter).write_text(text)
        self.entry=next(e for e in self.entries if e['source']=='ch01_pipeline.py')
    def test_clean_and_idempotent(self):
        self.assertEqual(sync_book.synchronize(self.book,root=self.repo),[])
        self.assertEqual(sync_book.synchronize(self.book,write=True,root=self.repo),[])
    def test_repository_change_updates_every_edition(self):
        source=self.repo/self.entry['source'];source.write_text(source.read_text().replace('fee_bps=10','fee_bps=11'))
        self.assertEqual(len(sync_book.synchronize(self.book,root=self.repo)),3)
        sync_book.synchronize(self.book,write=True,root=self.repo)
        for folder in sync_book.EDITIONS.values():
            self.assertIn('fee_bps=11',(self.book/folder/self.entry['chapter']).read_text())
        self.assertEqual(sync_book.synchronize(self.book,root=self.repo),[])
    def test_book_edit_never_changes_repository(self):
        source=self.repo/self.entry['source'];before=source.read_bytes()
        path=self.book/'manuscript'/self.entry['chapter'];path.write_text(path.read_text().replace('fee_bps=10','fee_bps=999'))
        self.assertEqual(len(sync_book.synchronize(self.book,root=self.repo)),1)
        self.assertEqual(source.read_bytes(),before)
        sync_book.synchronize(self.book,write=True,root=self.repo)
        self.assertNotIn('fee_bps=999',path.read_text());self.assertEqual(source.read_bytes(),before)
    def test_new_unmapped_python_block_rejected_before_writing(self):
        path=self.book/'manuscript'/'99-new.md';path.write_text('```python\nprint(1)\n```\n')
        with self.assertRaises(ValueError):sync_book.synchronize(self.book,write=True,root=self.repo)
    def test_missing_block_rejected(self):
        path=self.book/'manuscript'/self.entry['chapter'];path.write_text('')
        with self.assertRaises(ValueError):sync_book.synchronize(self.book,root=self.repo)
    def test_wrong_anchor_rejected(self):
        path=self.book/'manuscript'/self.entry['chapter'];path.write_text(path.read_text().replace(self.entry['id'],'wrong-source'))
        with self.assertRaises(ValueError):sync_book.synchronize(self.book,root=self.repo)
    def test_rehashed_archive_cannot_override_canonical_source(self):
        shutil.copytree(sync_book.ROOT/'book_sources',self.repo/'book_sources')
        with patch.object(verify_book_sources,'ROOT',self.repo):
            verify_book_sources.verify()
            index=self.repo/'book_sources/index.json';rows=json.loads(index.read_text())
            row=next(r for r in rows if r['language']=='python')
            path=self.repo/row['artifact'];path.write_text('# manuscript-only edit\n'+path.read_text())
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            for r in rows:
                if r['artifact']==row['artifact']:r['sha256']=sha
            with self.assertRaises(AssertionError):verify_book_sources.verify(rows)

    def test_refresh_refuses_book_drift(self):
        path=self.book/'manuscript'/self.entry['chapter'];path.write_text(path.read_text().replace('fee_bps=10','fee_bps=99'))
        with self.assertRaises(ValueError):sync_book.refresh_index(self.book,root=self.repo)
        self.assertFalse((self.repo/'book_sources/index.json').exists())

if __name__=='__main__':unittest.main()
