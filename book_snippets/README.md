# Maintained explanatory source

These 73 canonical Python fragments are repository-owned source for the printed explanations, shared by all editions. They were initialized from the already reviewed English source snapshot; future changes start here, never in the manuscript or `book_sources` archive.

They are deliberately stored as `.py.txt`: some require surrounding functions/objects, and four are intentional counterexamples. They are not automatically imported or executed as standalone programs. Tests execute selected functions with controlled collaborators.

`book_contract.json` specifies each fragment's type and manuscript slot. Runnable examples instead point directly to the actual chapter `.py` files. `sync_book.py` generates the book listings; `verify_book_sources.py` rejects a derived snapshot that differs from canonical source.
