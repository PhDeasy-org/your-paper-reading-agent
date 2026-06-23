# Progress Ledger — arXiv HTML Reader Plan

## Task List
- [x] Task 1: Create `arxiv_html.py` with migrated data symbols + section mapping
- [x] Task 2: Implement the HTML parser
- [x] Task 3: Implement `fetch_and_parse`
- [x] Task 4: Wire `pipeline.py` to use arXiv HTML with PDF fallback
- [x] Task 5: Drop the `vision` LLM role from `config.py` and add `max_figures`
- [x] Task 6: Delete `chat_vision()` from `llm.py` and clean up agents
- [ ] Task 7: Delete the old `figures.py` module and its tests
- [ ] Task 8: Update TUI to drop the vision menu entry
- [ ] Task 9: Update `cli.py` (`config_show`) and run final verification
- [ ] Task 10: Update README and run end-to-end smoke test

## History
Task 1: complete (prior session, commits 5107a4d)
Task 2: complete (prior session, commits 59670ec)
Task 3: complete (prior session, commits 8bd4440)
Task 4: complete (prior session, commits 498744f)
Task 5: complete (prior session, commits c2e037b)
Task 6: complete (prior session, commits bb51132)

## Minor Findings
