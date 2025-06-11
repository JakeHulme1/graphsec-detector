These outputs result from running the `export_vudenc_triples.py` preprocessing script

### File-level (`files.jsonl)
```json
{
  "repo": "https://github.com/LyleMi/Saker",
  "commit": "4a7915860cc482cb426cbf371ae785bfbae71881",
  "label": "command injection",
  "cwe_id": "CWE-77",
  "filepath": "saker/fuzzers/cmdi.py",
  "add": 32,
  "remove": 11,
  "badparts": [
    "@staticmethod",
    "def test(self):",
    "|id",
    ...
  ],
  "goodparts": [
    "splits = [",
    "@classmethod",
    "def test(cls, cmd=\"id\"):",
    ...
  ],
  "source": "from saker.fuzzers.fuzzer import Fuzzer\n\nclass CmdInjection(Fuzzer): ...",
  "sourceWithComments": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\nfrom saker.fuzzers..."
}
```

### Commit-level (`commits.jsonl)
```json
{
  "repo": "https://github.com/LyleMi/Saker",
  "commit": "4a7915860cc482cb426cbf371ae785bfbae71881",
  "label": "command injection",
  "cwe_id": "CWE-77",
  "keyword": "command injection update",
  "message": "update: command injection payload",
  "diff": "diff --git a/...",
  "files": [
    {
      "file": "saker/fuzzers/cmdi.py",
      "add": 32,
      "remove": 11,
      "badparts": [ ... ],
      "goodparts": [ ... ]
    }
  ],
  "total_add": 32,
  "total_remove": 11
}
```

### Repo-level (repos.jsonl)
```json
{
  "repo": "https://github.com/LyleMi/Saker",
  "total_commits": 5,
  "total_files_changed": 12,
  "total_add": 128,
  "total_remove": 47,
  "commits": [
    { /* full commit‐level object as above */ },
    ...
  ]
}
```