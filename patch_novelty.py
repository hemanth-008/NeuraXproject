import re
p = "src/vision/infer.py"
s = open(p, encoding="utf-8").read()
pattern = re.compile(r"^(\s*)novel_score = self\._novelty\(embedding\) if embedding is not None else None\s*$", re.MULTILINE)
m = pattern.search(s)
if not m:
    print("NOT FOUND - tell Claude")
else:
    indent = m.group(1)
    replacement = f"{indent}try:\n{indent}    novel_score = self._novelty(embedding) if embedding is not None else None\n{indent}except Exception:\n{indent}    novel_score = None"
    s2 = pattern.sub(replacement, s, count=1)
    open(p, "w", encoding="utf-8").write(s2)
    print("PATCHED")
