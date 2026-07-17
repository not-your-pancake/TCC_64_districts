#!/usr/bin/env python3
"""
Merge the co-author's literature-review BibTeX (from the .txt) with the
existing references.bib into one reconciled manuscript/references.bib:
  - dedupe keys defined twice (keeps first occurrence)
  - drop same-DOI duplicate papers (kamal2024wbgt, nakagawa2026)
  - convert @standard -> @misc (elsarticle-num can't format @standard)
  - carry over the existing-only entries not superseded by the new file
Run:  py -3.14 manuscript/build_bib.py
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\hp\Downloads\reference_for_literature_review_with_bibtext.txt"
EXIST = os.path.join(HERE, "references.bib")
OUT = os.path.join(HERE, "references.bib")


def extract(text):
    """Return list of (key_lower, raw_entry) with balanced braces."""
    out, i = [], 0
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        brace = text.find("{", at)
        typ = text[at + 1:brace].strip()
        if not re.match(r"^[A-Za-z]+$", typ):   # skip stray @ in text
            i = at + 1
            continue
        depth, j = 0, brace
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        raw = text[at:j + 1]
        key = raw[brace - at + 1:].split(",", 1)[0].strip()
        out.append((key.lower(), key, raw))
        i = j + 1
    return out


new = extract(open(SRC, encoding="utf-8", errors="replace").read())
exist = extract(open(EXIST, encoding="utf-8", errors="replace").read())

# same-DOI duplicates to drop (keep the other key)
DROP_NEW = {"kamal2024wbgt", "nakagawa2026"}
# existing entries superseded by a new-file version (semantic dups not caught by key)
DROP_EXIST = {"kamal2024", "iso7243", "chen2016", "ke2017", "prokhorenkova2018"}

def escape_specials(raw):
    """Escape underscores inside doi/url fields (elsarticle-num prints them raw,
    so a bare '_' is read as a subscript -> 'Missing $ inserted')."""
    def repl(m):
        return m.group(0).replace("_", r"\_")
    return re.sub(r'(doi|url)\s*=\s*\{[^}]*\}', repl, raw, flags=re.I)


seen, merged = set(), []
for kl, k, raw in new:
    if kl in seen or kl in DROP_NEW:
        continue
    if raw.lstrip().startswith("@standard"):
        raw = raw.replace("@standard", "@misc", 1)
    seen.add(kl)
    merged.append(escape_specials(raw))

added_exist = []
for kl, k, raw in exist:
    if kl in seen or kl in DROP_EXIST:
        continue
    seen.add(kl)
    added_exist.append(escape_specials(raw))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("%% Reconciled bibliography: co-author literature-review set + "
            "existing manuscript refs.\n")
    f.write("%% Regenerate with build_bib.py. [verify-intended] notes flag "
            "entries added by the assistant.\n\n")
    f.write("%% ===== Literature-review references (co-author, DOI-verified) =====\n\n")
    f.write("\n\n".join(e.strip() for e in merged))
    f.write("\n\n%% ===== Existing manuscript references (not in the new set) =====\n\n")
    f.write("\n\n".join(e.strip() for e in added_exist))
    f.write("\n")

print(f"new entries kept: {len(merged)} | existing carried: {len(added_exist)} | "
      f"total: {len(merged) + len(added_exist)}")
print("dropped same-DOI dups:", sorted(DROP_NEW))
print("dropped superseded existing:", sorted(DROP_EXIST))
