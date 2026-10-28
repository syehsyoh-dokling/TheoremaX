"""know87: for each know87 unit: is its curated frame cf-<stem> first in frame_bank.select, rendered whole (no
'[frame truncated]'), and what does names_for list first. (ids used only to measure.) Copy of check_frames81.py."""
import pathlib, sys
sys.path.insert(0, r"C:\Users\Saifuddin\Documents\Agenthon-2026\submissions\2026-10-13\t1_know89_ctx")
from agent_core import frame_bank, toolkit_hints
units = pathlib.Path(r"C:\Users\Saifuddin\Documents\track1-coding-public\units")
fails = ["t1-barrier-garch-var", "t1-corporate-action-adjustment", "t1-cta-basel-capital", "t1-evt-pot-var",
         ]
ok = 0
for u in fails:
    text = (units / u / "instruction.md").read_text(encoding="utf-8", errors="replace")
    sel = frame_bank.select(text)
    ids = [f.get("id") for f in sel]
    want = "cf-" + u[3:]
    blk = frame_bank.render(sel)
    first = ids[:1] == [want] or ids[:1] == ["cf-" + u]
    whole = "[frame truncated]" not in blk.split("F2.")[0]
    ok += first and whole
    print(f"{'ok ' if first and whole else '-- '}{u}: top={ids[:2]} whole_F1={whole} hints={toolkit_hints.names_for(text)[:3]}")
print(f"curated frame first and whole: {ok}/{len(fails)}")
