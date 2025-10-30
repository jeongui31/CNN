import os 

out_lines = []

with open("data/mr/neg", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        out_lines.append(f"0 {line}")

with open("data/mr/pos", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        out_lines.append(f"1 {line}")

with open("data/mr/all", "w", encoding="utf-8") as out:
    out.write("".join(out_lines))

print(f"[done] saved {len(out_lines)} lines to \"data/mr/all\"")
