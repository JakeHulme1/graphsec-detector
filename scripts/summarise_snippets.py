import glob, re, pandas as pd

rows=[]
pat  = re.compile(r'windows kept\s*:\s*([\d,]+)\s*/\s*([\d,]+)')
spar = re.compile(r'sparse positives\s*:\s*([\d,]+)')
dupl = re.compile(r'duplicate windows\s*:\s*([\d,]+)')
long = re.compile(r'>\d+ subtoks\s*:\s*([\d,]+)')

for logfile in glob.glob('logs/*.log'):
    tag = logfile.split('/')[-1].split('.log')[0]
    txt = open(logfile).read()
    keep, total = map(lambda x:int(x.replace(',','')), pat.search(txt).groups())
    d_sparse    = int(spar.search(txt).group(1).replace(',',''))
    d_dupes     = int(dupl.search(txt).group(1).replace(',',''))
    d_long      = int(long.search(txt).group(1).replace(',',''))
    rows.append((tag, total, keep, d_sparse, d_dupes, d_long))

df = pd.DataFrame(rows, columns=["tag","total","kept",
                                 "d_sparse","d_dupes","d_long"])
df["pos_rate"] = df["kept"].sub(df["d_sparse"]).div(df["kept"])  # crude estimate
print(df.sort_values("kept", ascending=False).to_string(index=False))
