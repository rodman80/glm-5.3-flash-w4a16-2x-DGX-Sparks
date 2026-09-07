#!/usr/bin/env python3
import argparse,json,statistics,sys
from collections import Counter
M="[glm53-kb-moe] "; B=("0","1","2","3","4","5_8","9_16","17_32","33_plus")
def main():
 a=argparse.ArgumentParser();a.add_argument("path");a.add_argument("--label",default="profile");x=a.parse_args();rows=[]
 for line in open(x.path,encoding="utf-8",errors="replace"):
  p=line.find(M)
  if p>=0:
   try:rows.append(json.loads(line[p+len(M):].strip()))
   except json.JSONDecodeError:pass
 if not rows:print("no KB1 rows",file=sys.stderr);return 2
 b=Counter();logical=padded=0;pad=[];means=[];maxes=[];ms=Counter();blocks=Counter()
 for r in rows:
  ms[r["logical_m"]]+=1;blocks[r["block_size_m"]]+=1
  for k in B:b[k]+=int(r["buckets"].get(k,0))
  logical+=r["logical_slots"];padded+=r["marlin_padded_slots_est"];pad.append(r["padding_overhead_x"]);means.append(r["mean_m_active_expert"]);maxes.append(r["max_m_expert"])
 tiny=sum(b[k] for k in ("1","2","3","4"));nz=sum(b[k] for k in B if k!="0");le8=tiny+b["5_8"];pct=lambda q,n:100*q/n if n else 0
 print(f"# KB1 MoE profile — {x.label}\nsamples: {len(rows)}\nlogical_m: {dict(sorted(ms.items()))}\nblock_size_m: {dict(sorted(blocks.items()))}\nexpert occupancy:")
 for k in B:print(f"  {k:>7}: {b[k]:>10} ({pct(b[k],sum(b.values())):6.2f}%)")
 ratio=padded/max(logical,1);share=tiny/max(nz,1)
 print(f"active experts M<=4: {pct(tiny,nz):.2f}%\nactive experts M<=8: {pct(le8,nz):.2f}%\nmean M active: {statistics.fmean(means):.3f}\nmedian max-M: {statistics.median(maxes):.1f}\nestimated Marlin padded slots: {padded}/{logical} = {ratio:.3f}x\nmedian padding overhead: {statistics.median(pad):.3f}x")
 print("KB2 verdict:","STRONG GO" if share>=.70 and ratio>=2 else "GO" if share>=.50 or ratio>=1.5 else "WEAK");return 0
if __name__=="__main__":raise SystemExit(main())
