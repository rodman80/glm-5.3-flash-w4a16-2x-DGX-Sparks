#!/usr/bin/env python3
"""Feature-gated KernelBench-style MoE occupancy profiler for pinned vLLM."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
MARK="[glm53-kb-moe]"; SITE="usr/local/lib/python3.12/dist-packages/vllm"; REL="model_executor/layers/fused_moe/experts/marlin_moe.py"; OUT="marlin_moe.kb_profile.py"
IMPORT_OLD="import math\nfrom collections.abc import Callable\n"; IMPORT_NEW="import math\nimport json\nimport os\nfrom collections.abc import Callable\n"; HELPER_ANCHOR="from vllm.scalar_type import ScalarType, scalar_types\n\n\n"
HELPER='''from vllm.scalar_type import ScalarType, scalar_types

# [glm53-kb-moe] Sample routing before Marlin pads active experts to M>=8.
_GLM53_KB_PROFILE=os.getenv("GLM53_KB_MOE_PROFILE","0")=="1"
_GLM53_KB_SAMPLE_EVERY=max(1,int(os.getenv("GLM53_KB_MOE_SAMPLE_EVERY","64")))
_GLM53_KB_MAX_LOGS=max(1,int(os.getenv("GLM53_KB_MOE_MAX_LOGS","2000")))
_GLM53_KB_CALLS=0; _GLM53_KB_LOGS=0

def _glm53_kb_profile_moe(topk_ids,logical_m,local_experts,global_experts,topk,block_size_m):
 global _GLM53_KB_CALLS,_GLM53_KB_LOGS
 if not _GLM53_KB_PROFILE or _GLM53_KB_LOGS>=_GLM53_KB_MAX_LOGS:return
 _GLM53_KB_CALLS+=1
 if (_GLM53_KB_CALLS-1)%_GLM53_KB_SAMPLE_EVERY:return
 with torch.no_grad():
  flat=topk_ids.reshape(-1); flat=flat[flat>=0].to(torch.int64)
  if flat.numel()==0:return
  n=int(global_experts)
  if n<=0:n=max(int(local_experts),int(flat.max().item())+1)
  c=torch.bincount(flat,minlength=n)[:n]; nz=c[c>0]; logical=int(c.sum().item()); p=torch.where(c>0,((c+block_size_m-1)//block_size_m)*block_size_m,c); padded=int(p.sum().item())
  b={"0":int((c==0).sum().item()),"1":int((c==1).sum().item()),"2":int((c==2).sum().item()),"3":int((c==3).sum().item()),"4":int((c==4).sum().item()),"5_8":int(((c>=5)&(c<=8)).sum().item()),"9_16":int(((c>=9)&(c<=16)).sum().item()),"17_32":int(((c>=17)&(c<=32)).sum().item()),"33_plus":int((c>=33).sum().item())}
  x={"call":_GLM53_KB_CALLS,"host":os.getenv("VLLM_HOST_IP",""),"logical_m":int(logical_m),"topk":int(topk),"global_experts":n,"active_experts":int((c>0).sum().item()),"block_size_m":int(block_size_m),"logical_slots":logical,"marlin_padded_slots_est":padded,"padding_overhead_x":padded/logical if logical else 0.0,"mean_m_active_expert":float(nz.float().mean().item()) if nz.numel() else 0.0,"max_m_expert":int(nz.max().item()) if nz.numel() else 0,"buckets":b}
 _GLM53_KB_LOGS+=1; print(f"[glm53-kb-moe] {json.dumps(x,separators=(',',':'))}",flush=True)

'''
CALL_OLD="""    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
"""; CALL_NEW="""    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)

    _glm53_kb_profile_moe(topk_ids, hidden_states.size(0), E, global_num_experts, topk, block_size_m)  # [glm53-kb-moe]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
"""
def repl(t,o,n,l):
 c=t.count(o)
 if c!=1:raise SystemExit(f"anchor error [{l}]: expected 1, found {c}")
 return t.replace(o,n,1)
def extract(image,work):
 work.mkdir(parents=True,exist_ok=True); p=work/"marlin_moe.pristine.py"; cid=subprocess.run(["docker","create",image],check=True,text=True,capture_output=True).stdout.strip()
 try:subprocess.run(["docker","cp",f"{cid}:/{SITE}/{REL}",str(p)],check=True,capture_output=True)
 finally:subprocess.run(["docker","rm",cid],check=False,capture_output=True)
 return p.read_text()
def verify(p):
 t=p.read_text()
 for n in (MARK,"_glm53_kb_profile_moe(",'"marlin_padded_slots_est"',"moe_align_block_size("):
  if n not in t:raise SystemExit(f"verify missing {n}")
 compile(t,str(p),"exec");print(f"KB1 overlay verify OK: {p}")
def main():
 a=argparse.ArgumentParser();a.add_argument("--image",required=True);a.add_argument("--work-dir",required=True);a.add_argument("--verify-only",action="store_true");x=a.parse_args();w=Path(x.work_dir);o=w/OUT
 if x.verify_only:verify(o);return 0
 t=extract(x.image,w);t=repl(t,IMPORT_OLD,IMPORT_NEW,"imports");t=repl(t,HELPER_ANCHOR,HELPER,"helper");t=repl(t,CALL_OLD,CALL_NEW,"call");o.write_text(t);verify(o);return 0
if __name__=="__main__":sys.exit(main())
