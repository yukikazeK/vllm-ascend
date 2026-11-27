# SPDX-License-Identifier: Apache-2.0
import json,os
from pathlib import Path
def rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return int(os.environ.get("LOCAL_RANK","0"))
def append(path,row):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(dict(row,pid=os.getpid(),tp_rank=rank()),ensure_ascii=False)+"\n").encode()
    fd=os.open(str(path),os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o644)
    try: os.write(fd,data)
    finally: os.close(fd)
