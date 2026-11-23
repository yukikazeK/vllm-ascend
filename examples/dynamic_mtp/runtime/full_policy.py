import json,os,time
from pathlib import Path
root=Path(os.environ['DYN_FULL_ROOT'])
_last_policy = None
_probe_pending = False
def event(name,**fields):
    global _last_policy, _probe_pending
    if name == 'policy':
        if not (root/'timing').exists():
            _last_policy = fields['k']
        elif _last_policy == fields['k']:
            return
        _last_policy = fields['k']
    elif name in ('late_switch','gdn_transfer'):
        pass
    elif name == 'dispatch':
        _probe_pending = fields['bs'] == 3 and fields['width'] == 2
        if not _probe_pending:
            return
    elif name == 'target_full' and _probe_pending:
        _probe_pending = False
    else:
        return
    with (root/'trace.jsonl').open('a') as f:print(json.dumps({'t':time.monotonic(),'event':name,**fields}),file=f)

def select(worker,scheduler_output):
    # Synchronous TP1/DP1 experiment. Skipped requests have no fresh drafter KV.
    # Resume positive K only once the scheduled requests have valid draft history.
    dirty=getattr(worker,'_skipped_draft_reqs',set())
    dirty.difference_update(scheduler_output.finished_req_ids)
    reqs=set(scheduler_output.num_scheduled_tokens)
    want=scheduler_output.num_spec_tokens_to_schedule
    if want<0 or want>worker.num_speculative_steps:raise ValueError(f'Unsupported draft tier {want}')
    k=0 if reqs & dirty else want
    if k==0:dirty.update(reqs)
    worker._skipped_draft_reqs=dirty
    worker._current_draft_k=k
    event('policy',bs=len(reqs),requested=want,k=k,previous_draft_tokens=sum(map(len,scheduler_output.scheduled_spec_decode_tokens.values())))
