"""Synthetic contract checks; provider transport is replaced, no efficacy."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from scripts import selector_llm_proposals as module


def test_fixed_budget_training_only_collection(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'_key',lambda p:'not-a-real-key')
    calls=[]
    def complete(self,*,system_message,payload):
        calls.append(payload)
        c=payload['call_index']
        return {'candidates':[{'equation':f'x0**{c*4+i+1}','parent_index':0,'rationale':'test'} for i in range(4)]}, {
            'provider_outcomes':[{'provider_http_status':200,'provider_prompt_tokens':10,'provider_completion_tokens':20}]}
    monkeypatch.setattr(module.ProposalRuntime,'complete_json',complete)
    fit=SimpleNamespace(X=np.arange(1.,41.)[:,None],y=np.arange(1.,41.))
    plans,audits=module.generate({'provider_url':'https://example.test/v1','provider_model':'test','provider_env':'unused','thinking_type':'enabled','reasoning_effort':'low'},
        {'E0_rows':[{'expression':'x0'}]},fit,[{'id':'r'}],tmp_path/'collection')
    assert len(calls)==8 and len(plans)==32 and len(audits)==8
    assert all(set(c)=={'contract','call_index','parents','n_features','training_examples','previous_equations','allowed_parent_indices'} for c in calls)
    assert (tmp_path/'collection'/'COLLECTION_FROZEN.json').exists()
    assert len({p.identity for p in plans})==32
    assert audits[0]['prompt_tokens']==10


def test_transport_failure_preserves_request_and_response(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'_key',lambda p:'not-a-real-key')
    monkeypatch.setattr(module.ProposalRuntime,'complete_json',lambda *a,**k: ({'candidates':[]},{'provider_outcomes':[{'provider_http_status':429}]}))
    with pytest.raises(ValueError,match='provider failure'):
        module.generate({'provider_url':'https://example.test/v1','provider_model':'test','provider_env':'unused','thinking_type':'enabled','reasoning_effort':'low'},
            {'E0_rows':[{'expression':'x0'}]},SimpleNamespace(X=np.ones((2,1)),y=np.ones(2)),[{'id':'r'}],tmp_path/'bad')
    assert (tmp_path/'bad'/'RESPONSE_00.json').exists()
    assert not (tmp_path/'bad'/'COLLECTION_FROZEN.json').exists()


def test_score_blind_pool_budget_preserves_unscored_alternatives():
    from scripts.run_predictive_plan_matrix import selection_blocks
    for size in range(4,100):
        assert 2*selection_blocks(size)<=size/2
        assert 1<=selection_blocks(size)<=8
    for size in (0,1,2,3):
        with pytest.raises(ValueError): selection_blocks(size)


def test_bad_parent_rejects_candidate_without_remapping_or_aborting_batch(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'_key',lambda p:'not-a-real-key')
    def complete(*a,**kw):
        return {'candidates':[{'equation':'x0','parent_index':i,'rationale':'test'} for i in range(4)]},{'provider_outcomes':[{'provider_http_status':200}]}
    monkeypatch.setattr(module.ProposalRuntime,'complete_json',complete)
    plans,audits=module.generate({'provider_url':'https://example.test/v1','provider_model':'test','provider_env':'unused','thinking_type':'enabled','reasoning_effort':'low'},
        {'E0_rows':[{'expression':'x0'}]},SimpleNamespace(X=np.ones((2,1)),y=np.ones(2)),[{'id':'r'}],tmp_path/'partial')
    assert len(plans)==8
    frozen=json.loads((tmp_path/'partial'/'COLLECTION_FROZEN.json').read_text())
    assert len(frozen['rejections'])==24 and len(audits)==8
