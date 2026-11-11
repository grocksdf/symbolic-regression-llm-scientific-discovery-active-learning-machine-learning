"""Fixed-budget fresh complete proposals. Only registered fitting data is visible."""
from dataclasses import asdict
import json
from pathlib import Path

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime, ProviderSettings, ProviderRoute
from hypothesis_mvp.discovery.region_site_attribution import digest
from hypothesis_mvp.discovery.structural_plan_pool import StructuralPlan
from scripts.provider_transport_preflight import _key
from scripts.run_regional_structural_matrix import write


SYSTEM = ('Return one JSON object with candidates: an array of exactly four objects '
          'with equation (complete RHS), parent_index (integer) and rationale (string). '
          'parent_index is ZERO-BASED: use only the explicitly listed allowed_parent_indices. '
          'Invent complementary symbolic predictions from the supplied training-only evidence. '
          'Use only x0,x1,..., finite numeric literals,+,-,*,/,constant powers,sin,cos,tanh,exp,log,sqrt,Abs. '
          'No target symbols, assignments, code, fitted scores or claimed admission. '
          'Different formulas should express different structural hypotheses.')


def generate(freeze, shared, fit, regions, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    settings=ProviderSettings(routes=(ProviderRoute(base_url=freeze['provider_url'],model=freeze['provider_model'],
        api_key=_key(Path(freeze['provider_env']))),),attempts=1,max_tokens=4096,
        thinking_type=freeze['thinking_type'],reasoning_effort=freeze['reasoning_effort'])
    runtime=ProposalRuntime(EquationRuntime(fit.X.shape[1],refit_policy='pcpi-expanded-fixed-inner-v1'),
                            fit.X.shape[1],settings,4)
    parents=[r['expression'] for r in shared['E0_rows']]
    # Fixed score-blind evenly spaced training examples, no audit/admission/reporting readers.
    import numpy as np
    indices=np.linspace(0,len(fit.X)-1,min(32,len(fit.X)),dtype=int)
    plans,audits,rejections=[],[],[]
    for call in range(8):
        payload={'contract':'selector-complete-training-only-v1','call_index':call,'parents':parents,
                 'n_features':fit.X.shape[1],'training_examples':[
                     {'x':fit.X[i].tolist(),'y':float(fit.y[i])} for i in indices],
                 'previous_equations':[p.expression for p in plans],
                 'allowed_parent_indices':list(range(len(parents)))}
        write(output/f'REQUEST_{call:02d}.json',{'system':SYSTEM,'payload':payload})
        raw,telemetry=runtime.complete_json(system_message=SYSTEM,payload=payload)
        write(output/f'RESPONSE_{call:02d}.json',{'raw':raw,'telemetry':telemetry})
        requests=telemetry.get('provider_requests',[telemetry])
        attempts=[o for r in requests for o in r.get('provider_outcomes',[])]
        if not attempts or any(o.get('provider_http_status')!=200 for o in attempts):
            raise ValueError('provider failure cannot become abstention')
        rows=raw.get('candidates')
        if not isinstance(rows,list) or len(rows)!=4: raise ValueError('exactly four complete proposals required')
        ids=[]
        for index,row in enumerate(rows):
            if (not isinstance(row,dict) or type(row.get('parent_index')) is not int
                    or not 0<=row['parent_index']<len(parents) or not isinstance(row.get('equation'),str)
                    or not isinstance(row.get('rationale'),str) or not row['rationale'].strip()):
                rejections.append({'call':call,'index':index,'reason':'invalid complete-proposal contract','raw_candidate':row})
                continue
            p=StructuralPlan(row['equation'],parents[row['parent_index']],regions[0]['id'],'complete_llm_proposal',
                'llm',digest([call,index,raw]),prompt_hash=digest([SYSTEM,payload]),response_hash=digest(raw))
            plans.append(p);ids.append(p.identity)
        audits.append({'plan_ids':ids,'batch_prompt_hash':digest([SYSTEM,payload]),'batch_response_hash':digest(raw),
                       'logical_calls':len(requests),'provider_attempts':len(attempts),
                       'prompt_tokens':sum(o['provider_prompt_tokens'] for o in attempts) if all(o.get('provider_prompt_tokens') is not None for o in attempts) else None,
                       'completion_tokens':sum(o['provider_completion_tokens'] for o in attempts) if all(o.get('provider_completion_tokens') is not None for o in attempts) else None})
    write(output/'COLLECTION_FROZEN.json',{'plans':[asdict(p) for p in plans],'audits':audits,'rejections':rejections})
    return tuple(plans),audits
