from scripts.formula_round2_source_first import freeze_source_first_banks
def test_source_first_banks_share_exact_engine_rows():
 e=[{"expression":"x0","source":"engine:polynomial_lasso",
     "origin":"deterministic"},
    {"expression":"x0**2","source":"engine:mcts","origin":"deterministic"}]
 r=freeze_source_first_banks(e,
  [{"expression":"sin(x0)","source":"llm","origin":"llm"}],
  [{"expression":"exp(x0)","source":"llm","origin":"llm"}],n_features=1)
 assert r["banks"]["multi_engine"]==e
 assert r["banks"]["blind_pre_admission"][:2]==e
 assert r["banks"]["gap_pre_admission"][:2]==e
 assert r["engine_executions_per_coordinate"]==1

def test_numeric_constant_engine_rows_normalize_to_intercept():
 e=[{"expression":"x0","source":"engine:polynomial_lasso",
     "origin":"deterministic"},
    {"expression":"0","source":"engine:mcts","origin":"deterministic"},
    {"expression":"0.0","source":"engine:mcts","origin":"deterministic"}]
 r=freeze_source_first_banks(e,[],[],n_features=1)
 assert r["constant_to_intercept_count"]==2
 assert [row["expression"] for row in
         r["banks"]["multi_engine"]]==["x0","1","1"]

def test_engine_formula_over_ast_budget_is_ledgered_not_executed():
 huge="+".join(["x0"]*200)
 e=[{"expression":"x0","source":"engine:polynomial_lasso",
     "origin":"deterministic"},
    {"expression":"x0**2","source":"engine:mcts","origin":"deterministic"},
    {"expression":huge,"source":"engine:sparse_library",
     "origin":"deterministic"}]
 r=freeze_source_first_banks(e,[],[],n_features=1)
 assert r["raw_engine_candidate_count"]==3
 assert r["registered_engine_candidate_count"]==2
 rejection=r["candidate_eligibility_rejections"][0]
 assert rejection["arm"]=="engine"
 assert rejection["reason"]=="unadaptable-to-registered-grammar"
 assert "finite budget" in rejection["diagnostic"]

def test_missing_registered_single_engine_scores_empty_bank():
 e=[{"expression":"x0","source":"engine:mcts",
     "origin":"deterministic"},
    {"expression":"x0**2","source":"engine:sparse_library",
     "origin":"deterministic"}]
 r=freeze_source_first_banks(e,[],[],n_features=1)
 assert r["banks"]["single_engine"]==[]
 assert r["single_engine_generation_failed"] is True
 assert r["single_engine_failure_reason"]==(
  "registered-single-engine-produced-no-eligible-support")
