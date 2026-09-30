import logging
import multiprocessing
import os
import sys
import yaml

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace

V101_EVAL_ENTRYPOINT_CONTRACT = "v10.1-budgeted-auditable-restart"

from bench.datamodules import get_datamodule
from bench.pipelines import EvaluationPipeline

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.info("[TRACE] eval.py cwd=%s", Path.cwd())
logging.info("[TRACE] eval.py file=%s", Path(__file__).resolve())
logging.info("[TRACE] sys.path_head=%s", sys.path[:5])
logging.info("[PCPI][v10.1] eval_entrypoint_contract=%s", V101_EVAL_ENTRYPOINT_CONTRACT)

parser = ArgumentParser()
parser.add_argument('--searcher_config', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--ds_root_folder', type=str, default=None)
parser.add_argument('--resume_from', type=str, default=None)
parser.add_argument('--problem_name', type=str, default=None)
parser.add_argument('--max_problems', type=int, default=None)
parser.add_argument('--local_llm_port', type=int, default=None)
parser.add_argument('--seed', type=int, default=None, help='Override searcher random seed for paired runs')
parser.add_argument('--budget', type=int, default=None, help='Hard candidate-validation budget for paired runs')
args = parser.parse_args()

# v771: carry the benchmark dataset family into inner PCPI code paths.
# This is metadata/environment propagation only; selection remains train/val-only.
os.environ.setdefault("PCPI_DATASET_NAME", str(args.dataset))
os.environ.setdefault("LLMSRBENCH_DATASET", str(args.dataset))
os.environ.setdefault("PCPI_TASK_FAMILY", str(args.dataset))
if str(args.dataset).strip().lower() == "phys_osc":
    os.environ["PCPI_PHYS_OSC_MODE"] = "1"
logging.info("[PCPI][v771] eval dataset_family=%s phys_osc_mode=%s", os.environ.get("PCPI_TASK_FAMILY"), os.environ.get("PCPI_PHYS_OSC_MODE", "0"))

now = datetime.now()
now_str = now.strftime("%m-%d-%Y_%H-%M-%S-%f")

dm = get_datamodule(name=args.dataset, root_folder=args.ds_root_folder)
dm.setup()

# Load searcher configuration from yaml file
with open(args.searcher_config) as f:
    searcher_cfg = yaml.safe_load(f)
searcher_cfg = Namespace(**searcher_cfg)

# Paired-run protocol overrides. The auditable runner passes these through the CLI;
# older runners can still set LLMSRBENCH_PAIRED_SEED / PCPI_LLM_SR_BUDGET.
seed_override = args.seed
if seed_override is None:
    for _env_key in ("LLMSRBENCH_PAIRED_SEED", "PCPI_EFFECTIVE_SEED", "PCPI_FIXED_SEED", "PCPI_LLM_SR_SEED"):
        _raw = os.environ.get(_env_key)
        if _raw not in (None, ""):
            try:
                seed_override = int(_raw)
                break
            except ValueError:
                logging.warning("Ignoring non-integer %s=%r", _env_key, _raw)
if seed_override is not None:
    setattr(searcher_cfg, "random_seed", int(seed_override))
    setattr(searcher_cfg, "seed", int(seed_override))
    logging.info("[PROTOCOL] effective_seed=%s", seed_override)

budget_override = (
    args.budget
    if args.budget is not None
    else os.environ.get("PCPI_EFFECTIVE_BUDGET") or os.environ.get("PCPI_LLM_SR_BUDGET")
)
if budget_override not in (None, ""):
    try:
        _budget_int = int(float(str(budget_override)))
    except Exception:
        raise ValueError(f"Invalid positive evaluation budget: {budget_override!r}")
    if _budget_int <= 0:
        raise ValueError(f"Invalid positive evaluation budget: {_budget_int}")
    setattr(searcher_cfg, "budget", _budget_int)
    setattr(searcher_cfg, "evaluation_budget", _budget_int)
    for _budget_key in ("global_max_sample_num", "max_num_samples"):
        if hasattr(searcher_cfg, _budget_key):
            setattr(searcher_cfg, _budget_key, _budget_int)
    logging.info("[PROTOCOL] effective_evaluation_budget=%s", _budget_int)

# Set up output directories for logging results
# If not resuming from previous run:
#   - Create new timestamped output directory under logs/{dataset}/{searcher}/
# If resuming:
#   - Use existing directory specified by resume_from argument
# Creates search_logs subdirectory and sets searcher's log path
if args.resume_from is None:
    output_root = Path(__file__).resolve().parent / "logs"
    output_path = output_root / dm.name / searcher_cfg.name / now_str
    output_path.mkdir(parents=True, exist_ok=True)
else:
    output_path = Path(args.resume_from)
searcher_log_path = output_path / "search_logs"
searcher_log_path.mkdir(exist_ok=True, parents=True)

temp_dir = Path("logs/tmp")
temp_dir.mkdir(exist_ok=True, parents=True)

# Set API key based on the searcher configuration type
# hfinf: HuggingFace Inference API
# vllm: Local VLLM server
# openai: OpenAI API
if searcher_cfg.api_type == "hfinf":
    api_key = os.environ['HFINF_API_KEY']
elif searcher_cfg.api_type == "vllm":
    api_key = os.environ['VLLM_API_KEY']
    searcher_cfg.api_url = searcher_cfg.api_url.format(args.local_llm_port)
elif searcher_cfg.api_type == "openai":
    api_key = os.environ.get('OPENAI_API_KEY', '')
else:
    api_key = None

if api_key:
    logging.info(
        "Using API key from %s (present, length=%s)",
        searcher_cfg.api_type,
        len(api_key),
    )
else:
    logging.warning("No API key configured for api_type=%s", searcher_cfg.api_type)


# Initialize searcher based on the searcher configuration class name
# LLMSRSearcher
# LasrSearcher
# Each searcher is initialized with specific configurations and parameters
if searcher_cfg.class_name == 'LLMSRSearcher':
    sys.path.append(os.path.join(os.path.dirname(__file__), "methods"))
    from methods.llmsr.searcher import LLMSRSearcher
    from methods.llmsr import config, sampler
    # os.environ["LLM_SR_SERVER_PORT"] = str(args.port)

    exp_conf = config.ExperienceBufferConfig(
        num_islands=searcher_cfg.num_islands
    )
    cfg = config.Config(
        experience_buffer=exp_conf,
        use_api = searcher_cfg.api_type != 'local',
        api_model = searcher_cfg.api_model,
        samples_per_prompt = searcher_cfg.samples_per_prompt,
    )
    sampler_class = lambda samples_per_prompt: sampler.LocalLLM(
        samples_per_prompt=samples_per_prompt,
        local_llm_url=searcher_cfg.api_url,
        api_url=searcher_cfg.api_url,
        api_key=api_key,
    )
    searcher = LLMSRSearcher(searcher_cfg.name,
                            cfg,
                            sampler_class,
                            global_max_sample_num=searcher_cfg.global_max_sample_num,
                            log_path=searcher_log_path)
elif searcher_cfg.class_name == 'LasrSearcher':
    sys.path.append(os.path.join(os.path.dirname(__file__), "methods"))
    from methods.lasr.searcher import LasrSearcher

    searcher = LasrSearcher(
        name=searcher_cfg.name,
        api_key=api_key,
        model=searcher_cfg.api_model,
        model_url=searcher_cfg.api_url,
        prompts_path='methods/lasr/prompts/',
        log_path=searcher_log_path,
        temp_dir=temp_dir,
        num_iterations=searcher_cfg.num_iterations,
        num_populations=searcher_cfg.num_populations,
        llm_weight=searcher_cfg.llm_weight,
        early_stopping_condition=searcher_cfg.early_stopping_condition,
        max_num_samples=searcher_cfg.max_num_samples,
    )
elif searcher_cfg.class_name == 'SGASearcher':
    sys.path.append(os.path.join(os.path.dirname(__file__), "methods",  "sga_sr"))
    from methods.sga_sr.searcher import SGASearcher
    searcher = SGASearcher(
        name=searcher_cfg.name,
        root=Path("methods/sga_sr").absolute(),
        path=str(searcher_log_path.absolute()),
        python_path=os.environ['SGA_PYTHON_PATH'],
        dataset_name=args.dataset,
        dataset_path=args.ds_root_folder,
        llm_api_url=searcher_cfg.api_url,
        llm_model=searcher_cfg.api_model,
        llm_api_key=api_key,
    )

elif searcher_cfg.class_name in {
        'HypothesisMVPSearcher', 'PCPISearcher', 'DRRBenchmarkSearcher'}:
    workspace_root = Path(__file__).resolve().parents[1]
    mainline_root = workspace_root / "hypothesis_mvp"
    if str(mainline_root) not in sys.path:
        sys.path.insert(0, str(mainline_root))
    from methods.hypothesis_mvp_pcpi.searcher import PCPISearcher
    from methods.hypothesis_mvp_pcpi.drr_searcher import DRRBenchmarkSearcher

    # Config aliases used by different experiment packs.  This keeps the
    # no-LLM/LLM paired ablation honest: use_llm=True is propagated into the
    # PCPI adapter, while no-LLM configs leave all LLM endpoints disabled.
    llm_enabled = bool(getattr(searcher_cfg, "llm_rl_enabled", getattr(searcher_cfg, "use_llm", False)))
    llm_api_url = (
        getattr(searcher_cfg, "llm_api_url", None)
        or getattr(searcher_cfg, "llm_api_base", None)
        or getattr(searcher_cfg, "api_url", None)
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )
    _env_llm_model = str(os.environ.get("LLM_MODEL") or "").strip()
    _yaml_llm_model = str(getattr(searcher_cfg, "llm_model", None) or "").strip()
    _yaml_api_model = str(getattr(searcher_cfg, "api_model", None) or "").strip()
    llm_model = _env_llm_model or _yaml_llm_model or _yaml_api_model or None
    llm_model_resolution_source = (
        "env.LLM_MODEL" if _env_llm_model
        else "yaml.llm_model" if _yaml_llm_model
        else "yaml.api_model" if _yaml_api_model
        else "unset"
    )
    os.environ["PCPI_LLM_MODEL_RESOLUTION_SOURCE"] = llm_model_resolution_source
    if llm_model:
        os.environ["PCPI_EFFECTIVE_LLM_MODEL"] = str(llm_model)
    if llm_enabled and _env_llm_model and str(llm_model or "") != _env_llm_model:
        raise RuntimeError(
            "v101_eval_llm_model_resolution_mismatch:"
            f"env={_env_llm_model!r}:effective={llm_model!r}:source={llm_model_resolution_source!r}"
        )
    logging.info(
        "[PCPI][v10.1] effective_llm_model=%s resolution_source=%s env_match=%s",
        llm_model,
        llm_model_resolution_source,
        bool(not _env_llm_model or str(llm_model or "") == _env_llm_model),
    )

    searcher_class = (
        DRRBenchmarkSearcher
        if searcher_cfg.class_name == 'DRRBenchmarkSearcher'
        else PCPISearcher)
    searcher_kwargs = dict(
        name=searcher_cfg.name,
        top_k_results=getattr(searcher_cfg, "top_k_results", 3),
        train_ratio=getattr(searcher_cfg, "train_ratio", getattr(searcher_cfg, "internal_train_frac", 0.7)),
        random_seed=getattr(searcher_cfg, "random_seed", getattr(searcher_cfg, "seed", 42)),
        use_library=getattr(searcher_cfg, "use_library", True),
        output_dir=str(output_path),
        library_path=(None if getattr(searcher_cfg, "strict_train_val_only", True) else getattr(searcher_cfg, "library_path", None)),
        llm_rl_enabled=llm_enabled,
        llm_api_url=llm_api_url if llm_enabled else None,
        llm_model=llm_model if llm_enabled else None,
        llm_model_resolution_source=llm_model_resolution_source,
        expected_env_llm_model=_env_llm_model,
        llm_api_key=api_key if llm_enabled else None,
        llm_timeout_s=getattr(searcher_cfg, "llm_timeout_s", 20.0),
        llm_thinking_type=getattr(searcher_cfg, "llm_thinking_type", ""),
        llm_reasoning_effort=getattr(
            searcher_cfg, "llm_reasoning_effort", ""),
        llm_do_sample=getattr(searcher_cfg, "llm_do_sample", None),
        use_restart_controller=getattr(searcher_cfg, "use_restart_controller", False),
        restart_action_ledger_path=getattr(searcher_cfg, "restart_action_ledger_path", None),
        discovery_config=vars(searcher_cfg),
    )
    if searcher_class is DRRBenchmarkSearcher:
        searcher_kwargs.update({
            "condition": getattr(searcher_cfg, "condition"),
            "agent_config": getattr(searcher_cfg, "agent_config"),
            "portfolio_method": getattr(searcher_cfg, "portfolio_method"),
        })
    searcher = searcher_class(**searcher_kwargs)
else:
    raise ValueError

# Filter problems based on command line arguments:
# - If problem_name is specified, only keep that specific problem
# - If max_problems is specified, keep only the first N problems after filtering
# - Otherwise use all problems from the dataset
problems = dm.problems
if args.problem_name is not None:
    problems = list(filter(lambda p: p.equation_idx == args.problem_name, problems))
if args.max_problems is not None:
    if args.max_problems < 0:
        raise ValueError("--max_problems must be non-negative")
    problems = problems[:args.max_problems]
print(f"Total number of problems: {len(problems)}")


# Create evaluation pipeline and run evaluation on all problems
# - Creates new EvaluationPipeline instance
# - Evaluates each problem using the configured searcher
# - Saves results to output_path in JSONL format with metrics
def _run_evaluation():
    pipeline = EvaluationPipeline()
    pipeline.evaluate_problems(
        problems,
        searcher,
        output_path,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _run_evaluation()
