"""
MLflow Model Registry integration for LexRAG.

Wraps the best-performing retriever config as a registered MLflow model.
What's being versioned here is the CONFIGURATION (chunk_size, retrieval_mode, embedding_model. rrf_k) that produced the best RAGAS scores - not trained weights.

Uses model version ALIASES, not stages. MLflow deprecated stages
(None/Staging/Production/Archived) in favor of aliases starting in
MLflow 2.8/2.9. Aliases are mutable named pointers (e.g. "champion")
that can be reassigned to any version.

Workflow: 
    1. Each experiment run logs ints config + RAGAS scores + completeness
    2. register_best_config() finds the run with the best composite score
       AMONG RUNS WITH COMPLETE DATA ONLY - partial runs are excluded
       even if their average looks high.
    3. The winning config is registered as a new model version
    4. The "champion" alias is reassigned to that version
"""

import tempfile
from typing import Optional

import mlflow
import yaml
from mlflow.tracking import MlflowClient

MODEL_NAME = "lexrag-retriever"
CHAMPION_ALIAS = "champion"
MIN_COMPLETION_FRACTION = 1.0

class RetrieverConfigModel(mlflow.pyfunc.PythonModel):
  """
  A minimal Mlflow PythonModel wrapper around a retriever config.
  Doesn't load the actual index - it's a versioned record of WHICH configuration produced the best results.
  """

  def __init__(self,config:dict):
    self.config = config
  
  def predict(self,context,model_input):
    import pandas as pd
    return pd.DataFrame([self.config])
  

def _is_run_complete(run) -> bool:
  """
  Check whether a run's RAGAS evaluation completed on all eval
  questions, using completeness metrics logged by run_ragas_evaluation.
  Excludes runs with no completeness metrics logged (older runs)
  rather than trusting them blindly.
  """
  metrics = run.data.metrics
  completeness_keys = [k for k in metrics if k.endswith("_completeness")]

  if not completeness_keys:
    print(
      f" Warning: run '{run.data.tags.get('mlflow.runName', run.info.run_id)}'"
      f"has no completeness metrics logged - excluding from "
      f"champion consideration to be safe."
    )
    return False
  
  return all(metrics[k]>=MIN_COMPLETION_FRACTION for k in completeness_keys)


def get_best_run(experiment_name: str = "lexrag_experiments") -> Optional[dict]:
  """
  Query Mlflow for the run with the highest RAGAS score,
  considering ONLY runs with complete evaluation data.
  """
  client = MlflowClient()
  experiment = client.get_experiment_by_name(experiment_name)

  if experiment is None:
    print(f"No experiment found named'{experiment_name}'")
    return None
  
  all_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.composite DESC"],
    max_results=50,
  )

  if not all_runs:
    print("No runs found in this experiment yet.")
    return None
  
  complete_runs = [r for r in all_runs if _is_run_complete(r)]

  if not complete_runs:
    print(
      "No runs with complete evaluation data found. "
      f"{len(all_runs)} run(s) exist but all are partial or missing "
      "completeness tracking. Re-run evaluation to completion before "
      "registering a champion config."      
    )
    return None
  
  best = complete_runs[0]
  excluded_count = len(all_runs) - len(complete_runs)
  if excluded_count>0:
    print(f"Excluded {excluded_count} run(s) with incomplete data from consideration.")
  
  return {
    "run_id": best.info.run_id,
    "run_name": best.data.tags.get("mlflow.runName","unnamed"),
    "params": best.data.params,
    "metrics": best.data.metrics,
  }

def register_best_config(experiment_name: str = "lexrag_experiments") -> Optional[str]:
  """
  Find the best run by composite score (complete-data runs only),
  register its config in the MLflow Model Registry as a new version,
  and point the "champion" alias at it.
  """
  best_run = get_best_run(experiment_name)
  if best_run is None:
    return None
  
  print(f"Best eligible run: {best_run['run_name']} (composite={best_run['metrics'].get('composite','N/A')})")

  config = {
        "chunk_size": best_run["params"].get("chunk_size"),
        "chunk_overlap": best_run["params"].get("chunk_overlap"),
        "retrieval_mode": best_run["params"].get("retrieval_mode"),
        "retrieval_k": best_run["params"].get("retrieval_k"),
        "top_k": best_run["params"].get("top_k"),
        "rrf_k": best_run["params"].get("rrf_k"),
        "llm_model": best_run["params"].get("llm_model"),
        "llm_provider": best_run["params"].get("llm_provider"),
    }
  
  with tempfile.NamedTemporaryFile(mode="w",suffix=".yaml",delete=False) as f:
    yaml.dump(config,f)
    config_path=f.name
  
  with mlflow.start_run(run_id=best_run["run_id"]):
    mlflow.pyfunc.log_model(
      artifact_path="retriever_config",
      python_model=RetrieverConfigModel(config),
      artifacts={"config":config_path},
      registered_model_name=MODEL_NAME,
      pip_requirements=["pyyaml","pandas"],
    )

  client = MlflowClient()
  versions = client.search_model_versions(f"name='{MODEL_NAME}'")
  latest_version = max(int(v.version) for v in versions)

  client.set_model_version_tag(
    MODEL_NAME, str(latest_version),"composite_score",
    str(best_run["metrics"].get("composite","N/A"))
  )
  client.set_model_version_tag(
    MODEL_NAME, str(latest_version), "run_name", best_run["run_name"]
  )

  client.set_registered_model_alias(
    name=MODEL_NAME,
    alias=CHAMPION_ALIAS,
    version=str(latest_version),
  )

  print(f"Registered '{MODEL_NAME}' version {latest_version} and set as @{CHAMPION_ALIAS}")
  return str(latest_version)

def load_champion_config() -> Optional[dict]:
  """
  Load the current champion config via its alias - this is how serving code fetches "whichever config is currently best" without
  hardcoding a version number.
  """
  try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    return model.unwrap_python_model().config
  except Exception as e:
    print(f"Could not load champion config: {e}")
    return None
  
if __name__ == "__main__":
  register_best_config()
