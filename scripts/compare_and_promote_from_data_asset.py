import os, json, time, glob
import numpy as np, pandas as pd, requests
from sklearn.metrics import mean_squared_error, r2_score

# v2 SDK (엔드포인트 조작용)
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential, DefaultAzureCredential

# v1 SDK (Legacy Dataset 다운로드용)
from azureml.core import Workspace, Dataset


# ---------- 환경변수 ----------
RG  = os.getenv("AZ_RESOURCE_GROUP")
WS  = os.getenv("AZ_ML_WORKSPACE")
SUB_ID = os.getenv("SUBSCRIPTION_ID")

DATA_ASSET_NAME    = os.getenv("DATA_ASSET_NAME")
DATA_ASSET_VERSION = os.getenv("DATA_ASSET_VERSION")
LABEL_COL          = os.getenv("LABEL_COL")
FEATURE_COLS       = os.getenv("FEATURE_COLS")

ENDPOINT_NAME      = os.getenv("ENDPOINT_NAME")
ENDPOINT_URL       = os.getenv("ENDPOINT_URL")
API_KEY            = os.getenv("API_KEY")
DEPLOYMENT_A       = os.getenv("DEPLOYMENT_A")   # Champion
DEPLOYMENT_B       = os.getenv("DEPLOYMENT_B")   # Challenger
PROMOTE_RULE       = os.getenv("PROMOTE_RULE", "rmse<=0.98 and r2>=-0.01")


# ---------- 공통 ----------
def _need(name: str):
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env: {name}")
    return v

def get_ml_client() -> MLClient:
    _need("SUBSCRIPTION_ID"); _need("AZ_RESOURCE_GROUP"); _need("AZ_ML_WORKSPACE")
    last = None
    for cred in (
        AzureCliCredential(),  
        DefaultAzureCredential(exclude_interactive_browser_credential=True),
    ):
        try:
            return MLClient(cred, SUB_ID, RG, WS)
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to create MLClient: {last}")


# ---------- 데이터 에셋 다운로드 ----------
def download_data_asset(name: str, version: str, out_dir: str = "./_aml_data") -> str:
    """
    test_csv 은 'Dataset type (from Azure ML v1 APIs)' 이므로 v1 SDK 필요.
    """
    if not name or not version:
        raise RuntimeError("DATA_ASSET_NAME and DATA_ASSET_VERSION are required")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] v1 SDK download {name}:{version} -> {out_dir}")

    ws = Workspace(subscription_id=SUB_ID, resource_group=RG, workspace_name=WS)

    ds = Dataset.get_by_name(workspace=ws, name=name, version=version)
    ds.download(target_path=out_dir, overwrite=True)

    # csv 기준으로 선택
    csvs = glob.glob(os.path.join(out_dir, "**", "*.csv"), recursive=True)
    if csvs:
        print(f"[INFO] Using CSV file: {csvs[0]}")
        return csvs[0]
    raise FileNotFoundError(f"No csv found under {out_dir}")


# ---------- 데이터 적재 ----------
def load_dataset(path: str, label_col: str, feature_cols_csv: str | None):
    df = pd.read_csv(path)   # ✅ CSV 기준

    if label_col not in df.columns:
        raise RuntimeError(f"Label '{label_col}' not in dataset columns: {list(df.columns)}")

    if feature_cols_csv:
        feats = [c.strip() for c in feature_cols_csv.split(",") if c.strip()]
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise RuntimeError(f"FEATURE_COLS not found in dataset: {missing}")
    else:
        feats = [c for c in df.columns if c != label_col]

    try:
        X = df[feats].apply(pd.to_numeric, errors="raise")
        y = pd.to_numeric(df[label_col], errors="raise").values
    except Exception as e:
        raise RuntimeError(f"Non-numeric in features/label. Error: {e}")

    if len(X) == 0:
        raise RuntimeError("Dataset is empty")

    return X, y, feats


# ---------- 엔드포인트 호출 ----------
def build_payload(Xb: pd.DataFrame, cols: list[str]) -> dict:
    return {"input_data": {"columns": list(cols), "data": Xb.values.tolist()}}

def _parse_pred(res_json):
    if isinstance(res_json, dict):
        for k in ("output", "predictions", "result", "scores"):
            if k in res_json:
                return np.array(res_json[k], dtype=float).reshape(-1)
    return np.array(res_json, dtype=float).reshape(-1)

def call_endpoint(endpoint_url: str, api_key: str, deployment: str, payload: dict, timeout: int = 60) -> np.ndarray:
    _need("ENDPOINT_URL"); _need("API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "azureml-model-deployment": deployment,
    }
    resp = requests.post(endpoint_url, headers=headers, data=json.dumps(payload), timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Endpoint call failed ({deployment}): HTTP {resp.status_code} - {resp.text[:500]}") from e
    return _parse_pred(resp.json())

def batched_predict(X: pd.DataFrame, cols: list[str], infer_fn, bs: int = 256) -> np.ndarray:
    out = np.empty(len(X), dtype=float)
    i = 0
    while i < len(X):
        j = min(i + bs, len(X))
        out[i:j] = infer_fn(build_payload(X.iloc[i:j], cols))
        i = j
    return out


# ---------- 평가/승격 ----------
def metrics(y, yhat):
    return {"rmse": mean_squared_error(y, yhat, squared=False),
            "r2":   r2_score(y, yhat)}

def decide(champion: dict, challenger: dict, rule: str) -> bool:
    rmse_factor, r2_delta = 1.0, -float("inf")
    for tok in rule.replace(" ", "").split("and"):
        if tok.startswith("rmse<="):
            rmse_factor = float(tok.split("<=")[1])
        elif tok.startswith("r2>="):
            r2_delta = float(tok.split(">=")[1])
    ok_rmse = challenger["rmse"] <= champion["rmse"] * rmse_factor
    ok_r2   = challenger["r2"]   >= champion["r2"] + r2_delta
    return ok_rmse and ok_r2


# ---------- 트래픽 전환 ----------
def switch_traffic_sdk(mlc: MLClient, endpoint: str, a: str, a_pct: int, b: str, b_pct: int):
    print(f"[INFO] Switching traffic: {a}={a_pct}, {b}={b_pct}")
    e = mlc.online_endpoints.get(name=endpoint)
    e.traffic = {a: a_pct, b: b_pct}
    mlc.online_endpoints.begin_create_or_update(e).result()

def delete_deployment_sdk(mlc: MLClient, endpoint: str, deployment: str):
    print(f"[INFO] Deleting deployment: {deployment}")
    mlc.online_deployments.begin_delete(name=deployment, endpoint_name=endpoint).result()


# ---------- 메인 ----------
def main():
    for env in [
        "AZ_RESOURCE_GROUP","AZ_ML_WORKSPACE","SUBSCRIPTION_ID",
        "DATA_ASSET_NAME","DATA_ASSET_VERSION",
        "LABEL_COL","ENDPOINT_NAME","ENDPOINT_URL","API_KEY",
        "DEPLOYMENT_A","DEPLOYMENT_B"
    ]:
        _need(env)

    mlc = get_ml_client()

    # 1) 데이터 다운로드
    data_path = download_data_asset(DATA_ASSET_NAME, DATA_ASSET_VERSION)
    X, y, feats = load_dataset(data_path, LABEL_COL, FEATURE_COLS)
    print(f"[INFO] Rows={len(X)}, Features={len(feats)} -> {feats[:8]}{'...' if len(feats)>8 else ''}")

    # 2) 추론
    infer_A = lambda p: call_endpoint(ENDPOINT_URL, API_KEY, DEPLOYMENT_A, p)
    infer_B = lambda p: call_endpoint(ENDPOINT_URL, API_KEY, DEPLOYMENT_B, p)
    yhat_A = batched_predict(X, feats, infer_A)
    yhat_B = batched_predict(X, feats, infer_B)

    # 3) 메트릭
    mA = metrics(y, yhat_A); mB = metrics(y, yhat_B)
    print(f"[METRIC] {DEPLOYMENT_A}: RMSE={mA['rmse']:.6f}, R2={mA['r2']:.6f}")
    print(f"[METRIC] {DEPLOYMENT_B}: RMSE={mB['rmse']:.6f}, R2={mB['r2']:.6f}")

    # 4) 승격
    if decide(mA, mB, PROMOTE_RULE):
        print(f"[INFO] Challenger passes rule ({PROMOTE_RULE}). Progressive shift...")
        for a_pct, b_pct in [(90,10), (50,50), (0,100)]:
            switch_traffic_sdk(mlc, ENDPOINT_NAME, DEPLOYMENT_A, a_pct, DEPLOYMENT_B, b_pct)
            time.sleep(3)
        print("[INFO] Challenger promoted. Removing old champion...")
        delete_deployment_sdk(mlc, ENDPOINT_NAME, DEPLOYMENT_A)
    else:
        print(f"[INFO] Challenger rejected by rule ({PROMOTE_RULE}). No changes.")

if __name__ == "__main__":
    main()
