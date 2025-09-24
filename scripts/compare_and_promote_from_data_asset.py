import os, json, time, subprocess, glob
import numpy as np, pandas as pd, requests
from sklearn.metrics import mean_squared_error, r2_score

# ===== Azure SDK (데이터 다운로드용) =====
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential, DefaultAzureCredential

# --------- 환경 변수 ---------
RG  = os.getenv("AZ_RESOURCE_GROUP")
WS  = os.getenv("AZ_ML_WORKSPACE")
SUB_ID = os.getenv("SUBSCRIPTION_ID")  # 워크플로 단계에서 $GITHUB_ENV로 주입

DATA_ASSET_NAME    = os.getenv("DATA_ASSET_NAME")
DATA_ASSET_VERSION = os.getenv("DATA_ASSET_VERSION")
LABEL_COL          = os.getenv("LABEL_COL")
FEATURE_COLS       = os.getenv("FEATURE_COLS")  # "col1,col2,..." or None

ENDPOINT_NAME      = os.getenv("ENDPOINT_NAME")
ENDPOINT_URL       = os.getenv("ENDPOINT_URL")
API_KEY            = os.getenv("API_KEY")
DEPLOYMENT_A       = os.getenv("DEPLOYMENT_A")   # Champion
DEPLOYMENT_B       = os.getenv("DEPLOYMENT_B")   # Challenger
PROMOTE_RULE       = os.getenv("PROMOTE_RULE", "rmse<=0.98 and r2>=-0.01")


# --------- 공통 유틸 ---------
def assert_env(var_name: str):
    v = os.getenv(var_name)
    if not v:
        raise RuntimeError(f"Missing required env: {var_name}")
    return v

def run_az(cmd:list) -> str:
    """ Azure CLI 호출 (ml 확장 최신화 전제) """
    res = subprocess.run(["az", *cmd], text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"az {' '.join(cmd)} failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res.stdout

def get_ml_client() -> MLClient:
    """
    1순위: AzureCliCredential (azure/login으로 CLI 로그인 되어 있음)
    2순위: DefaultAzureCredential (환경 변수/워크플로 토큰 등)
    """
    assert_env("SUBSCRIPTION_ID")
    assert_env("AZ_RESOURCE_GROUP")
    assert_env("AZ_ML_WORKSPACE")

    cred_chain = []
    try:
        cred_chain.append(AzureCliCredential())
    except Exception:
        pass
    cred_chain.append(DefaultAzureCredential(exclude_interactive_browser_credential=True))

    last_err = None
    for cred in cred_chain:
        try:
            return MLClient(cred, SUB_ID, RG, WS)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to create MLClient. Last error: {last_err}")

# --------- 데이터 에셋 다운로드 (SDK) ---------
def download_data_asset(name: str, version: str, out_dir: str = "./_aml_data") -> str:
    if not name or not version:
        raise RuntimeError("DATA_ASSET_NAME and DATA_ASSET_VERSION are required")

    os.makedirs(out_dir, exist_ok=True)
    mlc = get_ml_client()
    print(f"[INFO] Downloading data asset via SDK: {name}:{version} -> {out_dir}")
    mlc.data.download(name=name, version=version, download_path=out_dir)

    # 가장 먼저 발견되는 csv 또는 parquet 반환
    csvs = glob.glob(os.path.join(out_dir, "**", "*.csv"), recursive=True)
    pars = glob.glob(os.path.join(out_dir, "**", "*.parquet"), recursive=True)
    if csvs:
        print(f"[INFO] Using dataset file: {csvs[0]}")
        return csvs[0]
    if pars:
        print(f"[INFO] Using dataset file: {pars[0]}")
        return pars[0]
    raise FileNotFoundError(f"No csv/parquet found in {out_dir}")

# --------- 데이터 적재 ---------
def load_dataset(path: str, label_col: str, feature_cols_csv: str | None):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if label_col not in df.columns:
        raise RuntimeError(f"Label column '{label_col}' not in dataset columns: {list(df.columns)}")

    # 피처 목록
    if feature_cols_csv:
        feats = [c.strip() for c in feature_cols_csv.split(",") if c.strip()]
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise RuntimeError(f"FEATURE_COLS columns not found in dataset: {missing}")
    else:
        feats = [c for c in df.columns if c != label_col]

    # 수치형 변환 (불가한 값은 오류 원인이므로 명시적으로 raise)
    try:
        X = df[feats].apply(pd.to_numeric, errors="raise")
        y = pd.to_numeric(df[label_col], errors="raise").values
    except Exception as e:
        raise RuntimeError(f"Non-numeric data found in features/label. "
                           f"Set FEATURE_COLS or clean data. Error: {e}")

    if len(X) == 0:
        raise RuntimeError("Dataset is empty after loading")

    return X, y, feats

# --------- 엔드포인트 호출 ---------
def build_payload(Xb: pd.DataFrame, cols: list[str]) -> dict:
    return {"input_data": {"columns": list(cols), "data": Xb.values.tolist()}}

def parse_pred_response(res_json):
    # 점수 스키마 다양성 대응
    if isinstance(res_json, dict):
        for k in ("output", "predictions", "result", "scores"):
            if k in res_json:
                return np.array(res_json[k], dtype=float).reshape(-1)
    # 리스트/배열
    return np.array(res_json, dtype=float).reshape(-1)

def call_endpoint(endpoint_url: str, api_key: str, deployment: str, payload: dict, timeout: int = 60) -> np.ndarray:
    if not endpoint_url or not api_key or not deployment:
        raise RuntimeError("ENDPOINT_URL, API_KEY, DEPLOYMENT name are required")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "azureml-model-deployment": deployment,
    }
    resp = requests.post(endpoint_url, headers=headers, data=json.dumps(payload), timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Endpoint call failed for deployment '{deployment}': "
                           f"HTTP {resp.status_code} - {resp.text[:500]}") from e
    return parse_pred_response(resp.json())

def batched_predict(X: pd.DataFrame, cols: list[str], infer_fn, bs: int = 256) -> np.ndarray:
    out = np.empty(len(X), dtype=float)
    s = 0
    while s < len(X):
        e = min(s + bs, len(X))
        payload = build_payload(X.iloc[s:e], cols)
        out[s:e] = infer_fn(payload)
        s = e
    return out

# --------- 메트릭/의사결정 ---------
def metrics(y, yhat):
    return {"rmse": mean_squared_error(y, yhat, squared=False),
            "r2":   r2_score(y, yhat)}

def decide_migration(champion: dict, challenger: dict, rule: str) -> bool:
    """ rule 예: 'rmse<=0.98 and r2>=-0.01' """
    rmse_factor = 1.0
    r2_delta    = -float("inf")
    for tok in rule.replace(" ", "").split("and"):
        if tok.startswith("rmse<="):
            rmse_factor = float(tok.split("<=")[1])
        elif tok.startswith("r2>="):
            r2_delta = float(tok.split(">=")[1])

    ok_rmse = challenger["rmse"] <= champion["rmse"] * rmse_factor
    ok_r2   = challenger["r2"]   >= champion["r2"] + r2_delta
    return ok_rmse and ok_r2

# --------- 트래픽 전환/삭제 (CLI) ---------
def switch_traffic(endpoint: str, a: str, a_pct: int, b: str, b_pct: int):
    split = f"{a}={a_pct} {b}={b_pct}"
    print(f"[INFO] Switching traffic: {split}")
    run_az(["ml", "online-endpoint", "update", "--name", endpoint, "--traffic", split])

def delete_deployment(endpoint: str, deployment: str):
    print(f"[INFO] Deleting old deployment: {deployment}")
    run_az(["ml", "online-deployment", "delete", "--name", deployment, "--endpoint-name", endpoint, "--yes"])

# --------- 메인 ---------
def main():
    # 필수 환경 체크
    for v in ["AZ_RESOURCE_GROUP", "AZ_ML_WORKSPACE", "SUBSCRIPTION_ID",
              "DATA_ASSET_NAME", "DATA_ASSET_VERSION",
              "LABEL_COL", "ENDPOINT_NAME", "ENDPOINT_URL", "API_KEY",
              "DEPLOYMENT_A", "DEPLOYMENT_B"]:
        assert_env(v)

    # 1) 데이터 로드
    data_path = download_data_asset(DATA_ASSET_NAME, DATA_ASSET_VERSION)
    X, y, feats = load_dataset(data_path, LABEL_COL, FEATURE_COLS)
    print(f"[INFO] Using {len(X)} rows, {len(feats)} features: {feats[:8]}{'...' if len(feats)>8 else ''}")

    # 2) 예측 함수 래핑
    infer_A = lambda p: call_endpoint(ENDPOINT_URL, API_KEY, DEPLOYMENT_A, p)
    infer_B = lambda p: call_endpoint(ENDPOINT_URL, API_KEY, DEPLOYMENT_B, p)

    # 3) 배치 추론 & 평가
    yhat_A = batched_predict(X, feats, infer_A)
    yhat_B = batched_predict(X, feats, infer_B)
    mA = metrics(y, yhat_A); mB = metrics(y, yhat_B)
    print(f"[METRIC] {DEPLOYMENT_A}: RMSE={mA['rmse']:.6f}, R2={mA['r2']:.6f}")
    print(f"[METRIC] {DEPLOYMENT_B}: RMSE={mB['rmse']:.6f}, R2={mB['r2']:.6f}")

    # 4) 승격 여부 판단
    if decide_migration(mA, mB, PROMOTE_RULE):
        print(f"[INFO] Challenger passes rule ({PROMOTE_RULE}). Progressive traffic shift...")
        for a_pct, b_pct in [(90,10), (50,50), (0,100)]:
            switch_traffic(ENDPOINT_NAME, DEPLOYMENT_A, a_pct, DEPLOYMENT_B, b_pct)
            time.sleep(3)
        print("[INFO] Challenger promoted. Removing old champion...")
        delete_deployment(ENDPOINT_NAME, DEPLOYMENT_A)
    else:
        print(f"[INFO] Challenger rejected by rule ({PROMOTE_RULE}). No changes applied.")

if __name__ == "__main__":
    main()
