import os, json, math, time, subprocess, glob, sys
import numpy as np, pandas as pd, requests
from sklearn.metrics import mean_squared_error, r2_score

# ✅ 추가: SDK import
from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient

# 환경변수 세팅 (GitHub Actions에서 주입)
RG  = os.getenv("AZ_RESOURCE_GROUP")
WS  = os.getenv("AZ_ML_WORKSPACE")
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

def run_az(cmd:list):
    r = subprocess.run(["az",*cmd], check=True, text=True, capture_output=True)
    return r.stdout

# ✅ 교체: CLI → SDK 방식으로 데이터 에셋 다운로드
def download_data_asset(name, version, out_dir="./_aml_data"):
    """
    GitHub Actions에서 azure/login으로 이미 로그인된 세션을 SDK가 사용하도록 AzureCliCredential() 사용.
    워크플로에서 SUBSCRIPTION_ID를 $GITHUB_ENV에 넣어 주세요:
      echo "SUBSCRIPTION_ID=$(az account show --query id -o tsv)" >> $GITHUB_ENV
    """
    os.makedirs(out_dir, exist_ok=True)

    sub_id = os.getenv("SUBSCRIPTION_ID")  # 워크플로에서 주입
    rg = RG
    ws = WS
    assert sub_id and rg and ws, "SUBSCRIPTION_ID / AZ_RESOURCE_GROUP / AZ_ML_WORKSPACE 필요"

    mlc = MLClient(AzureCliCredential(), sub_id, rg, ws)
    print(f"[INFO] SDK download {name}:{version} -> {out_dir}")
    mlc.data.download(name=name, version=version, download_path=out_dir)

    # 다운로드된 파일에서 csv/parquet 하나 선택
    csvs = glob.glob(os.path.join(out_dir, "**", "*.csv"), recursive=True)
    pars = glob.glob(os.path.join(out_dir, "**", "*.parquet"), recursive=True)
    if csvs: return csvs[0]
    if pars: return pars[0]
    raise FileNotFoundError("csv/parquet 파일 없음")

def load_dataset(path, label_col, feature_cols_csv=None):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    feats = [c for c in df.columns if c != label_col] if not feature_cols_csv else [c.strip() for c in feature_cols_csv.split(",")]
    X = df[feats].astype(float)
    y = df[label_col].astype(float).values
    return X, y, feats

def build_payload(Xb, cols):
    return {"input_data":{"columns":list(cols),"data":Xb.astype(float).values.tolist()}}

def call_endpoint(endpoint_url, api_key, deployment, payload, timeout=60):
    h = {"Content-Type":"application/json","Authorization":f"Bearer {api_key}","azureml-model-deployment":deployment}
    r = requests.post(endpoint_url, headers=h, data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()
    res = r.json()
    if isinstance(res, dict):
        for k in ("output","predictions","result"):
            if k in res: return np.array(res[k], dtype=float)
    return np.array(res, dtype=float)

def batched_predict(X, cols, infer_fn, bs=256):
    out = np.zeros(len(X), float)
    for i in range(0, len(X), bs):
        Xb = X.iloc[i:i+bs]; payload = build_payload(Xb, cols)
        out[i:i+len(Xb)] = infer_fn(payload)
    return out

def metric(y, yhat):
    return {"rmse":mean_squared_error(y,yhat,squared=False),
            "r2":r2_score(y,yhat)}

def decide(champ, chall, rule="rmse<=0.98 and r2>=-0.01"):
    rmse_factor, r2_delta = 1.0, -float("inf")
    for tok in rule.replace(" ","").split("and"):
        if tok.startswith("rmse<="): rmse_factor=float(tok.split("<=")[1])
        if tok.startswith("r2>="):   r2_delta=float(tok.split(">=")[1])
    ok_rmse = chall["rmse"] <= champ["rmse"]*rmse_factor
    ok_r2   = chall["r2"]   >= champ["r2"] + r2_delta
    return ok_rmse and ok_r2

def switch_traffic(endpoint, a, a_pct, b, b_pct):
    split = f"{a}={a_pct} {b}={b_pct}"
    run_az(["ml","online-endpoint","update","--name",endpoint,"--traffic",split])

def delete_deployment(endpoint, deployment):
    run_az(["ml","online-deployment","delete","--name",deployment,"--endpoint-name",endpoint,"--yes"])

def main():
    data_path = download_data_asset(DATA_ASSET_NAME, DATA_ASSET_VERSION)
    X, y, feats = load_dataset(data_path, LABEL_COL, FEATURE_COLS)

    infer_A = lambda p: call_endpoint(ENDPOINT_URL, API_KEY, DEPLOYMENT_A, p)
    infer_B = lambda p: call_endpoint(ENDPOINT_URL, API_KEY, DEPLOYMENT_B, p)

    mA = metric(y, batched_predict(X, feats, infer_A))
    mB = metric(y, batched_predict(X, feats, infer_B))

    print(f"[{DEPLOYMENT_A}] RMSE={mA['rmse']:.4f}, R2={mA['r2']:.4f}")
    print(f"[{DEPLOYMENT_B}] RMSE={mB['rmse']:.4f}, R2={mB['r2']:.4f}")

    if decide(mA, mB, PROMOTE_RULE):
        for a,b in [(90,10),(50,50),(0,100)]:
            switch_traffic(ENDPOINT_NAME, DEPLOYMENT_A, a, DEPLOYMENT_B, b)
            time.sleep(3)
        print("Challenger promoted.")
        delete_deployment(ENDPOINT_NAME, DEPLOYMENT_A)   # ✅ 기존 Champion 삭제
    else:
        print("Challenger rejected.")

if __name__ == "__main__":
    main()
