# <scripts/compare_and_promote_from_data_asset.py>
import os, json, time, math
import numpy as np, pandas as pd, requests
from sklearn.metrics import mean_squared_error, r2_score

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential, DefaultAzureCredential

# ---------- ENV ----------
RG       = os.getenv("AZ_RESOURCE_GROUP")
WS       = os.getenv("AZ_ML_WORKSPACE")
SUB_ID   = os.getenv("SUBSCRIPTION_ID")

DATA_CSV_URL = os.getenv("DATA_CSV_URL")
LABEL_COL    = os.getenv("LABEL_COL")
FEATURE_COLS = os.getenv("FEATURE_COLS")

ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
ENDPOINT_URL  = os.getenv("ENDPOINT_URL")
API_KEY       = os.getenv("API_KEY")
DEPLOYMENT_A  = os.getenv("DEPLOYMENT_A")
DEPLOYMENT_B  = os.getenv("DEPLOYMENT_B")
PROMOTE_RULE  = os.getenv("PROMOTE_RULE", "rmse<=0.98 and r2>=-0.01")


def _need(name: str):
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env: {name}")
    return v


def get_ml_client() -> MLClient:
    _need("SUBSCRIPTION_ID"); _need("AZ_RESOURCE_GROUP"); _need("AZ_ML_WORKSPACE")
    last = None
    for cred in (AzureCliCredential(), DefaultAzureCredential(exclude_interactive_browser_credential=True)):
        try:
            return MLClient(cred, SUB_ID, RG, WS)
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to create MLClient: {last}")


# ---------- DATA LOADING & SANITIZATION ----------
EXPECTED = [c.strip() for c in (FEATURE_COLS or "").split(",") if c.strip()]
if not EXPECTED:
    raise RuntimeError("FEATURE_COLS must be provided (comma-separated)")


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # to numeric float, replace NaN/Inf
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            df[c] = s
        else:
            df[c] = 0.0
    return df


def _align_schema(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    # drop extras, enforce order and fill missing with 0.0
    df = df.reindex(columns=expected, fill_value=0.0)
    return _coerce_numeric(df, expected)


def load_csv_from_url(url: str, label_col: str, feature_cols_csv: str | None):
    df = pd.read_csv(url)

    if label_col not in df.columns:
        raise RuntimeError(f"Label '{label_col}' not in dataset columns: {list(df.columns)}")

    feats = [c.strip() for c in feature_cols_csv.split(",")] if feature_cols_csv else [c for c in df.columns if c != label_col]

    # label cleanup
    y = pd.to_numeric(df[label_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = y.notna()
    if (~keep).any():
        print(f"[WARN] Dropping {(~keep).sum()} rows with non-numeric/NaN label '{label_col}'")
    df = df.loc[keep].reset_index(drop=True)
    y  = y.loc[keep].astype(float).values

    X = _align_schema(df[feats].copy(), EXPECTED)
    if len(X) == 0:
        raise RuntimeError("Dataset is empty after label cleaning")
    return X, y, EXPECTED


# ---------- PAYLOAD BUILDERS ----------

def build_payload_columns(Xb: pd.DataFrame, cols: list[str]):
    return {"input_data": {"columns": list(cols), "data": Xb.values.tolist()}}


def build_payload_records(Xb: pd.DataFrame, _cols: list[str] | None = None):
    return {"data": Xb.where(pd.notna(Xb), None).to_dict(orient="records")}


def build_payload_designer_raw(Xb: pd.DataFrame, _cols: list[str] | None = None):
    return Xb.where(pd.notna(Xb), None).to_dict(orient="records")


def payload_builder_by_name(name: str):
    name = (name or "").lower()
    if name == "columns":
        return build_payload_columns
    if name == "records":
        return build_payload_records
    if name == "designer_raw":
        return build_payload_designer_raw
    if name == "adaptive":
        # try in order: designer_raw → columns → records
        def _adaptive(df, cols):
            last_err = None
            for fn in (build_payload_designer_raw, build_payload_columns, build_payload_records):
                try:
                    p = fn(df.head(2), cols)
                    _smoke_call(ENDPOINT_URL, API_KEY, DEPLOYMENT_A, p)  # fast check against champion
                    return fn(df, cols)
                except Exception as e:
                    last_err = e
            raise RuntimeError(f"All payload modes failed during smoke test: {last_err}")
        return _adaptive
    # default
    return build_payload_designer_raw


# ---------- ENDPOINT CALL ----------

def _parse_pred(res_json):
    # prefer 'output' then common aliases
    if isinstance(res_json, dict):
        for k in ("output", "outputs", "prediction", "predictions", "result", "scores"):
            if k in res_json:
                arr = np.array(res_json[k])
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr.ravel()
                return arr.astype(float).reshape(-1)
    arr = np.array(res_json)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    return arr.astype(float).reshape(-1)


def call_endpoint(endpoint_url: str, api_key: str, deployment: str, payload, timeout: int = 60) -> np.ndarray:
    _need("ENDPOINT_URL"); _need("API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "azureml-model-deployment": deployment,
    }
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Endpoint call failed ({deployment}): HTTP {resp.status_code} - {resp.text[:500]}") from e
    return _parse_pred(resp.json())


def _smoke_call(endpoint_url: str, api_key: str, deployment: str, payload) -> None:
    # small request to validate payload format quickly
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "azureml-model-deployment": deployment,
    }
    r = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Smoke test failed: HTTP {r.status_code} - {r.text[:300]}")


def batched_predict(X: pd.DataFrame, cols: list[str], infer_fn, payload_builder, bs: int = 256) -> np.ndarray:
    out = np.empty(len(X), dtype=float)
    i = 0
    while i < len(X):
        j = min(i + bs, len(X))
        payload = payload_builder(X.iloc[i:j], cols)
        out[i:j] = infer_fn(payload)
        i = j
    return out


# ---------- METRICS & DECISION ----------

def metrics(y, yhat):
    return {"rmse": mean_squared_error(y, yhat, squared=False), "r2": r2_score(y, yhat)}


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


# ---------- TRAFFIC SWITCH ----------

def switch_traffic_sdk(mlc: MLClient, endpoint: str, a: str, a_pct: int, b: str, b_pct: int):
    print(f"[INFO] Switching traffic: {a}={a_pct}, {b}={b_pct}")
    e = mlc.online_endpoints.get(name=endpoint)
    e.traffic = {a: a_pct, b: b_pct}
    mlc.online_endpoints.begin_create_or_update(e).result()


def delete_deployment_sdk(mlc: MLClient, endpoint: str, deployment: str):
    print(f"[INFO] Deleting deployment: {deployment}")
    mlc.online_deployments.begin_delete(name=deployment, endpoint_name=endpoint).result()


# ---------- MAIN ----------

def main():
    for env in [
        "AZ_RESOURCE_GROUP","AZ_ML_WORKSPACE","SUBSCRIPTION_ID",
        "LABEL_COL","ENDPOINT_NAME","ENDPOINT_URL","API_KEY",
        "DEPLOYMENT_A","DEPLOYMENT_B","DATA_CSV_URL"
    ]:
        _need(env)

    mlc = get_ml_client()

    # 1) Data
    X, y, feats = load_csv_from_url(DATA_CSV_URL, LABEL_COL, FEATURE_COLS)
    print(f"[INFO] Rows={len(X)}, Features={len(feats)} -> {feats[:8]}{'...' if len(feats)>8 else ''}")

    # 2) Payload mode (with adaptive smoke test if requested)
    payload_mode = os.getenv("PAYLOAD_MODE", "designer_raw").lower()
    pbuilder = payload_builder_by_name(payload_mode)

    infer_A = lambda p: call_endpoint(ENDPOINT_URL, API_KEY, DEPLOYMENT_A, p)
    infer_B = lambda p: call_endpoint(ENDPOINT_URL, API_KEY, DEPLOYMENT_B, p)

    yhat_A = batched_predict(X, feats, infer_A, pbuilder)
    yhat_B = batched_predict(X, feats, infer_B, pbuilder)

    # 3) Metrics
    mA = metrics(y, yhat_A); mB = metrics(y, yhat_B)
    print(f"[METRIC] {DEPLOYMENT_A}: RMSE={mA['rmse']:.6f}, R2={mA['r2']:.6f}")
    print(f"[METRIC] {DEPLOYMENT_B}: RMSE={mB['rmse']:.6f}, R2={mB['r2']:.6f}")

    # 4) Promote decision
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
