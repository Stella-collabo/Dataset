# -*- coding: utf-8 -*-
"""
================================================================================
 WIPO IPC-Technology Concordance 기반 기술 다각화(Technological Diversification)
 실증분석 프레임워크
--------------------------------------------------------------------------------
 - IV1_HW_Entropy / IV2_SW_Entropy : Jacquemin & Berry(1978) Shannon Entropy
 - 지뢰 ① 다중공선성   : 평균 중심화(Mean Centering) + VIF 진단
 - 지뢰 ② 내생성/이질성 : t+1 종속변수(ROA) + 기업/연도 고정효과(LSDV, HC3)
 - 산출물              : wipo_analysis_results.xlsx (기술통계/상관관계/VIF/회귀표)
--------------------------------------------------------------------------------
 실제 데이터 파일(2021.csv, 2023.csv, financial_database.csv)이 존재하면 이를
 사용하고, 없으면 연구 모형에 맞는 합성 패널 데이터(20개 기업 × 10개년)를
 자동 생성하여 전체 파이프라인이 끝까지 작동하도록 설계되어 있다.
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------------------------
# 0. 전역 설정
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATENT_FILES = [os.path.join(BASE_DIR, "2021.csv"),
                os.path.join(BASE_DIR, "2023.csv")]
FINANCIAL_FILE = os.path.join(BASE_DIR, "financial_database.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "wipo_analysis_results.xlsx")

RANDOM_SEED = 2026
N_FIRMS = 20                      # 합성 패널: 기업 수
PANEL_YEARS = list(range(2014, 2024))   # 합성 패널: 10개년 (2014~2023)

# --- [요구사항 1-1] WIPO IPC-Technology Concordance: 분석 대상 Subclass ---
WIPO_SW_IPC = {"G06F", "G06N", "G06Q", "G06K", "G06T", "G16H", "H04L"}
WIPO_HW_IPC = {"H01L", "H10K", "H10N", "H03K", "H03M", "H03F", "H03H"}
WIPO_VALID_IPC = WIPO_SW_IPC | WIPO_HW_IPC

# 통제변수 목록 (재무 데이터 기준)
CONTROL_VARS = ["Firm_Size", "Leverage", "Firm_Age"]


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ------------------------------------------------------------------------------
# 1. 데이터 로드 (실제 파일 우선, 없으면 합성 패널 자동 생성)
# ------------------------------------------------------------------------------
def generate_synthetic_patents(rng: np.random.Generator) -> pd.DataFrame:
    """20개 기업 × 10개년 특허 원시데이터 생성 (WIPO 대상 외 노이즈 IPC 포함)."""
    noise_ipc = ["A61K", "B60L", "C07D", "F16H", "E04B"]   # 필터링 검증용 노이즈
    sw_list, hw_list = sorted(WIPO_SW_IPC), sorted(WIPO_HW_IPC)

    records = []
    for i in range(N_FIRMS):
        firm = f"Firm_{i + 1:02d}"
        # 기업별 기술 성향: SW 지향 ~ HW 지향 스펙트럼
        sw_share = rng.uniform(0.2, 0.7)
        # 기업별 도메인 내 집중도(디리클레 농도) → 엔트로피의 횡단면 변동 확보
        sw_alpha = rng.uniform(0.3, 2.0)
        hw_alpha = rng.uniform(0.3, 2.0)
        sw_probs = rng.dirichlet([sw_alpha] * len(sw_list))
        hw_probs = rng.dirichlet([hw_alpha] * len(hw_list))

        for year in PANEL_YEARS:
            n_pat = int(rng.integers(40, 150))
            for _ in range(n_pat):
                u = rng.random()
                if u < 0.10:                       # 10%는 노이즈 특허
                    ipc = str(rng.choice(noise_ipc))
                elif u < 0.10 + 0.90 * sw_share:   # SW 특허
                    ipc = str(rng.choice(sw_list, p=sw_probs))
                else:                              # HW 특허
                    ipc = str(rng.choice(hw_list, p=hw_probs))
                records.append((firm, year, ipc))

    return pd.DataFrame(records, columns=["Firm", "Year", "IPC_Subclass"])


def generate_synthetic_financials(rng: np.random.Generator,
                                  patents: pd.DataFrame) -> pd.DataFrame:
    """특허 데이터와 정합적인 기업 재무 패널(ROA, R&D 집약도, 통제변수) 생성."""
    firms = sorted(patents["Firm"].unique())
    firm_effect = {f: rng.normal(0, 1.5) for f in firms}          # 기업 이질성
    year_effect = {y: rng.normal(0, 0.8) for y in PANEL_YEARS}    # 거시 충격
    firm_age0 = {f: int(rng.integers(5, 40)) for f in firms}
    firm_size_base = {f: rng.normal(12, 1.5) for f in firms}

    rows = []
    for f in firms:
        for y in PANEL_YEARS:
            rnd = max(rng.normal(4.0, 1.5), 0.3)                  # R&D 집약도(%)
            size = firm_size_base[f] + 0.05 * (y - PANEL_YEARS[0]) \
                + rng.normal(0, 0.2)
            lev = min(max(rng.normal(0.45, 0.12), 0.05), 0.95)
            roa = (3.0 + firm_effect[f] + year_effect[y]
                   + 0.3 * size - 4.0 * lev + 0.15 * rnd
                   + rng.normal(0, 1.2))
            rows.append((f, y, round(roa, 4), round(rnd, 4),
                         round(size, 4), round(lev, 4),
                         firm_age0[f] + (y - PANEL_YEARS[0])))

    return pd.DataFrame(rows, columns=["Firm", "Year", "ROA", "RnD_Intensity",
                                       "Firm_Size", "Leverage", "Firm_Age"])


def _find_column(df: pd.DataFrame, candidates) -> str | None:
    """열 이름 후보군과 대소문자 무시 매칭."""
    lookup = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def load_real_patents() -> pd.DataFrame:
    """실제 특허 CSV(2021.csv, 2023.csv)를 표준 스키마(Firm/Year/IPC)로 변환."""
    frames = []
    for path in PATENT_FILES:
        df = pd.read_csv(path, dtype=str)
        firm_col = _find_column(df, ["Firm", "기업명", "회사명", "출원인",
                                     "Company", "Applicant"])
        ipc_col = _find_column(df, ["IPC_Subclass", "IPC", "IPC코드",
                                    "ipc_code", "IPC_Code", "메인IPC"])
        year_col = _find_column(df, ["Year", "연도", "출원연도", "출원년도"])
        if firm_col is None or ipc_col is None:
            raise ValueError(f"{os.path.basename(path)}: 기업명/IPC 열을 "
                             f"찾을 수 없습니다. (열: {list(df.columns)})")

        out = pd.DataFrame({"Firm": df[firm_col].str.strip()})
        if year_col is not None:
            out["Year"] = pd.to_numeric(df[year_col].str[:4], errors="coerce")
        else:  # 연도 열이 없으면 파일명(2021.csv 등)에서 추론
            out["Year"] = int(os.path.basename(path)[:4])
        # 복수 IPC(';', '|' 구분) 대응 → 행 분리 후 첫 4자리 Subclass 추출
        out["IPC_Subclass"] = df[ipc_col].str.split(r"[;|,]")
        out = out.explode("IPC_Subclass")
        out["IPC_Subclass"] = (out["IPC_Subclass"].str.strip()
                               .str.upper().str[:4])
        frames.append(out.dropna(subset=["Firm", "Year", "IPC_Subclass"]))
    result = pd.concat(frames, ignore_index=True)
    result["Year"] = result["Year"].astype(int)
    return result


def load_real_financials() -> pd.DataFrame:
    """실제 재무 CSV를 표준 스키마로 변환."""
    df = pd.read_csv(FINANCIAL_FILE)
    mapping = {
        "Firm": ["Firm", "기업명", "회사명", "Company"],
        "Year": ["Year", "연도", "회계연도"],
        "ROA": ["ROA", "총자산이익률"],
        "RnD_Intensity": ["RnD_Intensity", "R&D_Intensity", "RND_Intensity",
                          "연구개발집약도", "RnD"],
        "Firm_Size": ["Firm_Size", "기업규모", "ln_Assets", "Size"],
        "Leverage": ["Leverage", "부채비율", "Lev"],
        "Firm_Age": ["Firm_Age", "기업연령", "업력", "Age"],
    }
    renamed = {}
    for std, cands in mapping.items():
        col = _find_column(df, cands)
        if col is None:
            raise ValueError(f"financial_database.csv에 '{std}' 열이 없습니다. "
                             f"(열: {list(df.columns)})")
        renamed[col] = std
    df = df.rename(columns=renamed)[list(mapping.keys())]
    df["Year"] = df["Year"].astype(int)
    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """실제 파일 존재 여부를 점검하고, 없으면 합성 패널로 대체."""
    real_ok = all(os.path.exists(p) for p in PATENT_FILES) \
        and os.path.exists(FINANCIAL_FILE)
    if real_ok:
        try:
            patents = load_real_patents()
            financials = load_real_financials()
            return patents, financials, "실제 데이터 (2021.csv / 2023.csv / financial_database.csv)"
        except Exception as exc:  # 스키마 불일치 등 → 합성 데이터로 안전 강하
            print(f"[경고] 실제 파일 로드 실패 → 합성 데이터로 대체합니다: {exc}")

    rng = np.random.default_rng(RANDOM_SEED)
    patents = generate_synthetic_patents(rng)
    financials = generate_synthetic_financials(rng, patents)
    return patents, financials, \
        f"합성 패널 데이터 (기업 {N_FIRMS}개 × {len(PANEL_YEARS)}개년, seed={RANDOM_SEED})"


# ------------------------------------------------------------------------------
# 2. [요구사항 1] WIPO 필터링 + Shannon Entropy 산출
# ------------------------------------------------------------------------------
def shannon_entropy(counts: pd.Series) -> float:
    """Jacquemin & Berry (1978): E = Σ p_i · ln(1/p_i)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float((p * np.log(1.0 / p)).sum())


def build_entropy_panel(patents: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_raw = len(patents)
    # [요구사항 1-1, 1-2] WIPO 대상 Subclass만 유지, 그 외 노이즈 기술 원천 차단
    patents = patents[patents["IPC_Subclass"].isin(WIPO_VALID_IPC)].copy()
    n_kept = len(patents)
    patents["Domain"] = np.where(patents["IPC_Subclass"].isin(WIPO_HW_IPC),
                                 "HW", "SW")

    # 기업-연도-도메인-Subclass 단위 특허 건수
    counts = (patents.groupby(["Firm", "Year", "Domain", "IPC_Subclass"])
              .size().rename("n").reset_index())

    # [요구사항 1-3] 기업-연도 단위 도메인별 Shannon Entropy
    ent = (counts.groupby(["Firm", "Year", "Domain"])["n"]
           .agg(shannon_entropy).rename("Entropy").reset_index())
    panel = (ent.pivot_table(index=["Firm", "Year"], columns="Domain",
                             values="Entropy", fill_value=0.0)
             .reindex(columns=["HW", "SW"], fill_value=0.0)
             .rename(columns={"HW": "IV1_HW_Entropy", "SW": "IV2_SW_Entropy"})
             .reset_index())
    panel.columns.name = None

    stats = {"원시 특허 건수": n_raw,
             "WIPO 필터 통과 건수": n_kept,
             "제외된 노이즈 건수": n_raw - n_kept,
             "필터 통과율(%)": round(100.0 * n_kept / n_raw, 2)}
    return panel, stats


# ------------------------------------------------------------------------------
# 3. 패널 병합 + [요구사항 3-1] t+1 종속변수 + [요구사항 2] 평균 중심화
# ------------------------------------------------------------------------------
def build_regression_panel(entropy_panel: pd.DataFrame,
                           financials: pd.DataFrame) -> pd.DataFrame:
    df = financials.merge(entropy_panel, on=["Firm", "Year"], how="left")
    df[["IV1_HW_Entropy", "IV2_SW_Entropy"]] = \
        df[["IV1_HW_Entropy", "IV2_SW_Entropy"]].fillna(0.0)

    # [요구사항 3-1] 내생성 완화: 종속변수 ROA는 t+1 시점 값을 매칭
    df = df.sort_values(["Firm", "Year"]).reset_index(drop=True)
    df["ROA_t1"] = df.groupby("Firm")["ROA"].shift(-1)
    df = df.dropna(subset=["ROA_t1"]).reset_index(drop=True)

    # [요구사항 2-1] 평균 중심화: 전체 표본 평균 차감 (상호작용항 생성 이전에 수행)
    for var in ["IV1_HW_Entropy", "IV2_SW_Entropy", "RnD_Intensity"]:
        df[f"c_{var}"] = df[var] - df[var].mean()

    # [요구사항 2-2] 중심화 변수의 곱으로 상호작용항 생성
    df["Int_HWxRnD"] = df["c_IV1_HW_Entropy"] * df["c_RnD_Intensity"]
    df["Int_SWxRnD"] = df["c_IV2_SW_Entropy"] * df["c_RnD_Intensity"]
    return df


# ------------------------------------------------------------------------------
# 4. [요구사항 2-3] VIF 진단
# ------------------------------------------------------------------------------
def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    vif_vars = CONTROL_VARS + ["c_IV1_HW_Entropy", "c_IV2_SW_Entropy",
                               "c_RnD_Intensity", "Int_HWxRnD", "Int_SWxRnD"]
    X = sm.add_constant(df[vif_vars].astype(float), has_constant="add")
    rows = [(var, variance_inflation_factor(X.values, i))
            for i, var in enumerate(X.columns) if var != "const"]
    vif_df = pd.DataFrame(rows, columns=["Variable", "VIF"])
    vif_df["판정"] = np.where(vif_df["VIF"] < 5, "양호(<5)",
                              np.where(vif_df["VIF"] < 10, "주의(5~10)",
                                       "심각(≥10)"))
    return vif_df


# ------------------------------------------------------------------------------
# 5. [요구사항 3] LSDV 고정효과 회귀 (기업 + 연도 더미, HC3)
# ------------------------------------------------------------------------------
def run_lsdv(df: pd.DataFrame, x_vars: list[str]):
    y = df["ROA_t1"].astype(float)
    X_core = df[x_vars].astype(float)

    # [요구사항 3-2] 기업/연도 고정효과 더미 — drop_first=True로 Dummy Trap 방지
    firm_dummies = pd.get_dummies(df["Firm"], prefix="Firm",
                                  drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(df["Year"].astype(int), prefix="Year",
                                  drop_first=True, dtype=float)

    X = pd.concat([X_core, firm_dummies, year_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")

    # [요구사항 3-3] 강건 표준오차 HC3
    return sm.OLS(y, X).fit(cov_type="HC3")


def significance_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def build_regression_table(models: dict, display_vars: list[str]) -> pd.DataFrame:
    """Model 1~3 비교표: 계수(표준오차)와 유의성 별표, 더미는 'Included'로 표기."""
    rows = []
    for var in ["const"] + display_vars:
        label = "상수항 (Constant)" if var == "const" else var
        row = {"Variable": label}
        for name, res in models.items():
            if var in res.params.index:
                b, se, p = res.params[var], res.bse[var], res.pvalues[var]
                row[name] = f"{b:.4f}{significance_stars(p)} ({se:.4f})"
            else:
                row[name] = ""
        rows.append(row)

    rows.append({"Variable": "기업 고정효과 (Firm FE)",
                 **{m: "Included" for m in models}})
    rows.append({"Variable": "연도 고정효과 (Year FE)",
                 **{m: "Included" for m in models}})
    rows.append({"Variable": "표준오차", **{m: "Robust (HC3)" for m in models}})
    rows.append({"Variable": "관측치 수 (N)",
                 **{m: f"{int(res.nobs)}" for m, res in models.items()}})
    rows.append({"Variable": "R-squared",
                 **{m: f"{res.rsquared:.4f}" for m, res in models.items()}})
    rows.append({"Variable": "Adj. R-squared",
                 **{m: f"{res.rsquared_adj:.4f}" for m, res in models.items()}})
    rows.append({"Variable": "F-statistic",
                 **{m: f"{res.fvalue:.2f}" for m, res in models.items()}})
    table = pd.DataFrame(rows)
    table["Note"] = ""
    table.loc[0, "Note"] = "*** p<0.01, ** p<0.05, * p<0.1 / 괄호 안은 HC3 표준오차"
    return table


# ------------------------------------------------------------------------------
# 6. [요구사항 4] 엑셀 리포트 저장
# ------------------------------------------------------------------------------
def export_to_excel(desc: pd.DataFrame, corr: pd.DataFrame,
                    vif_df: pd.DataFrame, reg_table: pd.DataFrame,
                    filter_stats: dict, source: str) -> None:
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        meta = pd.DataFrame({
            "항목": ["데이터 출처", "WIPO SW Subclass", "WIPO HW Subclass",
                     "원시 특허 건수", "WIPO 필터 통과 건수",
                     "제외된 노이즈 건수", "필터 통과율(%)",
                     "종속변수", "표준오차", "고정효과"],
            "내용": [source,
                     ", ".join(sorted(WIPO_SW_IPC)),
                     ", ".join(sorted(WIPO_HW_IPC)),
                     filter_stats["원시 특허 건수"],
                     filter_stats["WIPO 필터 통과 건수"],
                     filter_stats["제외된 노이즈 건수"],
                     filter_stats["필터 통과율(%)"],
                     "ROA(t+1) — Lagged DV", "Robust (HC3)",
                     "Firm FE + Year FE (LSDV, drop_first=True)"],
        })
        meta.to_excel(writer, sheet_name="0_분석개요", index=False)
        desc.to_excel(writer, sheet_name="1_기술통계량")
        corr.to_excel(writer, sheet_name="2_상관관계행렬")
        vif_df.to_excel(writer, sheet_name="3_VIF진단", index=False)
        reg_table.to_excel(writer, sheet_name="4_회귀분석(Model1-3)", index=False)

        # 열 너비 자동 조정 (가독성)
        for ws in writer.book.worksheets:
            for col_cells in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0
                            for c in col_cells)
                letter = col_cells[0].column_letter
                ws.column_dimensions[letter].width = min(width + 3, 60)


# ------------------------------------------------------------------------------
# 메인 파이프라인
# ------------------------------------------------------------------------------
def main() -> int:
    banner("STEP 1. 데이터 로드")
    patents, financials, source = load_data()
    print(f"  데이터 출처 : {source}")
    print(f"  특허 데이터 : {len(patents):,}건 / "
          f"재무 데이터 : {len(financials):,} 기업-연도")

    banner("STEP 2. WIPO IPC 필터링 & Shannon Entropy (Jacquemin & Berry, 1978)")
    entropy_panel, filter_stats = build_entropy_panel(patents)
    for k, v in filter_stats.items():
        print(f"  {k:<16}: {v:,}" if isinstance(v, int) else f"  {k:<16}: {v}")
    print(entropy_panel.head(5).to_string(index=False))

    banner("STEP 3. 패널 구축 (ROA t+1 매칭) & 평균 중심화 / 상호작용항")
    df = build_regression_panel(entropy_panel, financials)
    print(f"  최종 회귀 표본 : {len(df):,} 기업-연도 "
          f"(기업 {df['Firm'].nunique()}개, "
          f"연도 {df['Year'].min()}~{df['Year'].max()})")

    banner("STEP 4. VIF 진단 (다중공선성 점검)")
    vif_df = compute_vif(df)
    print(vif_df.to_string(index=False))
    if (vif_df["VIF"] >= 10).any():
        print("  [경고] VIF ≥ 10 변수 존재 — 다중공선성 재점검 필요")
    else:
        print("  → 모든 변수 VIF < 10 : 평균 중심화로 다중공선성 문제 해소 확인")

    banner("STEP 5. 3단계 위계적 회귀분석 (LSDV 고정효과, HC3)")
    main_effects = ["c_IV1_HW_Entropy", "c_IV2_SW_Entropy", "c_RnD_Intensity"]
    interactions = ["Int_HWxRnD", "Int_SWxRnD"]
    models = {
        "Model 1 (통제변수)": run_lsdv(df, CONTROL_VARS),
        "Model 2 (주효과)": run_lsdv(df, CONTROL_VARS + main_effects),
        "Model 3 (조절효과)": run_lsdv(df, CONTROL_VARS + main_effects
                                       + interactions),
    }
    display_vars = CONTROL_VARS + main_effects + interactions
    reg_table = build_regression_table(models, display_vars)
    print(reg_table.drop(columns=["Note"]).to_string(index=False))

    banner("STEP 6. 엑셀 리포트 저장")
    analysis_vars = ["ROA_t1", "IV1_HW_Entropy", "IV2_SW_Entropy",
                     "RnD_Intensity"] + CONTROL_VARS
    desc = df[analysis_vars].describe().T.round(4)
    corr = df[analysis_vars].corr().round(4)
    export_to_excel(desc, corr, vif_df, reg_table, filter_stats, source)
    print(f"  저장 완료 → {OUTPUT_FILE}")

    banner("분석 파이프라인 정상 종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
