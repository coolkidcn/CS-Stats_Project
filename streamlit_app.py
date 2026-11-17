import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="GA Education Equity & Pipeline Explorer", layout="wide")

# ---------------------------------------------------------
# EXACT FILE PATHS IN YOUR data/ FOLDER
# ---------------------------------------------------------
CPI_PATH = "data/CERTIFIED_PERSONNEL_2023-24_2025-01-14_16_46_12 - CERTIFIED_PERSONNEL_2023-24_2025-01-14_16_46_12.csv"
ATTEND_PATH = "data/Attendance_2023-24_2025-06-17_15_16_08 - Attendance_2023-24_2025-06-17_15_16_08.csv"
ENROLL_PATH = "data/Enrollment_by_Subgroup_Metrics_2023-24_2025-02-26_10_34_25 - Enrollment_by_Subgroup_Metrics_2023-24_2025-02-26_10_34_25.csv"
GRAD_PATH = "data/Graduation_Rate_2023-24_2025-01-14_16_53_10 - Graduation_Rate_2023-24_2025-01-14_16_53_10.csv"


# ---------------------------------------------------------
# 1. API HELPERS (Q1 Federal Data with Auto-Retries)
# ---------------------------------------------------------

def get_retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@st.cache_data(ttl=86400, show_spinner="Fetching Urban Institute API...")
def fetch_urban_paginated(url: str) -> pd.DataFrame:
    session = get_retry_session()
    results = []
    while url:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise requests.exceptions.RequestException(
                f"Urban Institute returned a non-JSON response for {resp.url} "
                f"(HTTP {resp.status_code})."
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise requests.exceptions.RequestException(
                f"Urban Institute returned an unexpected response format for {resp.url}."
            )
        results.extend(payload.get("results", []))
        url = payload.get("next")
    return pd.DataFrame(results)


@st.cache_data(ttl=86400, show_spinner="Loading Q1 Federal Datasets (CCD, SAIPE, CRDC AP)...")
def load_q1_federal_data() -> pd.DataFrame:
    try:
        # 1. District Directory (2021-22) - Crosswalk: leaid <-> GA state_leaid
        dir_df = fetch_urban_paginated(
            "https://educationdata.urban.org/api/v1/school-districts/ccd/directory/2021/?fips=13"
        )
        dir_df["leaid"] = dir_df["leaid"].astype(str).str.strip()
        dir_df["ga_system_id"] = (
            dir_df["state_leaid"].astype(str).str.replace("GA-", "", regex=False).str.strip().str.zfill(3)
        )
        dir_df = dir_df[["leaid", "ga_system_id", "lea_name", "county_code", "enrollment"]].drop_duplicates("leaid")

        # 2. District Demographic Makeup (2021-22 CCD Enrollment by Race)
        # FIX: In GA 2021 CCD data, race==99 returns 0; sum race codes 1..7 for true total enrollment
        race_df = fetch_urban_paginated(
            "https://educationdata.urban.org/api/v1/school-districts/ccd/enrollment/2021/grade-99/race/?fips=13"
        )
        race_df["leaid"] = race_df["leaid"].astype(str).str.strip()
        race_df["enrollment"] = pd.to_numeric(race_df["enrollment"], errors="coerce")

        valid_races = race_df[race_df["race"].isin([1, 2, 3, 4, 5, 6, 7]) & (race_df["enrollment"] >= 0)]
        total_enroll = (
            valid_races.groupby("leaid", as_index=False)["enrollment"]
            .sum()
            .rename(columns={"enrollment": "total_ccd_enrollment"})
        )
        black_enroll = race_df[race_df["race"] == 2][["leaid", "enrollment"]].rename(
            columns={"enrollment": "black_student_enrollment"}
        )

        demo_df = total_enroll.merge(black_enroll, on="leaid", how="left")
        demo_df["pct_black_students"] = np.where(
            demo_df["total_ccd_enrollment"] > 0,
            (demo_df["black_student_enrollment"] / demo_df["total_ccd_enrollment"] * 100).round(1),
            np.nan,
        )

        # 3. Socioeconomic Control: Census SAIPE Poverty Data (2021)
        # FIX: API returns decimal proportions (e.g., 0.2555), convert to 0-100 percentage
        saipe_df = fetch_urban_paginated(
            "https://educationdata.urban.org/api/v1/school-districts/saipe/2021/?fips=13"
        )
        saipe_df["leaid"] = saipe_df["leaid"].astype(str).str.strip()
        saipe_df["child_poverty_pct"] = (
            pd.to_numeric(saipe_df["est_population_5_17_poverty_pct"], errors="coerce") * 100
        ).round(1)
        saipe_df = saipe_df[["leaid", "child_poverty_pct"]]

        # Match Q2's 2023-24 outcomes with 2023 poverty and district urbanicity.
        context_dir = fetch_urban_paginated(
            "https://educationdata.urban.org/api/v1/school-districts/ccd/directory/2023/?fips=13"
        )
        context_dir["leaid"] = context_dir["leaid"].astype(str).str.strip()
        context_dir["ga_system_id"] = (
            context_dir["state_leaid"].astype(str).str.replace("GA-", "", regex=False).str.strip().str.zfill(3)
        )
        locale_groups = {
            11: "City", 12: "City", 13: "City",
            21: "Suburb", 22: "Suburb", 23: "Suburb",
            31: "Town", 32: "Town", 33: "Town",
            41: "Rural", 42: "Rural", 43: "Rural",
        }
        context_dir["urbanicity"] = pd.to_numeric(
            context_dir["urban_centric_locale"], errors="coerce"
        ).map(locale_groups)
        context_dir = context_dir[["leaid", "ga_system_id", "urbanicity"]].drop_duplicates("leaid")

        poverty_2023 = fetch_urban_paginated(
            "https://educationdata.urban.org/api/v1/school-districts/saipe/2023/?fips=13"
        )
        poverty_2023["leaid"] = poverty_2023["leaid"].astype(str).str.strip()
        poverty_2023["child_poverty_pct_2023"] = (
            pd.to_numeric(poverty_2023["est_population_5_17_poverty_pct"], errors="coerce") * 100
        ).round(1)
        context_df = context_dir.merge(
            poverty_2023[["leaid", "child_poverty_pct_2023"]], on="leaid", how="left"
        )

        # 4 & 5. CRDC AP Enrollment (2020)
        # CRDC records are school-level and split by race and sex, so aggregate them by district.
        ap_raw = fetch_urban_paginated(
            "https://educationdata.urban.org/api/v1/schools/crdc/ap-ib-enrollment/2020/race/sex/?fips=13"
        )
        ap_raw["leaid"] = ap_raw["leaid"].astype(str).str.strip()
        ap_raw["enrl_AP"] = pd.to_numeric(ap_raw["enrl_AP"], errors="coerce")

        # Filter out CRDC negative missing-value codes (-1, -2, -3) before summing
        ap_clean = ap_raw[ap_raw["enrl_AP"] >= 0]

        ap_tot = (
            ap_clean[ap_clean["race"] == 99]
            .groupby("leaid", as_index=False)["enrl_AP"]
            .sum()
            .rename(columns={"enrl_AP": "students_ap_enr"})
        )
        ap_black = (
            ap_clean[ap_clean["race"] == 2]
            .groupby("leaid", as_index=False)["enrl_AP"]
            .sum()
            .rename(columns={"enrl_AP": "black_students_ap_enr"})
        )

        q1_df = (
            dir_df
            .merge(demo_df, on="leaid", how="left")
            .merge(saipe_df, on="leaid", how="left")
            .merge(ap_tot, on="leaid", how="left")
            .merge(ap_black, on="leaid", how="left")
            .merge(context_df[["ga_system_id", "child_poverty_pct_2023", "urbanicity"]], on="ga_system_id", how="left")
        )
        q1_df["ap_participation_rate"] = np.where(
            q1_df["total_ccd_enrollment"] > 0,
            (q1_df["students_ap_enr"] / q1_df["total_ccd_enrollment"] * 100).round(2),
            np.nan,
        )
        q1_df["black_ap_participation_rate"] = np.where(
            q1_df["black_student_enrollment"] > 0,
            (q1_df["black_students_ap_enr"] / q1_df["black_student_enrollment"] * 100).round(2),
            np.nan,
        )
        return q1_df

    except requests.exceptions.RequestException as e:
        st.error(f"Urban Institute API error ({e}). Click 'Rerun' in the top-right menu to retry.")
        return pd.DataFrame()


# ---------------------------------------------------------
# 2. CLEAN & MERGE YOUR 4 EXACT GOEWS CSV FILES
# ---------------------------------------------------------

def _read_goews_district_csv(path: str) -> pd.DataFrame:
    """Reads a raw GOEWS CSV, normalizes headers, and filters to District-level rows."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    # Filter to District level (excludes 'State' and individual 'School' rows)
    if "DETAIL_LVL_DESC" in df.columns:
        df = df[df["DETAIL_LVL_DESC"].astype(str).str.strip().str.upper() == "DISTRICT"].copy()
    elif "INSTN_NUMBER" in df.columns:
        df = df[df["INSTN_NUMBER"].astype(str).str.strip().str.upper() == "ALL"].copy()

    # Create clean 3-digit Georgia System ID for joining
    id_col = next((c for c in ["SCHOOL_DSTRCT_CD", "SYSTEM_ID", "DISTRICT_ID"] if c in df.columns), df.columns[0])
    df["ga_system_id"] = df[id_col].astype(str).str.strip().str.zfill(3)
    return df


def _to_num(series: pd.Series) -> pd.Series:
    """Safely converts GOEWS numeric strings (handling commas, %, and '*' suppression flags)."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


@st.cache_data(show_spinner="Processing your 4 Georgia GOEWS CSVs...")
def load_all_4_goews_files() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    files = {
        "Certified Personnel": CPI_PATH,
        "Attendance": ATTEND_PATH,
        "Enrollment by Subgroup": ENROLL_PATH,
        "Graduation Rate": GRAD_PATH,
    }
    raw_tables = {}
    missing = [f for f in files.values() if not os.path.exists(f)]
    if missing:
        return pd.DataFrame(), raw_tables, missing

    cpi = _read_goews_district_csv(CPI_PATH)
    attend = _read_goews_district_csv(ATTEND_PATH)
    enroll = _read_goews_district_csv(ENROLL_PATH)
    grad = _read_goews_district_csv(GRAD_PATH)

    raw_tables = {
        "Certified Personnel": cpi,
        "Attendance": attend,
        "Enrollment by Subgroup": enroll,
        "Graduation Rate": grad,
    }

    # --- A. Certified Personnel (Teacher Race Counts) ---
    teachers = cpi[cpi["EMPLOYEE_TYPE"] == "PK-12 Teachers"]
    teacher_totals = teachers[
        (teachers["DATA_CATEGORY"] == "Positions") & (teachers["DATA_SUB_CATEGORY"] == "Number")
    ][["ga_system_id", "MEASURE"]].copy()
    teacher_totals["total_teachers"] = _to_num(teacher_totals["MEASURE"])
    teacher_totals = teacher_totals.groupby("ga_system_id", as_index=False)["total_teachers"].sum(min_count=1)

    black_teachers = teachers[
        (teachers["DATA_CATEGORY"] == "Race/Ethnicity") & (teachers["DATA_SUB_CATEGORY"] == "Black")
    ][["ga_system_id", "MEASURE"]].copy()
    black_teachers["black_teachers"] = _to_num(black_teachers["MEASURE"])
    black_teachers = black_teachers.groupby("ga_system_id", as_index=False)["black_teachers"].sum(min_count=1)

    # --- B. Black student enrollment and chronic absenteeism ---
    black_student_outcomes = attend[
        ["ga_system_id", "STUDENT_COUNT_BLACK", "CHRONIC_ABSENT_PERC_BLACK"]
    ].copy()
    black_student_outcomes["black_students"] = _to_num(black_student_outcomes["STUDENT_COUNT_BLACK"])
    black_student_outcomes["black_chronic_absent_pct"] = _to_num(
        black_student_outcomes["CHRONIC_ABSENT_PERC_BLACK"]
    )
    black_student_outcomes = black_student_outcomes[
        ["ga_system_id", "black_students", "black_chronic_absent_pct"]
    ].drop_duplicates("ga_system_id")

    # --- C. Black student enrollment share ---
    black_enrollment = enroll[["ga_system_id", "ENROLL_PCT_BLACK"]].copy()
    black_enrollment["black_enrollment_pct"] = _to_num(black_enrollment["ENROLL_PCT_BLACK"])
    black_enrollment = black_enrollment[["ga_system_id", "black_enrollment_pct"]].drop_duplicates("ga_system_id")

    # --- D. Black four-year graduation rate ---
    black_graduation = grad[
        grad["LABEL_LVL_1_DESC"].astype(str).str.strip() == "Grad Rate -Black"
    ][["ga_system_id", "PROGRAM_TOTAL", "PROGRAM_PERCENT", "TOTAL_COUNT"]].copy()
    black_graduation["black_grad_successes"] = _to_num(black_graduation["PROGRAM_TOTAL"])
    black_graduation["black_grad_rate"] = _to_num(black_graduation["PROGRAM_PERCENT"])
    black_graduation["black_grad_total"] = _to_num(black_graduation["TOTAL_COUNT"])
    black_graduation = black_graduation[
        ["ga_system_id", "black_grad_successes", "black_grad_total", "black_grad_rate"]
    ].drop_duplicates("ga_system_id")

    q2_df = (
        teacher_totals
        .merge(black_teachers, on="ga_system_id", how="outer")
        .merge(black_student_outcomes, on="ga_system_id", how="outer")
        .merge(black_enrollment, on="ga_system_id", how="left")
        .merge(black_graduation, on="ga_system_id", how="left")
    )
    q2_df["black_teacher_to_student_ratio"] = np.where(
        q2_df["black_students"] > 0,
        (q2_df["black_teachers"] / q2_df["black_students"] * 100).round(2),
        np.nan,
    )

    return q2_df, raw_tables, []


# ---------------------------------------------------------
# 3. BUILD UI TABS
# ---------------------------------------------------------
q1_fed_df = load_q1_federal_data()
q2_ga_df, raw_goews_tables, missing_files = load_all_4_goews_files()

st.title("🎓 Georgia Education Equity & Teacher Pipeline Explorer")

tab_q1, tab_q2, tab_master, tab_inspect = st.tabs([
    "📚 Q1: Curricular Access & Race",
    "👩‍🏫 Q2: Teacher Representation & Outcomes",
    "🔗 Connected Master Dataset",
    "🗂️ Raw GOEWS CSV Viewer"
])

# --- TAB 1: QUESTION 1 ---
with tab_q1:
    st.subheader("Q1: Does District Demographics Predict AP Course Access (Controlling for Size & Poverty)?")
    st.caption("Sources: Urban Institute 2021–22 CCD Enrollment + 2021 Census SAIPE Poverty + 2020 CRDC AP Enrollment")

    if not q1_fed_df.empty:
        plot_q1 = q1_fed_df.dropna(subset=["pct_black_students", "ap_participation_rate", "child_poverty_pct", "total_ccd_enrollment"])
        fig1 = px.scatter(
            plot_q1,
            x="pct_black_students",
            y="ap_participation_rate",
            size="total_ccd_enrollment",
            color="child_poverty_pct",
            hover_name="lea_name",
            labels={
                "pct_black_students": "% Black Students (CCD)",
                "ap_participation_rate": "AP Enrollment % of District (CRDC)",
                "child_poverty_pct": "Child Poverty % (SAIPE)"
            },
            title="AP Participation Rate vs. District % Black Students (Sized by Enrollment, Colored by SAIPE Poverty %)"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.dataframe(
            plot_q1[["ga_system_id", "lea_name", "total_ccd_enrollment", "pct_black_students", "child_poverty_pct", "students_ap_enr", "ap_participation_rate"]],
            use_container_width=True
        )

# --- TAB 2: QUESTION 2 ---
with tab_q2:
    st.subheader("Q2: Black Teacher-to-Student Ratio vs. Black Student Outcomes (2023–24)")
    st.caption(
        "Sources: GOEWS 2023–24 district data, Urban Institute/Census 2023 district poverty "
        "estimates, and 2023 district urbanicity."
    )

    if missing_files:
        st.error(f"Could not find these files: {missing_files}")
    elif not q2_ga_df.empty:
        q2_display_df = (
            q2_ga_df.merge(
                q1_fed_df[["ga_system_id", "lea_name", "child_poverty_pct_2023", "urbanicity"]],
                on="ga_system_id",
                how="left",
            )
            if not q1_fed_df.empty
            else q2_ga_df.assign(
                lea_name=q2_ga_df["ga_system_id"],
                child_poverty_pct_2023=np.nan,
                urbanicity="Unknown",
            )
        )

        metric_choice = st.radio(
            "Select Black Student Outcome:",
            ["black_grad_rate", "black_chronic_absent_pct"],
            format_func=lambda x: "Black 4-Year Graduation Rate (%)" if x == "black_grad_rate" else "Black Absent >15 Days (%)",
            horizontal=True
        )

        model_columns = [
            "black_teacher_to_student_ratio",
            "black_students",
            "child_poverty_pct_2023",
            "urbanicity",
            metric_choice,
        ]
        if metric_choice == "black_grad_rate":
            model_columns.extend(["black_grad_successes", "black_grad_total"])
        model_data = q2_display_df.dropna(subset=model_columns).copy()

        if metric_choice == "black_grad_rate":
            model_data = model_data[
                (model_data["black_grad_total"] > 0)
                & (model_data["black_grad_successes"] >= 0)
                & (model_data["black_grad_successes"] <= model_data["black_grad_total"])
            ].copy()
            model_data["black_grad_proportion"] = (
                model_data["black_grad_successes"] / model_data["black_grad_total"]
            )
        else:
            model_data = model_data[model_data["black_students"] > 0].copy()

        if (
            len(model_data) >= 8
            and model_data["black_teacher_to_student_ratio"].nunique() > 1
            and model_data["child_poverty_pct_2023"].nunique() > 1
        ):
            poverty_options = model_data["child_poverty_pct_2023"].astype(float)
            default_poverty = float(poverty_options.median())
            compare_poverty = st.slider(
                "Compare districts at this child-poverty rate (%)",
                min_value=float(poverty_options.min()),
                max_value=float(poverty_options.max()),
                value=default_poverty,
                step=0.1,
            )
            urbanicity_options = sorted(model_data["urbanicity"].unique())
            default_urbanicity = model_data["urbanicity"].mode().iloc[0]
            compare_urbanicity = st.selectbox(
                "Compare districts with this urbanicity",
                urbanicity_options,
                index=urbanicity_options.index(default_urbanicity),
            )

            model_formula = (
                f"{metric_choice} ~ black_teacher_to_student_ratio "
                "+ child_poverty_pct_2023 + C(urbanicity)"
            )
            if metric_choice == "black_grad_rate":
                model = smf.glm(
                    "black_grad_proportion ~ black_teacher_to_student_ratio "
                    "+ child_poverty_pct_2023 + C(urbanicity)",
                    data=model_data,
                    family=sm.families.Binomial(),
                    var_weights=model_data["black_grad_total"],
                ).fit(cov_type="HC3")
                outcome_label = "Black 4-Year Graduation Rate (%)"
                prediction_scale = 100
            else:
                model = smf.wls(
                    model_formula,
                    data=model_data,
                    weights=model_data["black_students"],
                ).fit(cov_type="HC3")
                outcome_label = "Black Chronic Absenteeism (%)"
                prediction_scale = 1

            ratio_grid = np.linspace(
                model_data["black_teacher_to_student_ratio"].min(),
                model_data["black_teacher_to_student_ratio"].max(),
                100,
            )
            prediction_data = pd.DataFrame(
                {
                    "black_teacher_to_student_ratio": ratio_grid,
                    "child_poverty_pct_2023": compare_poverty,
                    "urbanicity": compare_urbanicity,
                }
            )
            prediction = model.get_prediction(prediction_data).summary_frame()
            expected = prediction["mean"] * prediction_scale
            lower = prediction["mean_ci_lower"] * prediction_scale
            upper = prediction["mean_ci_upper"] * prediction_scale
            fig2 = px.scatter(
                model_data,
                x="black_teacher_to_student_ratio",
                y=metric_choice,
                size="black_students",
                color="child_poverty_pct_2023",
                hover_name="lea_name",
                labels={
                    "black_teacher_to_student_ratio": "Black Teachers per 100 Black Students",
                    "black_grad_rate": "Black 4-Year Graduation Rate (%)",
                    "black_chronic_absent_pct": "Black Absent >15 Days (%)",
                    "child_poverty_pct_2023": "District child poverty (%)",
                },
                title=f"{outcome_label} vs. Black Teacher Representation",
            )
            fig2.add_trace(
                go.Scatter(
                    x=ratio_grid,
                    y=upper,
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=ratio_grid,
                    y=lower,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(31, 119, 180, 0.18)",
                    name="95% confidence interval",
                    hoverinfo="skip",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=ratio_grid,
                    y=expected,
                    mode="lines",
                    line={"color": "#1f77b4", "width": 3},
                    name="Adjusted predicted outcome",
                )
            )
            st.plotly_chart(fig2, use_container_width=True)

            ratio_low, ratio_high = model_data["black_teacher_to_student_ratio"].quantile(
                [0.25, 0.75]
            )
            contrast_data = pd.DataFrame(
                {
                    "black_teacher_to_student_ratio": [ratio_low, ratio_high],
                    "child_poverty_pct_2023": compare_poverty,
                    "urbanicity": compare_urbanicity,
                }
            )
            predicted_contrast = (
                model.predict(contrast_data).iloc[1] - model.predict(contrast_data).iloc[0]
            ) * prediction_scale
            st.subheader("Adjusted comparison at the selected poverty and urbanicity")
            st.caption(
                "The line compares model-predicted outcomes across the observed teacher-ratio "
                "range while holding poverty and urbanicity at your selections. Graduation is "
                "modeled using Black graduates and cohort totals; chronic absenteeism uses "
                "Black student counts as precision weights. District-level observational "
                "associations are not student-level or causal effects. Poverty is estimated "
                "for all district children, not specifically Black students."
            )
            col_contrast, col_effect, col_ci, col_p, col_n = st.columns(5)
            col_contrast.metric(
                "Predicted difference (25th to 75th ratio)",
                f"{predicted_contrast:+.2f} pp",
            )
            if metric_choice == "black_grad_rate":
                teacher_odds_ratio = np.exp(model.params["black_teacher_to_student_ratio"])
                odds_interval = np.exp(
                    model.conf_int().loc["black_teacher_to_student_ratio"]
                )
                col_effect.metric("Odds ratio per +1 teacher / 100 students", f"{teacher_odds_ratio:.2f}")
                col_ci.metric(
                    "95% odds-ratio interval",
                    f"{odds_interval.iloc[0]:.2f} to {odds_interval.iloc[1]:.2f}",
                )
            else:
                teacher_effect = model.params["black_teacher_to_student_ratio"]
                teacher_interval = model.conf_int().loc["black_teacher_to_student_ratio"]
                col_effect.metric(
                    "Adjusted change per +1 teacher / 100 students",
                    f"{teacher_effect:.2f} pp",
                )
                col_ci.metric(
                    "95% confidence interval",
                    f"{teacher_interval.iloc[0]:.2f} to {teacher_interval.iloc[1]:.2f}",
                )
            col_p.metric("P-value", f"{model.pvalues['black_teacher_to_student_ratio']:.3g}")
            col_n.metric("Districts in model", f"{int(model.nobs)}")
        else:
            st.info(
                "The adjusted analysis needs at least 8 districts with valid outcomes, "
                "graduation cohort counts when applicable, poverty, and teacher-ratio variation."
            )
        st.dataframe(
            q2_display_df[
                [
                    "ga_system_id",
                    "lea_name",
                    "black_teachers",
                    "black_students",
                    "black_teacher_to_student_ratio",
                    "black_grad_rate",
                    "black_grad_successes",
                    "black_grad_total",
                    "black_chronic_absent_pct",
                    "child_poverty_pct_2023",
                    "urbanicity",
                ]
            ],
            use_container_width=True
        )
    else:
        st.warning("Files loaded, but 0 rows matched after filtering. Check the 'Raw GOEWS CSV Viewer' tab to inspect column headers.")

# --- TAB 3: CONNECTED MASTER DATASET ---
with tab_master:
    st.subheader("All Sources Joined (`leaid` ↔ `ga_system_id`)")
    if not q1_fed_df.empty and not q2_ga_df.empty:
        master_df = q1_fed_df.merge(q2_ga_df, on="ga_system_id", how="inner")
        st.dataframe(master_df, use_container_width=True)
    else:
        st.info("Once both federal API and GOEWS CSV tables are loaded, the joined dataset will appear here.")

# --- TAB 4: RAW GOEWS CSV VIEWER ---
with tab_inspect:
    st.subheader("Your 4 GOEWS CSV Files (District-Level Rows)")
    if raw_goews_tables:
        for name, df_table in raw_goews_tables.items():
            st.markdown(f"### {name} ({len(df_table)} district rows)")
            st.caption(f"Columns: {', '.join(df_table.columns)}")
            st.dataframe(df_table.head(50), use_container_width=True)