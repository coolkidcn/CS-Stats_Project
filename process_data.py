import os
import numpy as np
import pandas as pd

CPI_PATH = "data/CERTIFIED_PERSONNEL_2023-24_2025-01-14_16_46_12 - CERTIFIED_PERSONNEL_2023-24_2025-01-14_16_46_12.csv"
ATTEND_PATH = "data/Attendance_2023-24_2025-06-17_15_16_08 - Attendance_2023-24_2025-06-17_15_16_08.csv"
ENROLL_PATH = "data/Enrollment_by_Subgroup_Metrics_2023-24_2025-02-26_10_34_25 - Enrollment_by_Subgroup_Metrics_2023-24_2025-02-26_10_34_25.csv"
GRAD_PATH = "data/Graduation_Rate_2023-24_2025-01-14_16_53_10 - Graduation_Rate_2023-24_2025-01-14_16_53_10.csv"
AP_PATH = "data/AP_2023-24_2025-01-15_15_03_20 - AP_2023-24_2025-01-15_15_03_20.csv"
OUTPUT_PATH = "data/clean_ga_high_schools.csv"
DEFAULT_BLACK_TEACHER_THRESHOLD = 30.0


def to_num(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def read_school_level_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    if "DETAIL_LVL_DESC" in df.columns:
        df = df[
            df["DETAIL_LVL_DESC"].astype(str).str.strip().str.upper().str.startswith("SCHOOL")
        ].copy()

    district_id_col = next(
        (column for column in ("SCHOOL_DSTRCT_CD", "SCHOOL_DISTRCT_CD") if column in df.columns),
        None,
    )
    if district_id_col is None:
        raise ValueError(f"{path} is missing a district ID column.")
    if "INSTN_NUMBER" not in df.columns:
        raise ValueError(f"{path} is missing the INSTN_NUMBER school ID column.")

    dist_id = df[district_id_col].astype(str).str.strip().str.zfill(3)
    sch_id = df["INSTN_NUMBER"].astype(str).str.strip().str.zfill(4)
    df["school_uid"] = dist_id + "-" + sch_id
    return df


def build_clean_dataset(teacher_threshold: float = DEFAULT_BLACK_TEACHER_THRESHOLD) -> pd.DataFrame:
    if not 0 <= teacher_threshold <= 100:
        raise ValueError("teacher_threshold must be between 0 and 100.")

    cpi = read_school_level_csv(CPI_PATH)
    attend = read_school_level_csv(ATTEND_PATH)
    enroll = read_school_level_csv(ENROLL_PATH)
    grad = read_school_level_csv(GRAD_PATH)
    ap = read_school_level_csv(AP_PATH)

    # A. School Metadata & Enrollment by Race
    name_col = next((c for c in ["INSTN_NAME", "SCHOOL_NAME"] if c in enroll.columns), None)
    dist_col = next((c for c in ["SCHOOL_DSTRCT_NM", "DISTRICT_NAME"] if c in enroll.columns), None)
    tot_enroll_col = next((c for c in enroll.columns if "ENROLL" in c and ("ALL" in c or "TOT" in c or "COUNT" in c)), None)

    meta_cols = ["school_uid"]
    if name_col:
        meta_cols.append(name_col)
    if dist_col:
        meta_cols.append(dist_col)

    schools_df = enroll[meta_cols].drop_duplicates("school_uid").rename(
        columns={name_col: "school_name", dist_col: "district_name"}
    )

    enroll_clean = enroll[["school_uid"]].copy()
    if tot_enroll_col:
        enroll_clean["total_students"] = to_num(enroll[tot_enroll_col])
    if "ENROLL_PCT_BLACK" in enroll.columns:
        enroll_clean["pct_black_students"] = to_num(enroll["ENROLL_PCT_BLACK"])
    if "ENROLL_PCT_WHITE" in enroll.columns:
        enroll_clean["pct_white_students"] = to_num(enroll["ENROLL_PCT_WHITE"])
    if "ENROLL_PCT_ED" in enroll.columns:
        enroll_clean["pct_economically_disadvantaged"] = to_num(enroll["ENROLL_PCT_ED"])
    enroll_clean = enroll_clean.drop_duplicates("school_uid")

    attendance_clean = attend[["school_uid", "STUDENT_COUNT_ALL"]].copy()
    attendance_clean["total_students"] = to_num(attendance_clean["STUDENT_COUNT_ALL"])
    attendance_clean = attendance_clean[["school_uid", "total_students"]].drop_duplicates("school_uid")
    enroll_clean = enroll_clean.drop(columns=["total_students"], errors="ignore").merge(
        attendance_clean, on="school_uid", how="left"
    )

    # B. Teacher Demographics (PK-12 Teachers)
    teachers = cpi[cpi["EMPLOYEE_TYPE"].astype(str).str.strip() == "PK-12 Teachers"].copy()

    tot_t = teachers[
        (teachers["DATA_CATEGORY"] == "Positions") & (teachers["DATA_SUB_CATEGORY"] == "Number")
    ][["school_uid", "MEASURE"]].copy()
    tot_t["total_teachers"] = to_num(tot_t["MEASURE"])
    tot_t = tot_t.groupby("school_uid", as_index=False)["total_teachers"].sum(min_count=1)

    blk_t = teachers[
        (teachers["DATA_CATEGORY"] == "Race/Ethnicity") & (teachers["DATA_SUB_CATEGORY"] == "Black")
    ][["school_uid", "MEASURE"]].copy()
    blk_t["black_teachers"] = to_num(blk_t["MEASURE"])
    blk_t = blk_t.groupby("school_uid", as_index=False)["black_teachers"].sum(min_count=1)

    teachers_clean = tot_t.merge(blk_t, on="school_uid", how="inner")
    teachers_clean["pct_black_teachers"] = np.where(
        teachers_clean["total_teachers"] > 0,
        (teachers_clean["black_teachers"] / teachers_clean["total_teachers"] * 100).round(2),
        np.nan,
    )

    # C. Black 4-Year Graduation Rate (Isolates High Schools)
    grad_blk = grad[
        grad["LABEL_LVL_1_DESC"].astype(str).str.strip() == "Grad Rate -Black"
    ][["school_uid", "PROGRAM_TOTAL", "PROGRAM_PERCENT", "TOTAL_COUNT"]].copy()
    grad_blk["black_grad_successes"] = to_num(grad_blk["PROGRAM_TOTAL"])
    grad_blk["black_grad_rate"] = to_num(grad_blk["PROGRAM_PERCENT"])
    grad_blk["black_grad_cohort_size"] = to_num(grad_blk["TOTAL_COUNT"])
    grad_blk = grad_blk[
        ["school_uid", "black_grad_successes", "black_grad_rate", "black_grad_cohort_size"]
    ].drop_duplicates("school_uid")

    # D. AP Test Takers
    ap_sub = ap.copy()
    for col in ["TEST_CMPNT_TYP_NM", "TEST_CMPNT_TYP_CD", "SUBJECT", "SUBGROUP", "LABEL_LVL_1_DESC"]:
        if col in ap_sub.columns:
            all_rows = ap_sub[ap_sub[col].astype(str).str.contains("All", case=False, na=False)]
            if not all_rows.empty:
                ap_sub = all_rows

    ap_takers_col = next(
        (
            c
            for c in [
                "STUDENTS_TESTED",
                "TEST_TAKER_CNT",
                "NUM_TESTED",
                "STUDENT_COUNT",
                "TOTAL_COUNT",
                "NUMBER_OF_STUDENTS_TESTED",
            ]
            if c in ap_sub.columns
        ),
        None,
    )
    ap_clean = ap_sub[["school_uid"]].copy()
    if ap_takers_col:
        ap_clean["ap_test_takers"] = to_num(ap_sub[ap_takers_col])
    ap_clean = ap_clean.drop_duplicates("school_uid")

    # E. Merge & Create Study Groups
    master = (
        schools_df
        .merge(enroll_clean, on="school_uid", how="inner")
        .merge(teachers_clean, on="school_uid", how="left")
        .merge(grad_blk, on="school_uid", how="inner")
        .merge(ap_clean, on="school_uid", how="left")
    )

    if "total_students" in master.columns and "ap_test_takers" in master.columns:
        master["pct_ap_test_takers"] = np.where(
            master["total_students"] > 0,
            (master["ap_test_takers"] / master["total_students"] * 100).round(2),
            np.nan,
        )

    master["teacher_group_q1"] = pd.Series(pd.NA, index=master.index, dtype="string")
    has_teacher_data = master["pct_black_teachers"].notna()
    master.loc[has_teacher_data, "teacher_group_q1"] = np.where(
        master.loc[has_teacher_data, "pct_black_teachers"] > teacher_threshold,
        f">{teacher_threshold:g}% Black Teachers",
        f"≤{teacher_threshold:g}% Black Teachers",
    )

    def classify_school_demo(row):
        if pd.notna(row.get("pct_black_students")) and row["pct_black_students"] > 50.0:
            return "Majority Black (>50%)"
        if pd.notna(row.get("pct_white_students")) and row["pct_white_students"] > 50.0:
            return "Majority White (>50%)"
        return "No Majority / Diverse"

    master["school_demo_group_q2"] = master.apply(classify_school_demo, axis=1)

    os.makedirs("data", exist_ok=True)
    master.to_csv(OUTPUT_PATH, index=False)
    return master


if __name__ == "__main__":
    df_out = build_clean_dataset()
    print(f"Saved {len(df_out)} Georgia high schools to {OUTPUT_PATH}")
