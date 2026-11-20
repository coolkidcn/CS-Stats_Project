import os
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from process_data import build_clean_dataset

st.set_page_config(page_title="Blackness Impact on Education — Statistical Inference", layout="wide")

DATA_PATH = "data/clean_ga_high_schools.csv"
DEFAULT_BLACK_TEACHER_THRESHOLD = 30.0


@st.cache_data(show_spinner="Preparing clean Georgia high school dataset...")
def get_clean_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        required_columns = {
            "pct_economically_disadvantaged",
            "black_grad_successes",
            "black_grad_cohort_size",
        }
        if required_columns.issubset(df.columns):
            return df
    return build_clean_dataset()


def two_sample_t_analysis(group1: pd.Series, group2: pd.Series, alternative: str = "greater"):
    """
    Performs Welch's two-sample t-test and computes a 95% two-sample t-confidence interval
    for (mean_1 - mean_2).
    """
    g1 = group1.dropna().astype(float)
    g2 = group2.dropna().astype(float)

    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)

    diff = m1 - m2
    se = np.sqrt((s1**2 / n1) + (s2**2 / n2))

    df = ((s1**2 / n1) + (s2**2 / n2)) ** 2 / (
        ((s1**2 / n1) ** 2 / (n1 - 1)) + ((s2**2 / n2) ** 2 / (n2 - 1))
    )

    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False, alternative=alternative)

    t_crit = stats.t.ppf(0.975, df=df)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    return {
        "n1": n1, "m1": m1, "s1": s1,
        "n2": n2, "m2": m2, "s2": s2,
        "diff": diff, "se": se, "df": df,
        "t_stat": t_stat, "p_val": p_val,
        "ci_low": ci_low, "ci_high": ci_high,
    }


st.title("Statistical Inference:Impact on Education of Blackness (Georgia High Schools)")
st.caption("GaDOE / GOEWS Public High School Data")

try:
    df = get_clean_data()
except FileNotFoundError as e:
    st.error(f"Missing raw GOEWS CSV in `data/`: {e}")
    st.stop()

if st.sidebar.button("Rebuild Clean Dataset from Raw CSVs"):
    st.cache_data.clear()
    df = build_clean_dataset()
    st.rerun()


teacher_threshold = DEFAULT_BLACK_TEACHER_THRESHOLD

st.header("Statistical Inference")
st.subheader("Randomization")
st.markdown(
    "Choose a random seed for a the simple random sample of 50 schools for each group"
)
random_seed = st.number_input(
    "Random seed ",
    min_value=1,
    value=67,
    step=1,
    key="random_seed_input",
)
if st.button("View results", type="primary"):
    st.session_state["randomization_seed"] = int(random_seed)

if st.session_state.get("randomization_seed") != int(random_seed):
    st.info("Set up and view results")
    st.stop()

higher_teacher_label = f">{teacher_threshold:g}% Black Teachers"
lower_teacher_label = f"<{teacher_threshold:g}% Black Teachers"
teacher_share = pd.to_numeric(df["pct_black_teachers"], errors="coerce")
has_teacher_data = teacher_share.notna() & teacher_share.ne(teacher_threshold)
df["teacher_group_q1"] = pd.Series(pd.NA, index=df.index, dtype="string")
df.loc[has_teacher_data, "teacher_group_q1"] = np.where(
    teacher_share.loc[has_teacher_data] > teacher_threshold,
    higher_teacher_label,
    lower_teacher_label,
)

tab_qa, tab_qb, tab_data = st.tabs([
    "Question A",
    "Question B",
    "Matched High School Data"
])

# QUESTION A

with tab_qa:
    st.header("Question A: Black Teacher Representation & Black Student Graduation Rate")
    st.markdown(
        "**Research Question:** Is there a significant difference in mean Black student 4-year cohort "
        f"graduation rate between Georgia high schools with **{higher_teacher_label}** ($\\mu_1$) and "
        f"**{lower_teacher_label}** ($\\mu_2$)?"
    )
    st.latex(r"H_0: \mu_1 - \mu_2 = 0 \qquad \text{vs.} \qquad H_a: \mu_1 - \mu_2 > 0 \quad (\alpha = 0.05)")

    qa_df = df.dropna(subset=["teacher_group_q1", "black_grad_rate"]).copy()
    g1_df = qa_df[qa_df["teacher_group_q1"] == higher_teacher_label]
    g2_df = qa_df[qa_df["teacher_group_q1"] == lower_teacher_label]

    g1_df = g1_df.sample(n=min(50, len(g1_df)), random_state=int(random_seed))
    g2_df = g2_df.sample(n=min(50, len(g2_df)), random_state=int(random_seed))
    qa_plot_df = pd.concat([g1_df, g2_df])

    res_a = two_sample_t_analysis(g1_df["black_grad_rate"], g2_df["black_grad_rate"], alternative="greater")

    st.subheader("1. Sample Summary Statistics")
    summary_a = pd.DataFrame({
        "Group": [f"Group 1: {higher_teacher_label}", f"Group 2: {lower_teacher_label}"],
        "Sample Size (n)": [res_a["n1"], res_a["n2"]],
        "Sample Mean (x̄ %)": [round(res_a["m1"], 2), round(res_a["m2"], 2)],
        "Sample SD (s %)": [round(res_a["s1"], 2), round(res_a["s2"], 2)],
    })
    st.table(summary_a)

    fig_a = px.box(
        qa_plot_df,
        x="teacher_group_q1",
        y="black_grad_rate",
        color="teacher_group_q1",
        points="all",
        hover_name="school_name",
        labels={
            "teacher_group_q1": "Teacher Representation Group",
            "black_grad_rate": "Black 4-Year Graduation Rate (%)"
        },
        title="Distribution of Black Student Graduation Rates by Teacher Representation Group"
    )
    st.plotly_chart(fig_a, width="stretch")

    st.subheader("2. Two-Sample t-Test & 95% Confidence Interval")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Difference (x̄₁ − x̄₂)", f"{res_a['diff']:+.2f} pp")
    c2.metric("t-Statistic (df = {:.1f})".format(res_a["df"]), f"{res_a['t_stat']:.3f}")
    c3.metric("One-Sided p-value", f"{res_a['p_val']:.4f}")
    c4.metric("95% Confidence Interval", f"[{res_a['ci_low']:+.2f}, {res_a['ci_high']:+.2f}]")

    if res_a["p_val"] < 0.05:
        st.success(
            f"**Conclusion (Reject H₀ at α = 0.05):** Because p = {res_a['p_val']:.4f} is less than 0.05, "
            f"there is statistically significant evidence that Georgia high schools with {higher_teacher_label} "
            f"have a higher mean Black student graduation rate than schools with {lower_teacher_label}."
        )
    else:
        st.info(
            f"**Conclusion (Fail to Reject H₀ at α = 0.05):** Because p = {res_a['p_val']:.4f} is greater than 0.05, "
            f"we do not have statistically significant evidence at the 0.05 level that high schools with "
            f"{higher_teacher_label} have a higher mean Black student graduation rate."
        )
        


# QUESTION B

with tab_qb:
    st.header("Question B: School Demographics & AP Test Participation")
    st.markdown(
        "**Research Question:** Is there a significant difference in mean percentage of students taking AP exams "
        "between **Majority White high schools** ($\\mu_1$) and **Majority Black high schools** ($\\mu_2$)?"
    )
    st.latex(r"H_0: \mu_1 - \mu_2 = 0 \qquad \text{vs.} \qquad H_a: \mu_1 - \mu_2 > 0 \quad (\alpha = 0.05)")

    qb_df = df[df["school_demo_group_q2"].isin(["Majority Black (>50%)", "Majority White (>50%)"])].dropna(
        subset=["pct_ap_test_takers"]
    ).copy()

    b1_df = qb_df[qb_df["school_demo_group_q2"] == "Majority White (>50%)"]
    b2_df = qb_df[qb_df["school_demo_group_q2"] == "Majority Black (>50%)"]

    b1_df = b1_df.sample(n=min(50, len(b1_df)), random_state=int(random_seed))
    b2_df = b2_df.sample(n=min(50, len(b2_df)), random_state=int(random_seed))
    qb_plot_df = pd.concat([b1_df, b2_df])

    if len(b1_df) > 1 and len(b2_df) > 1:
        res_b = two_sample_t_analysis(b1_df["pct_ap_test_takers"], b2_df["pct_ap_test_takers"], alternative="greater")

        st.subheader("1. Sample Summary Statistics")
        summary_b = pd.DataFrame({
            "Group": ["Group 1: Majority White (>50%)", "Group 2: Majority Black (>50%)"],
            "Sample Size (n)": [res_b["n1"], res_b["n2"]],
            "Sample Mean (x̄ %)": [round(res_b["m1"], 2), round(res_b["m2"], 2)],
            "Sample SD (s %)": [round(res_b["s1"], 2), round(res_b["s2"], 2)],
        })
        st.table(summary_b)

        fig_b = px.box(
            qb_plot_df,
            x="school_demo_group_q2",
            y="pct_ap_test_takers",
            color="school_demo_group_q2",
            points="all",
            hover_name="school_name",
            labels={
                "school_demo_group_q2": "School Demographic Group",
                "pct_ap_test_takers": "AP Test Takers (% of Total Enrollment)"
            },
            title="Distribution of AP Exam Participation by School Racial Majority"
        )
        st.plotly_chart(fig_b, width="stretch")

        st.subheader("2. Two-Sample t-Test & 95% Confidence Interval")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Difference (x̄₁ − x̄₂)", f"{res_b['diff']:+.2f} pp")
        c2.metric("t-Statistic (df = {:.1f})".format(res_b["df"]), f"{res_b['t_stat']:.3f}")
        c3.metric("One-Sided p-value", f"{res_b['p_val']:.4f}")
        c4.metric("95% Confidence Interval", f"[{res_b['ci_low']:+.2f}, {res_b['ci_high']:+.2f}]")

        if res_b["p_val"] < 0.05:
            st.success(
                f"**Conclusion (Reject H₀ at α = 0.05):** Because p = {res_b['p_val']:.4f} is less than 0.05, "
                f"there is statistically significant evidence that majority-Black high schools have a lower mean "
                f"percentage of AP test takers than majority-White high schools."
            )
        else:
            st.info(
                f"**Conclusion (Fail to Reject H₀ at α = 0.05):** Because p = {res_b['p_val']:.4f} is greater than 0.05, "
                f"we do not have statistically significant evidence at the 0.05 level of a lower AP testing rate."
            )
    
# CLEAN SCHOOL-LEVEL DATA

with tab_data:
    st.subheader("Processed High School Dataset")
    st.dataframe(df, width="stretch")
