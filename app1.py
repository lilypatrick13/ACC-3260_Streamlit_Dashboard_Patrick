import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Benford Dashboard", layout="wide")
st.title("Benford Analysis Dashboard")


# ---------- load ----------
def load(f):
    df = pd.read_csv(f, low_memory=False) if f.name.lower().endswith(".csv") else pd.read_excel(f)
    df.columns = df.columns.str.strip().str.lower()
    return df


st.sidebar.header("1. Upload")
file1 = st.sidebar.file_uploader("Period 1 file (csv or Excel)", type=["csv", "xlsx", "xls"], key="f1")
file2 = st.sidebar.file_uploader("Period 2 file (optional)", type=["csv", "xlsx", "xls"], key="f2")
if file1 is None:
    st.info("Upload a csv or Excel file in the sidebar to begin.")
    st.stop()

df1 = load(file1)
df2 = load(file2) if file2 is not None else None

# Use the uploaded file names (without extension) as labels
name1 = file1.name.rsplit(".", 1)[0]
name2 = file2.name.rsplit(".", 1)[0] if file2 is not None else "Period 2"
if name1 == name2:
    name1, name2 = name1 + " (1)", name2 + " (2)"

# ---------- filters ----------
st.sidebar.header("2. Filters")
amount_col = st.sidebar.selectbox("Amount column", list(df1.columns),
                                  index=list(df1.columns).index("payment_amount") if "payment_amount" in df1.columns else 0)
min_amt = st.sidebar.number_input("Filter 1: Minimum amount", value=0.0)
max_amt = st.sidebar.number_input("Filter 1: Maximum amount (0 = no limit)", value=0.0)

cat_col = st.sidebar.selectbox("Filter 2: Pick a column", ["(none)"] + list(df1.columns))
cat_vals = []
if cat_col != "(none)":
    options = sorted(df1[cat_col].dropna().astype(str).unique())[:1000]
    cat_vals = st.sidebar.multiselect("Filter 2: Keep these values", options)

txt_col = st.sidebar.selectbox("Filter 3: Text search column", ["(none)"] + list(df1.columns))
txt = st.sidebar.text_input("Filter 3: Contains text") if txt_col != "(none)" else ""


def apply_filters(df):
    if amount_col not in df.columns:
        return df
    df = df.copy()
    amt = pd.to_numeric(df[amount_col], errors="coerce").abs()
    df = df[amt >= min_amt]
    amt = pd.to_numeric(df[amount_col], errors="coerce").abs()
    if max_amt > 0:
        df = df[amt <= max_amt]
    if cat_col != "(none)" and cat_vals and cat_col in df.columns:
        df = df[df[cat_col].astype(str).isin(cat_vals)]
    if txt_col != "(none)" and txt and txt_col in df.columns:
        df = df[df[txt_col].astype(str).str.contains(txt, case=False, na=False)]
    return df


df1 = apply_filters(df1)
df2 = apply_filters(df2) if df2 is not None else None
st.write(f"Rows after filters: {name1} = {len(df1):,}" + (f" | {name2} = {len(df2):,}" if df2 is not None else ""))


# ---------- benford ----------
def benford(df, n):
    amount = pd.to_numeric(df[amount_col], errors="coerce").abs()
    amount = amount[amount > 0]
    digit = np.floor(amount / (10 ** (np.floor(np.log10(amount)) - (n - 1)))).astype(int)
    digits = range(1, 10) if n == 1 else range(10, 100)
    counts = digit.value_counts().reindex(digits, fill_value=0)
    out = pd.DataFrame({"digit": list(digits), "count": counts.values})
    out["percentage"] = out["count"] / out["count"].sum() * 100
    out["benford_percentage"] = np.log10(1 + 1 / out["digit"]) * 100
    out["difference"] = out["percentage"] - out["benford_percentage"]
    return out


# EXACT KS statistic from the notebook
def ks_test(pct_a, pct_b, count_a):
    cum_diff = (np.cumsum(pct_a) - np.cumsum(pct_b)) / 100
    max_diff = np.max(np.abs(cum_diff))
    critical_value = 1.36 / np.sqrt(count_a.sum())
    return max_diff, critical_value


def plot(cols, labels, title, xlabel):
    fig, ax = plt.subplots(figsize=(12, 5))
    cols.plot(kind="bar", ax=ax, width=0.8, title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage")
    ax.legend(labels)
    plt.tight_layout()
    return fig


def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


# ---------- analysis ----------
all_tables, summary_rows, charts = [], [], {}
tabs = st.tabs(["First digit", "First two digits", "Summary statistics"])

for tab, n, name in zip(tabs[:2], [1, 2], ["First digit", "First two digits"]):
    with tab:
        t1 = benford(df1, n)
        merged = t1
        series = ["percentage", "benford_percentage"]
        labels = [name1, "Benford's Law"]
        if df2 is not None:
            t2 = benford(df2, n)
            merged = t1.merge(t2, on="digit", suffixes=("_p1", "_p2"))
            merged["benford_percentage"] = merged["benford_percentage_p1"]
            series = ["percentage_p1", "percentage_p2", "benford_percentage"]
            labels = [name1, name2, "Benford's Law"]
        st.subheader(name)
        st.dataframe(merged, use_container_width=True)
        fig = plot(merged.set_index("digit")[series], labels, f"{name} Percentages vs. Benford's Law", name)
        st.pyplot(fig)
        charts[name] = fig_bytes(fig)

        # KS statistics
        st.markdown("**KS test results** (reject if Maximum difference > Critical value)")
        tests = [(f"{name1} vs Benford", t1["percentage"], t1["benford_percentage"], t1["count"])]
        if df2 is not None:
            tests.append((f"{name2} vs Benford", t2["percentage"], t2["benford_percentage"], t2["count"]))
            tests.append((f"{name1} vs {name2}", t1["percentage"], t2["percentage"], t1["count"]))
        rows = []
        for label, a, b, cnt in tests:
            max_diff, crit = ks_test(a.values, b.values, cnt)
            rows.append({"Digits": name, "Comparison": label, "Maximum difference": max_diff,
                         "Critical value (alpha=0.05)": crit,
                         "Result": "Statistically different" if max_diff > crit else "Not different"})
        res = pd.DataFrame(rows)
        st.dataframe(res, use_container_width=True)
        summary_rows.append(res)
        merged.insert(0, "analysis", name)
        all_tables.append(merged)

with tabs[2]:
    desc = {name1: pd.to_numeric(df1[amount_col], errors="coerce").describe()}
    if df2 is not None:
        desc[name2] = pd.to_numeric(df2[amount_col], errors="coerce").describe()
    describe = pd.concat(desc, axis=1)
    st.dataframe(describe.style.format("{:,.2f}"), use_container_width=True)

# ---------- downloads ----------
st.header("Download and share")
c1, c2, c3, c4 = st.columns(4)
c1.download_button("Summary statistics (csv)", describe.to_csv().encode(), "summary_statistics.csv")
c2.download_button("Benford tables (csv)", pd.concat(all_tables).to_csv(index=False).encode(), "benford_tables.csv")
c3.download_button("KS results (csv)", pd.concat(summary_rows).to_csv(index=False).encode(), "ks_results.csv")
c4.download_button("First-digit chart (png)", charts["First digit"], "first_digit_chart.png", "image/png")
st.download_button("First-two-digits chart (png)", charts["First two digits"], "first_two_digits_chart.png", "image/png")
st.caption("Share the app by copying the web address of this page.")