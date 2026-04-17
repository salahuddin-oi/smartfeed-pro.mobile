import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="SmartFeed Pro Mobile", layout="wide")

PROFILE_DEFAULTS = {
    "Layer Peak": {"Total inclusion": 100.0, "CP %": 18.0, "ME kcal/kg": 2800.0, "Lys %": 0.90, "Met %": 0.42, "Ca %": 4.00, "AvP %": 0.45, "Max Fiber %": 7.0, "Target Cost/kg": 0.42},
    "Layer Late": {"Total inclusion": 100.0, "CP %": 16.5, "ME kcal/kg": 2750.0, "Lys %": 0.78, "Met %": 0.36, "Ca %": 4.10, "AvP %": 0.40, "Max Fiber %": 7.5, "Target Cost/kg": 0.39},
    "Broiler Starter": {"Total inclusion": 100.0, "CP %": 22.0, "ME kcal/kg": 3000.0, "Lys %": 1.20, "Met %": 0.52, "Ca %": 1.00, "AvP %": 0.50, "Max Fiber %": 5.5, "Target Cost/kg": 0.48},
    "Broiler Grower": {"Total inclusion": 100.0, "CP %": 20.0, "ME kcal/kg": 3150.0, "Lys %": 1.05, "Met %": 0.45, "Ca %": 0.90, "AvP %": 0.45, "Max Fiber %": 6.0, "Target Cost/kg": 0.46},
    "Custom": {"Total inclusion": 100.0, "CP %": 18.0, "ME kcal/kg": 2800.0, "Lys %": 0.90, "Met %": 0.42, "Ca %": 4.00, "AvP %": 0.45, "Max Fiber %": 7.0, "Target Cost/kg": 0.42},
}
WEIGHTS_DEFAULT = {"Nutrition adequacy": 0.40, "Gut score": 0.25, "Cost score": 0.20, "Energy:protein balance": 0.15}

MARKET_DEFAULTS = pd.DataFrame([
    {"Commodity": "Corn Futures", "Raw Market Quote": 4.4874, "Quote Unit": "USD/bushel corn", "Last Checked": "2026-04-16"},
    {"Commodity": "Soybean Meal Futures", "Raw Market Quote": 331.6, "Quote Unit": "USD/metric ton", "Last Checked": "2026-04-16"},
    {"Commodity": "Soybean Oil", "Raw Market Quote": 0.69, "Quote Unit": "USD/lb", "Last Checked": "2026-04-16"},
    {"Commodity": "Fish Meal", "Raw Market Quote": 1.45, "Quote Unit": "USD/kg", "Last Checked": "2026-04-16"},
])

INGREDIENT_DEFAULTS = pd.DataFrame([
    {"Ingredient": "Corn", "Category": "Energy", "Inclusion %": 58.0, "Price Mode": "Benchmark", "Manual Price/kg": 0.32, "Benchmark Commodity": "Corn Futures", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 45.0, "Max %": 65.0, "CP %": 8.5, "ME kcal/kg": 3350, "Lys %": 0.26, "Met %": 0.18, "Ca %": 0.02, "AvP %": 0.08, "Fiber %": 2.2, "Gut Score 0-10": 5.0, "Digestibility 0-10": 7.0},
    {"Ingredient": "Soybean Meal", "Category": "Protein", "Inclusion %": 24.0, "Price Mode": "Benchmark", "Manual Price/kg": 0.52, "Benchmark Commodity": "Soybean Meal Futures", "Conversion Factor": 1.0, "Premium Adj/kg": 0.05, "Min %": 15.0, "Max %": 35.0, "CP %": 46.0, "ME kcal/kg": 2450, "Lys %": 2.90, "Met %": 0.62, "Ca %": 0.30, "AvP %": 0.25, "Fiber %": 3.5, "Gut Score 0-10": 6.0, "Digestibility 0-10": 8.0},
    {"Ingredient": "Wheat Bran", "Category": "Fiber/Byproduct", "Inclusion %": 5.0, "Price Mode": "Manual", "Manual Price/kg": 0.24, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 0.0, "Max %": 12.0, "CP %": 15.5, "ME kcal/kg": 1700, "Lys %": 0.55, "Met %": 0.25, "Ca %": 0.13, "AvP %": 0.95, "Fiber %": 10.0, "Gut Score 0-10": 6.0, "Digestibility 0-10": 5.0},
    {"Ingredient": "Soybean Oil", "Category": "Fat", "Inclusion %": 3.0, "Price Mode": "Benchmark", "Manual Price/kg": 1.45, "Benchmark Commodity": "Soybean Oil", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 0.0, "Max %": 8.0, "CP %": 0.0, "ME kcal/kg": 8800, "Lys %": 0.00, "Met %": 0.00, "Ca %": 0.00, "AvP %": 0.00, "Fiber %": 0.0, "Gut Score 0-10": 4.0, "Digestibility 0-10": 9.0},
    {"Ingredient": "Limestone", "Category": "Mineral", "Inclusion %": 8.5, "Price Mode": "Manual", "Manual Price/kg": 0.08, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 6.0, "Max %": 10.0, "CP %": 0.0, "ME kcal/kg": 0, "Lys %": 0.00, "Met %": 0.00, "Ca %": 38.00, "AvP %": 0.00, "Fiber %": 0.0, "Gut Score 0-10": 5.0, "Digestibility 0-10": 7.0},
    {"Ingredient": "DCP", "Category": "Mineral", "Inclusion %": 1.0, "Price Mode": "Manual", "Manual Price/kg": 0.75, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 0.0, "Max %": 3.0, "CP %": 0.0, "ME kcal/kg": 0, "Lys %": 0.00, "Met %": 0.00, "Ca %": 23.00, "AvP %": 18.00, "Fiber %": 0.0, "Gut Score 0-10": 5.0, "Digestibility 0-10": 7.0},
])

NUMERIC_COLS = ["Inclusion %","Manual Price/kg","Conversion Factor","Premium Adj/kg","Min %","Max %","CP %","ME kcal/kg","Lys %","Met %","Ca %","AvP %","Fiber %","Gut Score 0-10","Digestibility 0-10"]
MIN_NUTRIENTS = ["CP %","ME kcal/kg","Lys %","Met %","Ca %","AvP %"]

def convert_to_usd_per_kg(raw_quote, unit):
    if pd.isna(raw_quote):
        return np.nan
    if unit == "USD/bushel corn":
        return raw_quote / 25.401
    if unit == "USD/bushel soybeans":
        return raw_quote / 27.216
    if unit == "USD/metric ton":
        return raw_quote / 1000.0
    if unit == "USD/lb":
        return raw_quote * 2.20462
    return raw_quote

def prepare_market_df(df):
    out = df.copy()
    out["Benchmark USD/kg"] = out.apply(lambda r: convert_to_usd_per_kg(r["Raw Market Quote"], r["Quote Unit"]), axis=1)
    return out

def prepare_ingredient_df(ingredients, market):
    bench = dict(zip(market["Commodity"], market["Benchmark USD/kg"]))
    out = ingredients.copy()
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["Ingredient"] = out["Ingredient"].fillna("").astype(str)
    out["Price Mode"] = out["Price Mode"].fillna("Manual")
    out["Benchmark Commodity"] = out["Benchmark Commodity"].fillna("")
    out["Active Price/kg"] = np.where(
        out["Price Mode"].eq("Benchmark"),
        out["Benchmark Commodity"].map(bench).fillna(out["Manual Price/kg"]) * out["Conversion Factor"] + out["Premium Adj/kg"],
        out["Manual Price/kg"]
    )
    return out

def current_targets(selected_profile, custom_targets):
    return custom_targets if selected_profile == "Custom" else PROFILE_DEFAULTS[selected_profile]

def weighted_average(df, col):
    return (df["Inclusion %"] * df[col]).sum() / 100.0

def compute_metrics(ingredients, targets, weights):
    df = ingredients.copy()
    total = df["Inclusion %"].sum()
    if total <= 0:
        total = 1e-9
    cost = weighted_average(df, "Active Price/kg")
    cp = weighted_average(df, "CP %")
    me = weighted_average(df, "ME kcal/kg")
    lys = weighted_average(df, "Lys %")
    met = weighted_average(df, "Met %")
    ca = weighted_average(df, "Ca %")
    avp = weighted_average(df, "AvP %")
    fiber = weighted_average(df, "Fiber %")
    gut = max(0.0, min(10.0, (0.65 * weighted_average(df, "Gut Score 0-10")) + (0.35 * weighted_average(df, "Digestibility 0-10")) - max(0, fiber - targets["Max Fiber %"]) * 0.6))
    adequacy_scores = [
        min(total / targets["Total inclusion"], 1.0) * 100 if targets["Total inclusion"] else 100,
        min(cp / targets["CP %"], 1.0) * 100 if targets["CP %"] else 100,
        min(me / targets["ME kcal/kg"], 1.0) * 100 if targets["ME kcal/kg"] else 100,
        min(lys / targets["Lys %"], 1.0) * 100 if targets["Lys %"] else 100,
        min(met / targets["Met %"], 1.0) * 100 if targets["Met %"] else 100,
        min(ca / targets["Ca %"], 1.0) * 100 if targets["Ca %"] else 100,
        min(avp / targets["AvP %"], 1.0) * 100 if targets["AvP %"] else 100,
        min(targets["Max Fiber %"] / fiber, 1.0) * 100 if fiber > 0 else 100,
    ]
    nutrition_adequacy = round(float(np.mean(adequacy_scores)), 1)
    target_ep = targets["ME kcal/kg"] / targets["CP %"] if targets["CP %"] else 0
    actual_ep = me / cp if cp else 0
    ep_balance = 0 if target_ep == 0 else max(0.0, 100.0 - abs(actual_ep - target_ep) / target_ep * 100.0)
    cost_score = 0 if cost == 0 else min(targets["Target Cost/kg"] / cost, 1.0) * 100.0
    feed_eff = (
        weights["Nutrition adequacy"] * nutrition_adequacy +
        weights["Gut score"] * (gut * 10.0) +
        weights["Cost score"] * cost_score +
        weights["Energy:protein balance"] * ep_balance
    ) / sum(weights.values())
    status = "Good" if nutrition_adequacy >= 95 else ("Watch" if nutrition_adequacy >= 80 else "Fix")
    return {"cost": cost, "cp": cp, "me": me, "lys": lys, "met": met, "ca": ca, "avp": avp, "fiber": fiber, "gut": gut, "feed_eff": feed_eff, "nutrition_adequacy": nutrition_adequacy, "status": status, "total": total}

def nutrient_capacity_analysis(df, targets):
    rows = []
    for nutrient in MIN_NUTRIENTS:
        max_possible = (df["Max %"] * df[nutrient]).sum() / 100.0
        rows.append({"Nutrient": nutrient, "Target": targets[nutrient], "Max achievable": round(max_possible, 4), "Gap": round(max_possible - targets[nutrient], 4), "Status": "OK" if max_possible >= targets[nutrient] else "BLOCKED"})
    fiber_min_possible = (df["Min %"] * df["Fiber %"]).sum() / 100.0
    rows.append({"Nutrient": "Fiber % (max)", "Target": targets["Max Fiber %"], "Max achievable": np.nan, "Gap": round(targets["Max Fiber %"] - fiber_min_possible, 4), "Status": "OK" if fiber_min_possible <= targets["Max Fiber %"] else "BLOCKED"})
    return pd.DataFrame(rows)

def top_contributors(df, column, top_n=3):
    out = df[["Ingredient", "Max %", column]].copy()
    out["Potential"] = out["Max %"] * out[column] / 100.0
    return out.sort_values("Potential", ascending=False).head(top_n)[["Ingredient", "Potential"]]

def diagnose_feasibility(df, targets):
    issues, fixes, summary = [], [], []
    min_total = df["Min %"].sum()
    max_total = df["Max %"].sum()
    if min_total > targets["Total inclusion"]:
        issues.append(f"Minimum inclusions already total {min_total:.2f}%, above the required {targets['Total inclusion']:.2f}%.")
        fixes.append("Reduce one or more Min % values.")
    if max_total < targets["Total inclusion"]:
        issues.append(f"Maximum inclusions total only {max_total:.2f}%, below the required {targets['Total inclusion']:.2f}%.")
        fixes.append("Increase one or more Max % values so the formula can reach 100%.")
    cap = nutrient_capacity_analysis(df, targets)
    blocked = cap[cap["Status"] == "BLOCKED"]
    for _, r in blocked.iterrows():
        if r["Nutrient"] == "Fiber % (max)":
            issues.append("Fiber is blocked. Your forced minimum structure is already too fibrous.")
            names = ", ".join(top_contributors(df, "Fiber %")["Ingredient"].astype(str).tolist())
            fixes.append(f"Reduce high-fiber ingredients such as {names}, or relax the fiber limit.")
        else:
            issues.append(f"{r['Nutrient']} is blocked. Best possible value is {r['Max achievable']:.2f}, below the target of {r['Target']:.2f}.")
            names = ", ".join(top_contributors(df, r["Nutrient"])["Ingredient"].astype(str).tolist())
            fixes.append(f"Increase Max % for strong {r['Nutrient']} sources such as {names}, add a richer ingredient, or lower the target.")
    if blocked.empty and not issues:
        summary.append("No major mathematical blocker found.")
    elif not blocked.empty:
        primary = blocked.iloc[0]["Nutrient"]
        mapping = {
            "CP %": "Protein ceiling is too low.",
            "ME kcal/kg": "Energy density is too low.",
            "Lys %": "Lysine target is too high for this ingredient system.",
            "Met %": "Methionine target is too high for this ingredient system.",
            "Ca %": "Calcium supply cannot reach the target under current bounds.",
            "AvP %": "Available phosphorus supply is too low.",
            "Fiber % (max)": "The formula is being forced to stay too fibrous."
        }
        summary.append(mapping.get(primary, "Ingredient bounds conflict with the targets."))
    return {"summary": summary, "issues": issues, "fixes": list(dict.fromkeys(fixes)), "capacity_table": cap}

def optimize_formula(df, targets):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed. Run: python -m pip install scipy")
    c = df["Active Price/kg"].to_numpy()
    A_ub, b_ub = [], []
    A_ub.append(-df["CP %"].to_numpy()); b_ub.append(-targets["CP %"] * 100)
    A_ub.append(-df["ME kcal/kg"].to_numpy()); b_ub.append(-targets["ME kcal/kg"] * 100)
    A_ub.append(-df["Lys %"].to_numpy()); b_ub.append(-targets["Lys %"] * 100)
    A_ub.append(-df["Met %"].to_numpy()); b_ub.append(-targets["Met %"] * 100)
    A_ub.append(-df["Ca %"].to_numpy()); b_ub.append(-targets["Ca %"] * 100)
    A_ub.append(-df["AvP %"].to_numpy()); b_ub.append(-targets["AvP %"] * 100)
    A_ub.append(df["Fiber %"].to_numpy()); b_ub.append(targets["Max Fiber %"] * 100)
    A_eq = [np.ones(len(df))]
    b_eq = [targets["Total inclusion"]]
    bounds = list(zip(df["Min %"].to_numpy(), df["Max %"].to_numpy()))
    return linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")

def make_excel_download(ingredients, market, targets_df, diagnostic_table):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ingredients.to_excel(writer, index=False, sheet_name="Ingredient_DB")
        market.to_excel(writer, index=False, sheet_name="Market_Prices")
        targets_df.to_excel(writer, index=False, sheet_name="Targets")
        diagnostic_table.to_excel(writer, index=False, sheet_name="Diagnostics")
    output.seek(0)
    return output

def reset_data():
    st.session_state.market_df = MARKET_DEFAULTS.copy()
    st.session_state.ingredients_df = INGREDIENT_DEFAULTS.copy()

if "market_df" not in st.session_state:
    reset_data()
if "selected_profile" not in st.session_state:
    st.session_state.selected_profile = "Broiler Starter"
if "custom_targets" not in st.session_state:
    st.session_state.custom_targets = PROFILE_DEFAULTS["Custom"].copy()
if "weights" not in st.session_state:
    st.session_state.weights = WEIGHTS_DEFAULT.copy()

with st.sidebar:
    st.title("SmartFeed Pro")
    page = st.radio("Workflow", ["Home", "Setup Feed", "Ingredients", "Diagnostic", "Optimize", "Results", "Export"])
    if st.button("Reset demo data"):
        reset_data()
        st.rerun()

market_now = prepare_market_df(st.session_state.market_df)
ing_now = prepare_ingredient_df(st.session_state.ingredients_df, market_now)
targets_now = current_targets(st.session_state.selected_profile, st.session_state.custom_targets)
metrics_now = compute_metrics(ing_now, targets_now, st.session_state.weights)
diagnosis_now = diagnose_feasibility(ing_now, targets_now)

if page == "Home":
    st.title("SmartFeed Pro")
    st.caption("Mobile-first poultry feed decision engine")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", metrics_now["status"])
    c2.metric("Cost/kg", f"${metrics_now['cost']:.4f}")
    c3.metric("Gut Score", f"{metrics_now['gut']:.2f}/10")
    d1, d2 = st.columns(2)
    d1.metric("Feed Efficiency", f"{metrics_now['feed_eff']:.1f}/100")
    d2.metric("Nutrition Adequacy", f"{metrics_now['nutrition_adequacy']:.1f}%")
    if diagnosis_now["summary"]:
        st.info(diagnosis_now["summary"][0])
    st.write("Use the left menu: Setup Feed → Ingredients → Diagnostic → Optimize → Results")

elif page == "Setup Feed":
    st.subheader("Setup Feed")
    profile = st.selectbox("Diet profile", list(PROFILE_DEFAULTS.keys()), index=list(PROFILE_DEFAULTS.keys()).index(st.session_state.selected_profile))
    st.session_state.selected_profile = profile
    if st.button("Load profile into editable custom targets"):
        st.session_state.custom_targets = PROFILE_DEFAULTS[profile].copy()
        st.session_state.selected_profile = "Custom"
        st.rerun()
    targets = st.session_state.custom_targets.copy() if st.session_state.selected_profile == "Custom" else PROFILE_DEFAULTS[st.session_state.selected_profile].copy()
    cols = st.columns(2)
    for i, k in enumerate(targets.keys()):
        with cols[i % 2]:
            step = 0.01 if "%" in k or "Cost" in k else 1.0
            targets[k] = st.number_input(k, value=float(targets[k]), step=step, key=f"target_{k}")
    st.session_state.custom_targets = targets.copy()

elif page == "Ingredients":
    st.subheader("Ingredients")
    with st.expander("Quick add ingredient", expanded=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Ingredient name")
        category = c2.selectbox("Category", ["Energy", "Protein", "Fat", "Mineral", "Fiber/Byproduct", "Additive"])
        c3, c4 = st.columns(2)
        cp = c3.number_input("CP %", min_value=0.0, value=0.0, step=0.1)
        me = c4.number_input("ME kcal/kg", min_value=0.0, value=0.0, step=10.0)
        c5, c6 = st.columns(2)
        price = c5.number_input("Manual Price/kg", min_value=0.0, value=0.0, step=0.01)
        max_pct = c6.number_input("Max %", min_value=0.0, value=10.0, step=0.1)
        if st.button("Add ingredient"):
            if name.strip():
                new_row = {"Ingredient": name.strip(), "Category": category, "Inclusion %": 0.0, "Price Mode": "Manual", "Manual Price/kg": price, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.0, "Min %": 0.0, "Max %": max_pct, "CP %": cp, "ME kcal/kg": me, "Lys %": 0.0, "Met %": 0.0, "Ca %": 0.0, "AvP %": 0.0, "Fiber %": 0.0, "Gut Score 0-10": 5.0, "Digestibility 0-10": 5.0}
                st.session_state.ingredients_df = pd.concat([st.session_state.ingredients_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success("Ingredient added.")
                st.rerun()
            else:
                st.error("Enter an ingredient name.")
    with st.expander("Advanced ingredient editor", expanded=False):
        commodity_options = [""] + market_now["Commodity"].tolist()
        edited = st.data_editor(
            st.session_state.ingredients_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Price Mode": st.column_config.SelectboxColumn(options=["Manual", "Benchmark"]),
                "Benchmark Commodity": st.column_config.SelectboxColumn(options=commodity_options),
                "Gut Score 0-10": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.1),
                "Digestibility 0-10": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.1),
            },
            hide_index=True,
            key="ingredient_editor"
        )
        st.session_state.ingredients_df = edited.copy()

elif page == "Diagnostic":
    st.subheader("Smart Analysis")
    if diagnosis_now["summary"]:
        st.info(diagnosis_now["summary"][0])
    if diagnosis_now["issues"]:
        st.markdown("### What is wrong")
        for issue in diagnosis_now["issues"]:
            st.error(issue)
    else:
        st.success("No major mathematical blocker found.")
    if diagnosis_now["fixes"]:
        st.markdown("### What to do next")
        for i, fix in enumerate(diagnosis_now["fixes"], start=1):
            st.write(f"{i}. {fix}")
    blocked = diagnosis_now["capacity_table"][diagnosis_now["capacity_table"]["Status"] == "BLOCKED"].copy()
    if not blocked.empty:
        chart = alt.Chart(blocked).mark_bar().encode(x="Nutrient:N", y="Gap:Q", tooltip=["Nutrient", "Target", "Max achievable", "Gap"])
        st.altair_chart(chart, use_container_width=True)
    st.dataframe(diagnosis_now["capacity_table"], use_container_width=True, hide_index=True)

elif page == "Optimize":
    st.subheader("Optimize Feed")
    st.write("Use this only after fixing the main diagnostic blockers.")
    if not SCIPY_AVAILABLE:
        st.warning("SciPy is not installed. Run: python -m pip install scipy")
    else:
        if st.button("Fix My Formula"):
            try:
                result = optimize_formula(ing_now, targets_now)
                if result.success:
                    st.session_state.ingredients_df["Inclusion %"] = result.x
                    st.success("Optimization solved.")
                    st.rerun()
                else:
                    st.error("Optimization failed.")
                    st.code(str(result.message))
            except Exception as e:
                st.error(f"Optimizer error: {e}")

elif page == "Results":
    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", metrics_now["status"])
    c2.metric("Cost/kg", f"${metrics_now['cost']:.4f}")
    c3.metric("Gut Score", f"{metrics_now['gut']:.2f}/10")
    c4.metric("Feed Efficiency", f"{metrics_now['feed_eff']:.1f}/100")
    mix = ing_now[ing_now["Inclusion %"] > 0][["Ingredient", "Inclusion %"]].sort_values("Inclusion %", ascending=False)
    if not mix.empty:
        pie = alt.Chart(mix).mark_arc().encode(theta="Inclusion %:Q", color="Ingredient:N", tooltip=["Ingredient", "Inclusion %"])
        st.altair_chart(pie, use_container_width=True)
    nutrient_view = pd.DataFrame([
        {"Metric": "CP %", "Achieved": metrics_now["cp"], "Target": targets_now["CP %"]},
        {"Metric": "ME kcal/kg", "Achieved": metrics_now["me"], "Target": targets_now["ME kcal/kg"]},
        {"Metric": "Lys %", "Achieved": metrics_now["lys"], "Target": targets_now["Lys %"]},
        {"Metric": "Met %", "Achieved": metrics_now["met"], "Target": targets_now["Met %"]},
        {"Metric": "Ca %", "Achieved": metrics_now["ca"], "Target": targets_now["Ca %"]},
        {"Metric": "AvP %", "Achieved": metrics_now["avp"], "Target": targets_now["AvP %"]},
        {"Metric": "Fiber %", "Achieved": metrics_now["fiber"], "Target": targets_now["Max Fiber %"]},
    ])
    st.dataframe(nutrient_view, use_container_width=True, hide_index=True)

elif page == "Export":
    st.subheader("Export")
    targets_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in targets_now.items()])
    xlsx = make_excel_download(ing_now, market_now, targets_df, diagnosis_now["capacity_table"])
    st.download_button("Download workbook snapshot", data=xlsx, file_name="smartfeed_mobile_snapshot.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download ingredient DB as CSV", data=ing_now.to_csv(index=False).encode("utf-8"), file_name="ingredient_db.csv", mime="text/csv")
