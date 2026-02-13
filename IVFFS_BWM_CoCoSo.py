import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Core Helper Functions for IVIFN Calculations ---
# These functions are used by both the BWM and CoCoSo models.

def parse_ivifn(ivifn_str):
    """Parses a string like '([0.70, 0.80], [0.10, 0.20])' into a dictionary."""
    try:
        s = ivifn_str.strip().replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        parts = [float(p.strip()) for p in s.split(',')]
        if len(parts) == 4:
            return {'mu_a': parts[0], 'mu_b': parts[1], 'v_a': parts[2], 'v_b': parts[3]}
        return None
    except (ValueError, IndexError):
        return None

def score_function(ivifn):
    """Calculates the improved accuracy score for an IVIFN (Equation 9)."""
    if not isinstance(ivifn, dict): return 0
    mu_a, mu_b, v_a, v_b = ivifn['mu_a'], ivifn['mu_b'], ivifn['v_a'], ivifn['v_b']
    pi_a = 1 - mu_a - v_a
    pi_b = 1 - mu_b - v_b
    # CORRECTED: The formula now directly uses pi_a and pi_b as per the source document.
    return (mu_a + mu_b * pi_a + mu_b + mu_a * pi_b) / 2

def ivifwa(ivifns, weights):
    """Performs the IVIF-Weighted Averaging (IVIFWA) operation."""
    if not ivifns or not weights or len(ivifns) != len(weights): return None
    if not np.isclose(sum(weights), 1.0):
        st.error(f"IVIFWA Error: Weights must sum to 1. Got sum: {sum(weights):.4f}")
        return None
    
    term1 = 1 - np.prod([(1 - ivifn['mu_a'])**w for ivifn, w in zip(ivifns, weights)])
    term2 = 1 - np.prod([(1 - ivifn['mu_b'])**w for ivifn, w in zip(ivifns, weights)])
    term3 = np.prod([ivifn['v_a']**w for ivifn, w in zip(ivifns, weights)])
    term4 = np.prod([ivifn['v_b']**w for ivifn, w in zip(ivifns, weights)])
    return {'mu_a': term1, 'mu_b': term2, 'v_a': term3, 'v_b': term4}

def ivifwg(ivifns, weights):
    """Performs the IVIF-Weighted Geometric (IVIFWG) operation."""
    if not ivifns or not weights or len(ivifns) != len(weights): return None
    if not np.isclose(sum(weights), 1.0):
        st.error(f"IVIFWG Error: Weights must sum to 1. Got sum: {sum(weights):.4f}")
        return None
        
    term1 = np.prod([ivifn['mu_a']**w for ivifn, w in zip(ivifns, weights)])
    term2 = np.prod([ivifn['mu_b']**w for ivifn, w in zip(ivifns, weights)])
    term3 = 1 - np.prod([(1 - ivifn['v_a'])**w for ivifn, w in zip(ivifns, weights)])
    term4 = 1 - np.prod([(1 - ivifn['v_b'])**w for ivifn, w in zip(ivifns, weights)])
    return {'mu_a': term1, 'mu_b': term2, 'v_a': term3, 'v_b': term4}

# --- 2. Main Application UI ---

st.set_page_config(layout="wide", page_title="IVIFN MCDM Toolkit")
st.title("🛠️ IVIFN Multi-Criteria Decision-Making Toolkit")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a Model", ["IVIFN-BWM Assistant", "IVIFN-CoCoSo Modeler"])
st.sidebar.markdown("---")
st.sidebar.info("Select a model to begin your analysis. Each model runs independently.")

# --- 3. IVIFN-BWM Assistant Module ---

if page == "IVIFN-BWM Assistant":
    st.header("🗂️ IVIFN Best-Worst Method (BWM) Assistant")
    st.markdown("This tool helps with data preparation, LINGO model generation, and final weight calculation for criteria.")

    BWM_LINGUISTIC_SCALE = {
        "Equally important (EI)": "([1.00, 1.00], [0.00, 0.00])",
        "Weakly important (WI)": "([0.50, 0.60], [0.30, 0.40])",
        "Strongly important (SI)": "([0.60, 0.70], [0.20, 0.30])",
        "Very important (VI)": "([0.70, 0.80], [0.10, 0.20])",
        "Absolutely important (AI)": "([0.80, 0.90], [0.10, 0.20])"
    }
    BWM_LINGUISTIC_VALUES = {k: parse_ivifn(v) for k, v in BWM_LINGUISTIC_SCALE.items()}
    BWM_ABBR_MAP = {k.split('(')[1][:-1]: k for k in BWM_LINGUISTIC_SCALE.keys()}

    with st.expander("Show BWM Linguistic Scale Reference", expanded=True):
        st.table(pd.DataFrame(list(BWM_LINGUISTIC_SCALE.items()), columns=['Linguistic Attribute', 'IVIFN Value']))

    st.subheader("Step 1: Define Criteria & Select Best/Worst")
    criteria_input = st.text_input("Enter criteria, separated by commas", "Driving Force, Pressure, State, Impact, Response", key="bwm_criteria")
    criteria = [c.strip() for c in criteria_input.split(',') if c.strip()]
    
    if len(criteria) < 2:
        st.warning("Please enter at least two criteria.")
    else:
        crit_map = {name: i + 1 for i, name in enumerate(criteria)}
        c1, c2 = st.columns(2)
        best_criterion = c1.selectbox("Select the BEST criterion", options=criteria, index=len(criteria)-1, key="bwm_best")
        worst_criterion = c2.selectbox("Select the WORST criterion", options=criteria, index=2, key="bwm_worst")
        
        if best_criterion == worst_criterion:
            st.error("Best and Worst criteria cannot be the same.")
        else:
            st.subheader("Step 2: Formulate Comparison Matrices")
            df_best_data = [{"Criterion": c, "Linguistic Term": "VI"} for c in criteria if c != best_criterion]
            df_worst_data = [{"Criterion": c, "Linguistic Term": "SI"} for c in criteria if c != worst_criterion]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Best-to-Others (Best = _{best_criterion}_)**")
                edited_df_best = st.data_editor(pd.DataFrame(df_best_data), 
                    column_config={"Linguistic Term": st.column_config.SelectboxColumn("Linguistic Term", options=list(BWM_ABBR_MAP.keys()))}, 
                    hide_index=True, key="editor_best")
            with col2:
                st.markdown(f"**Others-to-Worst (Worst = _{worst_criterion}_)**")
                edited_df_worst = st.data_editor(pd.DataFrame(df_worst_data),
                    column_config={"Linguistic Term": st.column_config.SelectboxColumn("Linguistic Term", options=list(BWM_ABBR_MAP.keys()))},
                    hide_index=True, key="editor_worst")
            
            st.subheader("Step 3: Generate LINGO Optimization Model")
            if st.button("Generate LINGO Model Code", type="primary", key="bwm_generate_lingo"):
                b_idx, w_idx, n_crit = crit_map[best_criterion], crit_map[worst_criterion], len(criteria)
                model_str = ["model:", "min=z;"]
                for _, row in edited_df_best.iterrows():
                    full_term = BWM_ABBR_MAP[row['Linguistic Term']]
                    ivifn = BWM_LINGUISTIC_VALUES[full_term]
                    j_idx = crit_map[row['Criterion']]
                    model_str.extend([f"@abs(x{b_idx}L + y{j_idx}L - x{b_idx}L * y{j_idx}L - {ivifn['mu_a']}) <= z;", f"@abs(x{b_idx}U + y{j_idx}U - x{b_idx}U * y{j_idx}U - {ivifn['mu_b']}) <= z;", f"@abs(y{b_idx}L + x{j_idx}L - {ivifn['v_a']}) <= z;", f"@abs(y{b_idx}U + x{j_idx}U - {ivifn['v_b']}) <= z;"])
                for _, row in edited_df_worst.iterrows():
                    full_term = BWM_ABBR_MAP[row['Linguistic Term']]
                    ivifn = BWM_LINGUISTIC_VALUES[full_term]
                    j_idx = crit_map[row['Criterion']]
                    model_str.extend([f"@abs(x{j_idx}L + y{w_idx}L - x{j_idx}L * y{w_idx}L - {ivifn['mu_a']}) <= z;", f"@abs(x{j_idx}U + y{w_idx}U - x{j_idx}U * y{w_idx}U - {ivifn['mu_b']}) <= z;", f"@abs(y{j_idx}L + x{w_idx}L - {ivifn['v_a']}) <= z;", f"@abs(y{j_idx}U + x{w_idx}U - {ivifn['v_b']}) <= z;"])
                model_str.append(" + ".join([f"(x{i}L + x{i}U - y{i}L - y{i}U)" for i in range(1, n_crit + 1)]) + " = 2;")
                for i in range(1, n_crit + 1):
                    model_str.extend([f"x{i}U + y{i}U <= 1;", f"x{i}L <= x{i}U;", f"y{i}L <= y{i}U;", f"x{i}L >= 0;", f"y{i}L >= 0;"])
                model_str.append("\nend")
                st.code("\n".join(model_str), language='lingo')

            st.subheader("Step 4: Input Optimal Weights & Calculate Final Ranking")
            num_experts = st.number_input("How many expert groups?", min_value=1, value=1, key="bwm_num_experts")
            
            st.markdown("**Expert Weights**")
            expert_w = []
            if num_experts > 1:
                cols = st.columns(num_experts)
                for i in range(num_experts):
                    with cols[i]:
                        weight = st.number_input(f"Weight for Expert {i+1}", min_value=0.0, max_value=1.0, value=1.0/num_experts, step=0.05, format="%.2f", key=f"bwm_expert_weight_{i}")
                        expert_w.append(weight)
                if not np.isclose(sum(expert_w), 1.0):
                    st.error(f"Expert weights must sum to 1.0. Current sum is {sum(expert_w):.2f}.")
                    st.stop()
            else:
                expert_w = [1.0]

            expert_weights_data = {}
            for i in range(num_experts):
                st.markdown(f"--- \n#### Optimal Weights for Expert Group {i+1}")
                default_val = '\n'.join([f"{c}: ([0.5, 0.6], [0.3, 0.4])" for c in criteria])
                weights_input = st.text_area(f"Paste weights for Group {i+1}", height=150, value=default_val, key=f"bwm_weights_{i}")
                expert_weights_data[i] = {c.strip(): parse_ivifn(v) for line in weights_input.strip().split('\n') if ':' in line for c, v in [line.split(':', 1)] if c.strip() in criteria}

            if st.button("Calculate Final Aggregated Weights", key="bwm_calculate"):
                if all(d and set(d.keys()) == set(criteria) for d in expert_weights_data.values()):
                    agg_weights = {c: ivifwa([expert_weights_data[i][c] for i in range(num_experts)], expert_w) for c in criteria}
                    
                    st.markdown("#### Aggregated IVIFN Weights (ω_j)")
                    agg_df_data = {
                        'Criterion': list(agg_weights.keys()),
                        '[μ_a, μ_b]': [f"[{w['mu_a']:.4f}, {w['mu_b']:.4f}]" for w in agg_weights.values()],
                        '[ν_a, ν_b]': [f"[{w['v_a']:.4f}, {w['v_b']:.4f}]" for w in agg_weights.values()]
                    }
                    st.table(pd.DataFrame(agg_df_data))

                    scores = {c: score_function(w) for c, w in agg_weights.items()}
                    total_score = sum(scores.values())
                    final_data = [{'Criterion': c, 'Score S(ω_j)': s, 'Final Crisp Weight': s / total_score if total_score > 0 else 0} for c, s in scores.items()]
                    final_df = pd.DataFrame(final_data).sort_values(by='Final Crisp Weight', ascending=False).reset_index(drop=True)
                    final_df['Rank'] = final_df.index + 1
                    
                    st.markdown("#### Final Results & Ranking")
                    st.dataframe(final_df.style.format({'Score S(ω_j)': '{:.4f}', 'Final Crisp Weight': '{:.4f}'}), use_container_width=True)
                else:
                    st.error("Mismatch in criteria for one or more expert groups. Please check pasted weights.")

# --- 4. IVIFN-CoCoSo Modeler Module ---
elif page == "IVIFN-CoCoSo Modeler":
    st.header("📊 IVIFN-CoCoSo Modeler")
    st.markdown("This tool implements the IVIFN Combined Compromise Solution (CoCoSo) method to rank alternatives.")

    # Updated CoCoSo linguistic scale as requested
    COCOSO_LINGUISTIC_SCALE = {
        "Extremely bad (EB)": "([0.05, 0.10], [0.80, 0.90])",
        "Very bad (VB)": "([0.10, 0.20], [0.70, 0.80])",
        "Bad (B)": "([0.20, 0.30], [0.60, 0.70])",
        "Medium bad (MB)": "([0.30, 0.40], [0.50, 0.60])",
        "Medium (M)": "([0.40, 0.50], [0.40, 0.50])",
        "Medium good (MG)": "([0.50, 0.60], [0.30, 0.40])",
        "Good (G)": "([0.60, 0.70], [0.20, 0.30])",
        "Very good (VG)": "([0.70, 0.80], [0.10, 0.20])",
        "Extremely good (EG)": "([0.80, 0.90], [0.05, 0.10])"
    }
    
    COCOSO_LINGUISTIC_VALUES = {k: parse_ivifn(v) for k, v in COCOSO_LINGUISTIC_SCALE.items()}
    COCOSO_ABBR_MAP = {k.split('(')[1][:-1]: k for k in COCOSO_LINGUISTIC_SCALE.keys()}

    with st.expander("Show CoCoSo Linguistic Scale Reference"):
        st.table(pd.DataFrame(list(COCOSO_LINGUISTIC_SCALE.items()), columns=['Linguistic Attribute', 'IVIFN Value']))

    st.subheader("Step 1: Define Alternatives, Criteria, and Weights")
    c1, c2 = st.columns(2)
    alts_in = c1.text_input("Enter Alternatives (comma-separated)", "Tech A, Tech B, Tech C", key="cocoso_alts")
    crits_in = c2.text_input("Enter Criteria (comma-separated)", "Cost, Efficiency, Safety", key="cocoso_crits")
    alternatives = [a.strip() for a in alts_in.split(',') if a.strip()]
    criteria = [c.strip() for c in crits_in.split(',') if c.strip()]

    if not alternatives or not criteria:
        st.warning("Please define at least one alternative and one criterion.")
    else:
        st.markdown("**Criteria Details & Weights**")
        if 'cocoso_crit_df' not in st.session_state or set(st.session_state.cocoso_crit_df['Criterion']) != set(criteria):
            w = [round(1/len(criteria), 4)] * len(criteria)
            if len(criteria) > 0:
                w[-1] = 1.0 - sum(w[:-1])
            st.session_state.cocoso_crit_df = pd.DataFrame({'Criterion': criteria, 'Type': ['Cost'] + ['Benefit'] * (len(criteria) - 1), 'Weight (ωj)': w})
        
        edited_crit_df = st.data_editor(st.session_state.cocoso_crit_df, hide_index=True, key="cocoso_crit_editor")
        criteria_weights = edited_crit_df['Weight (ωj)'].tolist()
        criteria_types = edited_crit_df['Type'].tolist()
        if not np.isclose(sum(criteria_weights), 1.0):
            st.error(f"Criteria weights must sum to 1.0. Current sum: {sum(criteria_weights):.4f}.")
            st.stop()

        st.subheader("Step 2: Expert Evaluation of Alternatives")
        num_experts = st.number_input("Enter the number of experts", min_value=1, value=2, key="cocoso_experts")
        
        st.markdown("**Expert Weights**")
        expert_weights = []
        if num_experts > 1:
            cols = st.columns(num_experts)
            for i in range(num_experts):
                with cols[i]:
                    weight = st.number_input(f"Weight E{i+1}", min_value=0.0, max_value=1.0, value=1.0/num_experts, step=0.05, format="%.2f", key=f"cocoso_exp_w_{i}")
                    expert_weights.append(weight)
            if not np.isclose(sum(expert_weights), 1.0):
                st.error(f"Expert weights must sum to 1.0. Current sum: {sum(expert_weights):.2f}.")
                st.stop()
        else:
            expert_weights = [1.0]

        if 'cocoso_expert_dfs' not in st.session_state: st.session_state.cocoso_expert_dfs = {}
        if len(st.session_state.cocoso_expert_dfs) != num_experts or (num_experts > 0 and (set(st.session_state.cocoso_expert_dfs[0].index) != set(alternatives) or set(st.session_state.cocoso_expert_dfs[0].columns) != set(criteria))):
            st.session_state.cocoso_expert_dfs = {i: pd.DataFrame("M", index=alternatives, columns=criteria) for i in range(num_experts)}
            
        expert_tabs = st.tabs([f"Expert {i+1}" for i in range(num_experts)])
        for i, tab in enumerate(expert_tabs):
            with tab:
                st.session_state.cocoso_expert_dfs[i] = st.data_editor(st.session_state.cocoso_expert_dfs[i], column_config={c: st.column_config.SelectboxColumn(c, options=list(COCOSO_ABBR_MAP.keys())) for c in criteria}, key=f"cocoso_editor_{i}")

        st.subheader("Step 3: Run CoCoSo Calculation")
        if st.button("Calculate Final Ranking", type="primary", key="cocoso_calculate"):
            with st.spinner('Calculating...'):
                st.markdown("#### 3.1 - Aggregated IVIFN Decision Matrix")
                agg_matrix = pd.DataFrame(index=alternatives, columns=criteria, dtype=object)
                for alt in alternatives:
                    for crit in criteria:
                        ivifns = [COCOSO_LINGUISTIC_VALUES[COCOSO_ABBR_MAP[st.session_state.cocoso_expert_dfs[i].loc[alt, crit]]] for i in range(num_experts)]
                        agg_matrix.at[alt, crit] = ivifwa(ivifns, expert_weights)
                
                def format_ivifn_df(df):
                    return df.applymap(lambda x: f"([{x['mu_a']:.4f}, {x['mu_b']:.4f}], [{x['v_a']:.4f}, {x['v_b']:.4f}])" if isinstance(x, dict) else "N/A")
                st.table(format_ivifn_df(agg_matrix))

                st.markdown("#### 3.2 - Normalized Decision Matrix")
                norm_matrix = agg_matrix.copy()
                for j, crit in enumerate(criteria):
                    if criteria_types[j] == 'Cost':
                        for i, alt in enumerate(alternatives):
                            ivifn = norm_matrix.at[alt, crit]
                            if isinstance(ivifn, dict): norm_matrix.at[alt, crit] = {'mu_a': ivifn['v_a'], 'mu_b': ivifn['v_b'], 'v_a': ivifn['mu_a'], 'v_b': ivifn['mu_b']}
                st.table(format_ivifn_df(norm_matrix))

                st.markdown("#### 3.3 - Weighted Sum (SBi) and Product (PBi) Scores")
                results = [{'Alternative': alt, 'SBi': ivifwa(norm_matrix.loc[alt].tolist(), criteria_weights), 'PBi': ivifwg(norm_matrix.loc[alt].tolist(), criteria_weights)} for alt in alternatives]
                results_df = pd.DataFrame(results)
                
                # --- NEW: Display SBi and PBi as IVIFNs ---
                st.markdown("##### SBi and PBi as IVIFNs")
                sbi_pbi_ivifn_df = pd.DataFrame({
                    'Alternative': results_df['Alternative'],
                    'SBi': results_df['SBi'].apply(lambda x: f"([{x['mu_a']:.4f}, {x['mu_b']:.4f}], [{x['v_a']:.4f}, {x['v_b']:.4f}])" if isinstance(x, dict) else "N/A"),
                    'PBi': results_df['PBi'].apply(lambda x: f"([{x['mu_a']:.4f}, {x['mu_b']:.4f}], [{x['v_a']:.4f}, {x['v_b']:.4f}])" if isinstance(x, dict) else "N/A")
                })
                st.table(sbi_pbi_ivifn_df)
                # --- END NEW SECTION ---

                results_df['SBi_score'] = results_df['SBi'].apply(score_function)
                results_df['PBi_score'] = results_df['PBi'].apply(score_function)

                st.markdown("##### SBi and PBi as Crisp Scores")
                st.dataframe(results_df[['Alternative', 'SBi_score', 'PBi_score']].style.format(precision=4))

                st.markdown("#### 3.4 - Final Ranking")
                s, p = results_df['SBi_score'], results_df['PBi_score']
                results_df['pia'] = (s + p) / (s + p).sum() if (s + p).sum() != 0 else 0
                results_df['pib'] = (s / s.min() if s.min() != 0 else 0) + (p / p.min() if p.min() != 0 else 0)
                tau = 0.5
                denom_pic = (tau * s.max() + (1 - tau) * p.max())
                results_df['pic'] = (tau * s + (1 - tau) * p) / denom_pic if denom_pic != 0 else 0
                results_df['pi'] = ((results_df['pia'] * results_df['pib'] * results_df['pic'])**(1/3)) + ((results_df['pia'] + results_df['pib'] + results_df['pic']) / 3)
                results_df['Rank'] = results_df['pi'].rank(ascending=False, method='min').astype(int)

                final_table = results_df[['Alternative', 'pia', 'pib', 'pic', 'pi', 'Rank']].sort_values(by='Rank')
                st.dataframe(final_table.style.format(precision=4), use_container_width=True)
