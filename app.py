import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="DC Circuits Analyzer", layout="wide")
st.title("⚡ DC Circuits Analysis Suite - MNA Solver")
st.markdown("---")

# -------------------------------------------------------------------
# 2. INITIALIZE SESSION STATE
# -------------------------------------------------------------------
if 'df' not in st.session_state:
    default_data = {
        "Ref": ["V1", "R1", "R2", "R3"],
        "Type": ["V", "R", "R", "R"],
        "Node1": [1, 1, 2, 0],
        "Node2": [0, 2, 0, 3],
        "Value": ["12", "1000", "2000", "3000"],  # <<<<<<< تم التحويل إلى نصوص
        "Controlling": ["", "", "", ""],
        "Gain": [0.0, 0.0, 0.0, 0.0]
    }
    st.session_state.df = pd.DataFrame(default_data)
    st.session_state.solution = None
    st.session_state.thevenin = None

# -------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------
def parse_component_value(val_str):
    if isinstance(val_str, (int, float)):
        return float(val_str)
    val_str = str(val_str).strip().lower()
    multipliers = {'k': 1e3, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'meg': 1e6, 'g': 1e9}
    for suffix, mult in multipliers.items():
        if val_str.endswith(suffix):
            try:
                return float(val_str[:-len(suffix)]) * mult
            except:
                return 0.0
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def solve_mna(elements_df):
    def build_mna(elements_df):
        nodes = set()
        for _, row in elements_df.iterrows():
            nodes.add(int(row['Node1']))
            nodes.add(int(row['Node2']))
        nodes.discard(0)
        node_list = sorted(list(nodes))
        n = len(node_list)
        node_index = {node: i for i, node in enumerate(node_list)}
        
        G = np.zeros((n, n))
        I = np.zeros(n)
        voltage_equations = []
        elements = elements_df.to_dict('records')
        
        for el in elements:
            typ = str(el['Type']).upper()
            n1 = int(el['Node1']); n2 = int(el['Node2'])
            val = parse_component_value(el['Value'])
            ctrl = str(el['Controlling']).strip()
            gain = parse_component_value(el['Gain'])
            idx1 = node_index.get(n1, -1)
            idx2 = node_index.get(n2, -1)
            
            if typ == 'R':
                g = 1.0 / val if val != 0 else 1e12
                if idx1 != -1: G[idx1][idx1] += g
                if idx2 != -1: G[idx2][idx2] += g
                if idx1 != -1 and idx2 != -1:
                    G[idx1][idx2] -= g; G[idx2][idx1] -= g
            elif typ == 'I':
                if idx1 != -1: I[idx1] += val
                if idx2 != -1: I[idx2] -= val
            elif typ == 'V':
                voltage_equations.append({'n1': idx1, 'n2': idx2, 'val': val})
            elif typ == 'VCVS':
                if ctrl:
                    ctrl_row = elements_df[elements_df['Ref'] == ctrl]
                    if not ctrl_row.empty:
                        c_el = ctrl_row.iloc[0]
                        c_n1 = int(c_el['Node1']); c_n2 = int(c_el['Node2'])
                        c_idx1 = node_index.get(c_n1, -1); c_idx2 = node_index.get(c_n2, -1)
                        voltage_equations.append({
                            'n1': idx1, 'n2': idx2, 'val': 0,
                            'ctrl_n1': c_idx1, 'ctrl_n2': c_idx2, 'gain': gain, 'type': 'VCVS'
                        })
            elif typ == 'CCVS':
                if ctrl:
                    ctrl_row = elements_df[elements_df['Ref'] == ctrl]
                    if not ctrl_row.empty:
                        c_el = ctrl_row.iloc[0]
                        c_val = parse_component_value(c_el['Value'])
                        c_n1 = int(c_el['Node1']); c_n2 = int(c_el['Node2'])
                        c_idx1 = node_index.get(c_n1, -1); c_idx2 = node_index.get(c_n2, -1)
                        if c_val != 0:
                            trans_gain = gain / c_val
                            voltage_equations.append({
                                'n1': idx1, 'n2': idx2, 'val': 0,
                                'ctrl_n1': c_idx1, 'ctrl_n2': c_idx2, 'gain': trans_gain, 'type': 'CCVS'
                            })
            elif typ == 'VCCS':
                if ctrl:
                    ctrl_row = elements_df[elements_df['Ref'] == ctrl]
                    if not ctrl_row.empty:
                        c_el = ctrl_row.iloc[0]
                        c_n1 = int(c_el['Node1']); c_n2 = int(c_el['Node2'])
                        c_idx1 = node_index.get(c_n1, -1); c_idx2 = node_index.get(c_n2, -1)
                        if idx1 != -1:
                            if c_idx1 != -1: G[idx1][c_idx1] += gain
                            if c_idx2 != -1: G[idx1][c_idx2] -= gain
                        if idx2 != -1:
                            if c_idx1 != -1: G[idx2][c_idx1] -= gain
                            if c_idx2 != -1: G[idx2][c_idx2] += gain
            elif typ == 'CCCS':
                if ctrl:
                    ctrl_row = elements_df[elements_df['Ref'] == ctrl]
                    if not ctrl_row.empty:
                        c_el = ctrl_row.iloc[0]
                        c_val = parse_component_value(c_el['Value'])
                        c_n1 = int(c_el['Node1']); c_n2 = int(c_el['Node2'])
                        c_idx1 = node_index.get(c_n1, -1); c_idx2 = node_index.get(c_n2, -1)
                        if c_val != 0 and c_idx1 != -1 and c_idx2 != -1:
                            gm = gain / c_val
                            if idx1 != -1:
                                G[idx1][c_idx1] += gm; G[idx1][c_idx2] -= gm
                            if idx2 != -1:
                                G[idx2][c_idx1] -= gm; G[idx2][c_idx2] += gm

        m = len(voltage_equations)
        total_size = n + m
        A = np.zeros((total_size, total_size))
        B = np.zeros(total_size)
        A[0:n, 0:n] = G
        B[0:n] = I

        for eq_idx, eq in enumerate(voltage_equations):
            row = n + eq_idx
            n1 = eq.get('n1', -1); n2 = eq.get('n2', -1)
            if n1 != -1:
                A[n1][row] = 1; A[row][n1] = 1
            if n2 != -1:
                A[n2][row] = -1; A[row][n2] = -1
            
            if eq.get('type') == 'VCVS':
                gain = eq.get('gain', 1)
                if eq.get('ctrl_n1', -1) != -1:
                    A[row][eq['ctrl_n1']] -= gain
                if eq.get('ctrl_n2', -1) != -1:
                    A[row][eq['ctrl_n2']] += gain
            elif eq.get('type') == 'CCVS':
                gain = eq.get('gain', 1)
                if eq.get('ctrl_n1', -1) != -1:
                    A[row][eq['ctrl_n1']] -= gain
                if eq.get('ctrl_n2', -1) != -1:
                    A[row][eq['ctrl_n2']] += gain
            else:
                if n1 != -1: A[row][n1] = 1
                if n2 != -1: A[row][n2] = -1
                B[row] = eq.get('val', 0)

        try:
            X = np.linalg.solve(A, B)
            node_voltages = {node: X[node_index[node]] for node in node_list}
            currents = {f"I_Vsrc_{eq_idx}": X[n + eq_idx] for eq_idx in range(m)}
            return node_voltages, currents, A, B, X, node_list
        except np.linalg.LinAlgError:
            return None, None, A, B, None, node_list

    return build_mna(elements_df)

# -------------------------------------------------------------------
# 4. UI: EDITOR & CONTROLS
# -------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("📝 Circuit Netlist (Interactive Table)")
    st.caption("Use '0' for Ground. For Dependent Sources, fill 'Controlling' (Ref) and 'Gain'.")
    
    # <<<<<<< التصحيح: نأخذ نسخة من البيانات ونتأكد أن عمود القيمة نصي >>>>>>>
    df_to_edit = st.session_state.df.copy()
    df_to_edit['Value'] = df_to_edit['Value'].astype(str)  # هذه هي الجملة السحرية التي تمنع الخطأ
    
    edited_df = st.data_editor(
        df_to_edit,  # <<<<<<< نمرر النسخة المعدلة
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ref": st.column_config.TextColumn("Ref (e.g., R1)"),
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=["R", "V", "I", "VCVS", "VCCS", "CCVS", "CCCS"]
            ),
            "Node1": st.column_config.NumberColumn("Node +", step=1),
            "Node2": st.column_config.NumberColumn("Node -", step=1),
            "Value": st.column_config.TextColumn("Value (e.g., 10k, 5m)"),  # عمود نصي
            "Controlling": st.column_config.TextColumn("Control Ref (e.g., R3)"),
            "Gain": st.column_config.NumberColumn("Gain (α, β, gm, rm)", step=0.1)
        }
    )
    st.session_state.df = edited_df

with col2:
    st.subheader("⚙️ Analysis Mode")
    show_steps = st.toggle("📖 Step-by-Step", value=False)
    thev_node1 = st.number_input("Thevenin Node A", value=1, step=1)
    thev_node2 = st.number_input("Thevenin Node B", value=2, step=1)
    
    solve_btn = st.button("🚀 Solve DC Circuit", type="primary", use_container_width=True)
    thev_btn = st.button("🔌 Compute Thevenin @ Node", use_container_width=True)

if solve_btn:
    with st.spinner("Building MNA Matrix and Solving..."):
        node_V, currents, A_mat, B_vec, X, node_list = solve_mna(st.session_state.df)
        if X is not None:
            st.session_state.solution = (node_V, currents, A_mat, B_vec, X, node_list)
            st.success("✅ Solution Converged!")
        else:
            st.error("❌ Singular Matrix! Check your circuit.")
            st.session_state.solution = None

if st.session_state.solution is not None:
    node_V, currents, A_mat, B_vec, X, node_list = st.session_state.solution
    colA, colB = st.columns(2)
    with colA:
        st.subheader("📊 Node Voltages")
        v_df = pd.DataFrame(list(node_V.items()), columns=["Node", "Voltage (V)"])
        st.dataframe(v_df, hide_index=True, use_container_width=True)
        if currents:
            st.subheader("🔁 Voltage Source Currents")
            c_df = pd.DataFrame(list(currents.items()), columns=["Source", "Current (A)"])
            st.dataframe(c_df, hide_index=True, use_container_width=True)
    with colB:
        st.subheader("⚡ Power Dissipated (Resistors)")
        res_power = {}
        for _, row in st.session_state.df.iterrows():
            if row['Type'] == 'R':
                n1 = int(row['Node1']); n2 = int(row['Node2'])
                v1 = node_V.get(n1, 0); v2 = node_V.get(n2, 0)
                r_val = parse_component_value(row['Value'])
                if r_val != 0:
                    v_diff = v1 - v2
                    res_power[row['Ref']] = (v_diff ** 2) / r_val
        if res_power:
            p_df = pd.DataFrame(list(res_power.items()), columns=["Resistor", "Power (W)"])
            st.dataframe(p_df, hide_index=True, use_container_width=True)
    
    if show_steps:
        st.divider()
        st.subheader("📐 MNA Extended Matrix [A] * [X] = [B]")
        colM1, colM2 = st.columns(2)
        with colM1:
            st.write("**Matrix A**")
            st.dataframe(A_mat, use_container_width=True)
        with colM2:
            st.write("**Vector B**")
            st.dataframe(B_vec.reshape(-1, 1), use_container_width=True)

if thev_btn:
    st.subheader("🔍 Thevenin Equivalent Calculation")
    df_copy = st.session_state.df.copy()
    node_V, currents, _, _, _, _ = solve_mna(df_copy)
    if node_V is None:
        st.error("Circuit unstable. Cannot compute Thevenin.")
    else:
        Voc = node_V.get(thev_node1, 0) - node_V.get(thev_node2, 0)
        short_row = pd.DataFrame([{
            "Ref": "V_short", "Type": "V", "Node1": thev_node1, 
            "Node2": thev_node2, "Value": "0", "Controlling": "", "Gain": 0.0
        }])
        df_short = pd.concat([df_copy, short_row], ignore_index=True)
        node_V_short, currents_short, _, _, _, _ = solve_mna(df_short)
        if node_V_short is None:
            st.error("Cannot compute short circuit current.")
        else:
            Isc = abs(currents_short.get("I_V_short", 0))
            Rth = Voc / Isc if Isc != 0 else float('inf')
            st.metric("Open Circuit Voltage (Voc)", f"{Voc:.4f} V")
            st.metric("Short Circuit Current (Isc)", f"{Isc:.4f} A")
            st.metric("Thevenin Resistance (Rth)", f"{Rth:.4f} Ω" if Rth != float('inf') else "Infinite")

with st.sidebar:
    st.header("📘 Quick Reference")
    st.markdown("""
    **Dependent Sources Syntax**:
    - **VCVS**: Gain = V/V. Controlling = resistor reference.
    - **VCCS**: Gain = A/V. Controlling = resistor reference.
    - **CCVS**: Gain = V/A. Controlling = resistor reference.
    - **CCCS**: Gain = A/A. Controlling = resistor reference.
    """)
