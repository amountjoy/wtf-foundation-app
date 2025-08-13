
import streamlit as st
import pandas as pd
import numpy as np
import importlib.util
import matplotlib.pyplot as plt

import datetime


st.set_page_config(
    page_title="WTF Concept Design",  # Title shown in the browser tab
    page_icon="🌀",                   # Optional: emoji or image
    layout="centered",                   # Optional: 'centered' or 'wide'
    initial_sidebar_state="expanded" # Optional: 'auto', 'expanded', 'collapsed'
)


@st.cache_data
def run_optimisation_cached(mat_props, nominal_df, factored_df, Ext_wout_pf, Ext_w_pf, **kwargs):
    st.write("⏱️ Optimisation run at:", datetime.datetime.now())  # Optional debug
    wf = WTF_Concept.WTF_Concept_Design(submerged=st.session_state.get("submerged", True))
    wf.mat_props = mat_props
    return wf.optimise_foundation_geometry_parallel(
        No_Gap_LCs_wout_pf=nominal_df,
        No_Gap_LCs_w_pf=factored_df,
        Ext_wout_pf = Ext_wout_pf,
        Ext_w_pf = Ext_w_pf,
        **kwargs
    )



# Load the WTF_Concept module from the uploaded file
spec = importlib.util.spec_from_file_location("WTF_Concept", "WTF_Concept.py")
WTF_Concept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(WTF_Concept)

# Initialise session state
if 'mat_props' not in st.session_state:
    st.session_state.mat_props = {}
if 'load_data' not in st.session_state:
    st.session_state.load_data = {}
if 'optimisation_results' not in st.session_state:
    st.session_state.optimisation_results = {}
if 'selected_geometry' not in st.session_state:
    st.session_state.selected_geometry = {}

# Page 1: Material Properties Input
def page_material_properties():
    st.title("Material Properties Input")

    with st.form("material_form"):
        st.subheader("Geotechnical and Material Properties")
        
        st.session_state.submerged = st.checkbox(
            "Is the foundation submerged?",
            value=st.session_state.get("submerged", True)
        )


        g_ballast_dry = st.number_input("Dry Bulk Density of Ballast (kN/m³)", value=18.0)
        g_ballast_wet = st.number_input("Saturated Bulk Density of Ballast (kN/m³)", value=20.0)
        g_water = st.number_input("Specific Weight of Water (kN/m³)", value=-9.81)
        phi_prime = st.number_input("Effective Friction Angle (ø') [degrees]", value=30.0)
        allowable_bp = st.number_input("Allowable Bearing Pressure (kPa)", value=250.0)
        g_concrete = st.number_input("Density of Concrete (kN/m³)", value=24.5)
        g_steel = st.number_input("Density of Steel (kN/m³)", value=77.0)
        rebar = st.number_input("Rebar Mass (kg/m³)", value=150.0)

        submitted = st.form_submit_button("Save Material Properties")

        if submitted:
            st.session_state.mat_props = {
                "g_ballast_dry": g_ballast_dry,
                "g_ballast_wet": g_ballast_wet,
                "g_water": g_water,
                "phi_prime": phi_prime,
                "allowable_bp": allowable_bp,
                "g_concrete": g_concrete,
                "g_steel": g_steel,
                "rebar": rebar
            }
            st.success("Material properties saved.")


# Page 2: Load Case Upload
def page_load_upload():
    st.title("Load Case Upload")
    st.write("Upload loading CSV files, including no gapping governing case and extreme loads")

    load_file_ext_wout = st.file_uploader("Extreme Unfactored Loading CSV", type="csv", key="ext_wout")
    load_file_ext_w = st.file_uploader("Extreme Factored Loading CSV", type="csv", key="ext_w")
    load_file_nominal = st.file_uploader("No Gapping Case Unfactored Loading CSV", type="csv", key="nominal")
    load_file_factored = st.file_uploader("No Gapping Case Factored Loading CSV", type="csv", key="factored")
    climate_multiplier = st.number_input("Climate Change Wind Speed Multiplier", value=1.05)

    if load_file_ext_wout and load_file_ext_w and load_file_nominal and load_file_factored:
        wf = WTF_Concept.WTF_Concept_Design(submerged=st.session_state.get("submerged", True))


        # Save uploaded files temporarily
        with open("ext_wout.csv", "wb") as f:
            f.write(load_file_ext_wout.getbuffer())
        with open("ext_w.csv", "wb") as f:
            f.write(load_file_ext_w.getbuffer())
        with open("nominal.csv", "wb") as f:
            f.write(load_file_nominal.getbuffer())
        with open("factored.csv", "wb") as f:
            f.write(load_file_factored.getbuffer())

        # Read and map columns using read_LCs
        df_ext_wout = wf.read_LCs(filename="ext_wout.csv", map_file="column_map.csv")
        df_ext_w = wf.read_LCs(filename="ext_w.csv", map_file="column_map.csv")
        df_nominal = wf.read_LCs(filename="nominal.csv", map_file="column_map.csv")
        df_factored = wf.read_LCs(filename="factored.csv", map_file="column_map.csv")

        # Apply climate multiplier only to specific columns
        for df in [df_ext_wout, df_ext_w, df_nominal, df_factored]:
            for col in ["Resolved shear (kN)", "Torsional moment (kNm)", "Resolved moment (kNm)"]:
                if col in df.columns:
                    df[col] = df[col] * climate_multiplier**2


        st.session_state.load_data = {
            "ext_wout": df_ext_wout,
            "ext_w": df_ext_w,
            "nominal": df_nominal,
            "factored": df_factored
        }

        st.success("Load cases uploaded and climate multiplier applied.")

        # Display the DataFrames
        st.subheader("Extreme Unfactored Loads")
        st.dataframe(df_ext_wout)
        
        st.subheader("Extreme Factored Loads")
        st.dataframe(df_ext_w)
        
        st.subheader("No Gapping Case Unfactored Loads")
        st.dataframe(df_nominal)

        st.subheader("No Gapping Case Factored Loads")
        st.dataframe(df_factored)



# Page 3: Geometry Optimisation
def page_geometry_optimisation():
    st.title("Geometry Optimisation")
    st.write("Set parameter ranges and run geometry optimisation.")

    # Add cache clear button
    if st.button("🧹 Clear cached optimisation results"):
        st.cache_data.clear()
        st.success("✅ Cache cleared. You can now re-run the optimisation.")

    if st.session_state.get("load_data") and st.session_state.get("mat_props"):
        with st.form("optimisation_form"):
            st.subheader("Geometry Ranges")
            d1_min = st.number_input("Base Diameter Min (d1_min)", value=30.0)
            d1_max = st.number_input("Base Diameter Max (d1_max)", value=35.0)
            d_1_steps = st.number_input("Steps for d1", value=10, step=1)

            h1_min = st.number_input("Base Thickness Min (h1_min)", value=1)
            h1_max = st.number_input("Base Thickness Max (h1_max)", value=1.5)
            h_1_steps = st.number_input("Steps for h1", value=10, step=1)

            h2_min = st.number_input("Haunch Height Min (h2_min)", value=2.5)
            h2_max = st.number_input("Haunch Height Max (h2_max)", value=3)
            h_2_steps = st.number_input("Steps for h2", value=10, step=1)

            h3_min = st.number_input("Pedestal Height Min (h3_min)", value=0.5)
            h3_max = st.number_input("Pedestal Height Max (h3_max)", value=1.0)
            h_3_steps = st.number_input("Steps for h3", value=10, step=1)

            st.subheader("Fixed Geometry")
            d2 = st.number_input("Pedestal Diameter (d2)", value=7.0)
            b = st.number_input("Downstand Breadth (b)", value=7.0)
            h4 = st.number_input("Upstand Height (h4)", value=0.55)
            h5 = st.number_input("Downstand Depth (h5)", value=0.15)

            st.subheader("Design Constraints")
            h1_h2_thk_tol = st.number_input("Thickness Tolerance (±)", value=0.75)
            theta_min_deg = st.number_input("Min Haunch Angle (deg)", value=6.0)
            theta_max_deg = st.number_input("Max Haunch Angle (deg)", value=12.0)

            submitted = st.form_submit_button("Run Optimisation")

        if submitted:
            with st.spinner("Running optimisation..."):
                optimal, df_results = run_optimisation_cached(
                    mat_props=st.session_state.mat_props,
                    nominal_df=st.session_state.load_data["nominal"],
                    factored_df=st.session_state.load_data["factored"],
                    Ext_wout_pf=st.session_state.load_data["ext_wout"],
                    Ext_w_pf=st.session_state.load_data["ext_w"],
                    d1_min=d1_min, d1_max=d1_max, d_1_steps=d_1_steps,
                    h1_min=h1_min, h1_max=h1_max, h_1_steps=h_1_steps,
                    h2_min=h2_min, h2_max=h2_max, h_2_steps=h_2_steps,
                    h3_min=h3_min, h3_max=h3_max, h_3_steps=h_3_steps,
                    d2=d2, b=b, h4=h4, h5=h5,
                    h1_h2_thk_tol=h1_h2_thk_tol,
                    theta_min_deg=theta_min_deg, theta_max_deg=theta_max_deg
                )

            st.session_state.optimisation_results = df_results
            st.session_state.optimal_geometry = optimal
            st.success("✅ Optimisation completed.")





# Page 4: Results Visualisation
def page_results_visualisation():
    st.title("Optimisation Results")

    # Safely retrieve from session state
    df_results = st.session_state.get("optimisation_results")
    optimal = st.session_state.get("optimal_geometry")

    # Debugging output (optional, can be removed later)
    st.write("🔍 Debug: df_results is None?", df_results is None)
    st.write("🔍 Debug: df_results is empty?", df_results.empty if df_results is not None else "N/A")
    st.write("🔍 Debug: optimal_geometry is None?", optimal is None)

    if df_results is not None and not df_results.empty and optimal is not None:
        wf = WTF_Concept.WTF_Concept_Design(submerged=st.session_state.get("submerged", True))
        figs1 = wf.visualise_design_space(df_results, optimal)
        figs2 = wf.visualise_design_space_frontier(df_results, optimal)

        for fig in figs1 + figs2:
            st.pyplot(fig)
    else:
        st.warning("⚠️ No optimisation results available. Please run the optimisation first.")




# Page 5: Geometry Display
def page_geometry_display():
    st.title("Optimised Geometry Display")
    if st.session_state.optimal_geometry is not None:
        best_geometry = st.session_state.optimal_geometry
        st.session_state.selected_geometry = best_geometry
        wf = WTF_Concept.WTF_Concept_Design(submerged=st.session_state.get("submerged", True))
        wf.mat_props = st.session_state.mat_props
        fig1, fig2 = wf.plot_foundation(
            best_geometry["d1"], best_geometry["d2"], best_geometry["h1"],
            best_geometry["h2"], best_geometry["h3"], best_geometry["h4"],
            best_geometry["h5"], best_geometry["b"], best_geometry["hwt"]
        )
        st.pyplot(fig1)
        st.pyplot(fig2)




# Page 6: Interactive Adjustment
def page_interactive_adjustment():
    st.title("Adjust Geometry")
    if st.session_state.selected_geometry is not None: # and isinstance(st.session_state.selected_geometry, dict):
        geom = st.session_state.selected_geometry
        with st.form("adjust_geometry"):
            d1 = st.slider("Base Diameter (d1)", min_value=5.0, max_value=50.0, value=float(geom["d1"]))
            d2 = st.slider("Pedestal Diameter (d2)", min_value=5.0, max_value=15.0, value=float(geom["d2"]))
            h1 = st.slider("Base Thickness (h1)", min_value=0.5, max_value=5.0, value=float(geom["h1"]))
            h2 = st.slider("Haunch Height (h2)", min_value = 0.5, max_value=5.0, value=float(geom["h2"]))
            h3 = st.slider("Pedestal Height (h3)", min_value=0.5, max_value=2.0, value=float(geom["h3"]))
            h4 = st.slider("Height of Upstand Above FGL (h4)", min_value=0.0, max_value=2.0, value=float(geom["h4"]))
            h5 = st.slider("Downstand Height (h5)", min_value=0.0, max_value=1.0, value=float(geom["h5"]))
            b = st.slider("Downstand breadth (b)", min_value=0.0, max_value=15.0, value=float(geom["b"]))
            hwt = st.slider("Water Table Height (hwt)", min_value=0.0, max_value=30.0, value=float(geom["hwt"]))
            submitted = st.form_submit_button("Update Geometry")

        if submitted:
            geom["d1"] = d1
            geom["d2"] = d2
            geom["h1"] = h1
            geom["h2"] = h2
            geom["h3"] = h3
            geom["h4"] = h4
            geom["h5"] = h5
            geom["b"] = b
            geom["hwt"] = hwt
            st.session_state.selected_geometry = geom

        wf = WTF_Concept.WTF_Concept_Design(submerged=st.session_state.get("submerged", True))
        wf.mat_props = st.session_state.mat_props
        fig1, fig2 = wf.plot_foundation(
            geom["d1"], geom["d2"], geom["h1"], geom["h2"], geom["h3"],
            geom["h4"], geom["h5"], geom["b"], geom["hwt"]
        )
        st.pyplot(fig1)
        st.pyplot(fig2)


# Page 7: Verification Checks
def page_verification_checks():
    st.title("Verification Checks")
    if st.session_state.selected_geometry is not None and st.session_state.load_data is not None:
        geom = st.session_state.selected_geometry
        loads_ext_wout = st.session_state.load_data["ext_wout"]
        loads_ext_w = st.session_state.load_data["ext_w"]
        loads_unf = st.session_state.load_data["nominal"]
        loads_fact = st.session_state.load_data["factored"]
        props = st.session_state.mat_props
        wf = WTF_Concept.WTF_Concept_Design(submerged=st.session_state.get("submerged", True))
        wf.mat_props = props

        # Extract geometry parameters
        d1 = geom["d1"]
        d2 = geom["d2"]
        h1 = geom["h1"]
        h2 = geom["h2"]
        h3 = geom["h3"]
        h4 = geom["h4"]
        h5 = geom["h5"]
        b = geom["b"]
        hwt = geom["hwt"]

        # Compute volumes
        V_c = wf.vol_conc(h1, h2, h3, h5, d1, d2, b)
        V_h_f = wf.vol_haunch_fill(d1, h2, wf.vol_haunch(d1, d2, h2))
        V_p_f = wf.vol_pedestal_fill(d1, d2, h3, h4)
        V_d = wf.vol_downstand(b, h5)
        V_w = wf.vol_water(d1, hwt, V_d)

        # Compute loads
        Conc_DL = wf.foundation_perm_load(V_c, props["g_concrete"])
        Ballast_Sub = wf.foundation_perm_load(
            V_h_f + V_p_f,
            props["g_ballast_wet"] if wf.submerged else props["g_ballast_dry"]
        )

        Hydrostatic_Uplift = wf.foundation_perm_load(V_w, props["g_water"])

        # Calculate moments
        wf.M_top_bottom(loads_ext_wout, h1, h2, h3, h4)
        wf.M_top_bottom(loads_ext_w, h1, h2, h3, h4)
        wf.M_top_bottom(loads_unf, h1, h2, h3, h4)
        wf.M_top_bottom(loads_fact, h1, h2, h3, h4)

        # Perform checks
        no_gap_df = wf.no_gapping(d1, loads_unf["M_Res Bottom of Slab (kNm)"], loads_unf["Axial (kN)"], Conc_DL + Ballast_Sub + Hydrostatic_Uplift)
        ground_contact_df = wf.no_gapping(d1, loads_ext_wout["M_Res Bottom of Slab (kNm)"], loads_ext_wout["Axial (kN)"], Conc_DL + Ballast_Sub + Hydrostatic_Uplift, R_ratio=1/0.59)
        sbp_df = wf.soil_bearing_pressure(d1, loads_ext_wout["M_Res Bottom of Slab (kNm)"], loads_ext_wout["Axial (kN)"], Conc_DL + Ballast_Sub + Hydrostatic_Uplift)
        overturning_df = wf.overturning(d1, loads_ext_w["Axial (kN)"], Conc_DL, 0, Ballast_Sub, Hydrostatic_Uplift, loads_ext_w["M_Res Bottom of Slab (kNm)"], loads_ext_w["ULS partial factor"])
        sliding_df = wf.sliding(d1, props["phi_prime"], loads_ext_w["Axial (kN)"], Conc_DL, 0, Ballast_Sub, Hydrostatic_Uplift, loads_ext_w["Resolved shear (kN)"], loads_ext_w["Torsional moment (kNm)"], loads_ext_w["M_Res Bottom of Slab (kNm)"], loads_ext_w["ULS partial factor"])

        # Display results
        st.subheader("No Gapping Check")
        st.write("✅ Pass" if all(no_gap_df["Result"] == "Pass") else "❌ Fail")
        st.dataframe(no_gap_df)

        st.subheader("Ultimate Loads Ground Contact Check")
        st.write("✅ Pass" if all(ground_contact_df["Result"] == "Pass") else "❌ Fail")
        st.dataframe(ground_contact_df)

        st.subheader("Soil Bearing Pressure Check")
        st.write("✅ Pass" if all(sbp_df["Result_max"] == "Pass") and all(sbp_df["Result_mean"] == "Pass") else "❌ Fail")
        st.dataframe(sbp_df)

        st.subheader("Overturning Check")
        st.write("✅ Pass" if all(overturning_df["Result"] == "Pass") else "❌ Fail")
        st.dataframe(overturning_df)

        st.subheader("Sliding Check")
        st.write("✅ Pass" if all(sliding_df["Result"] == "Pass") else "❌ Fail")
        st.dataframe(sliding_df)




# Streamlit Navigation
page_names = [
    "Material Properties Input",
    "Load Case Upload",
    "Geometry Optimisation",
    "Results Visualisation",
    "Geometry Display",
    "Interactive Adjustment",
    "Verification Checks"
]

page_funcs = [
    page_material_properties,
    page_load_upload,
    page_geometry_optimisation,
    page_results_visualisation,
    page_geometry_display,
    page_interactive_adjustment,
    page_verification_checks
]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", page_names)
page_funcs[page_names.index(page)]()
