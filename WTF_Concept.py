# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:18:40 2025

@author: MOUNTJA
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import FloatSlider, VBox, interactive_output, Layout
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import clear_output, display, HTML
from IPython.display import Image as IPImage
from PIL import Image
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

class Utilities:
    """
    A utility class for displaying images in Jupyter notebooks using IPython display tools.
    """

    def __init__(self):
        """
        Initialises the Utilities class.
        Currently, no initialisation parameters are required.
        """
        pass

    def disp_image(self, file_path):
        """
        Displays an image from the specified file path in a Jupyter notebook.

        Parameters:
        file_path (str): The path to the image file to be displayed.
        """
        display(IPImage(filename=file_path))


class WTF_Concept_Design:
    """
    A class for wind turbine gravity foundation concept design.
    """
    
    #Set global parameters
    foundation_params = {}
    opt_params = {}
    value_widgets = {}
    mat_props = {}
    
    def __init__(self, submerged = True):   
        """Initialize the WTF_Concept_Design class."""
        self.submerged = submerged
        #return
    
    def vol_base_slab(self, d1, h1):

        """
        Calculate the volume of the circular base slab.

    Parameters:
        d1 (float): Diameter of the base slab.
        h1 (float): Height (thickness) of the base slab.

    Returns:
        float: Volume of the base slab.
        """

        return (d1**2 * np.pi/4)*h1

    def vol_haunch(self, d1, d2, h2):
        """
        Calculate the volume of the haunch (frustrum of a cone).

    Parameters:
        d1 (float): Diameter of the bottom of the haunch.
        d2 (float): Diameter of the top of the haunch.
        h2 (float): Height of the haunch.

    Returns:
        float: Volume of the haunch.
        """
        
        return 1/3*np.pi*((d1/2)**2+(d1/2)*(d2/2)+(d2/2)**2)*h2
    

    def vol_pedestal(self, d2, h3):
        """
        Calculate the volume of the cylindrical pedestal.

        Parameters:
        d2 (float): Diameter of the pedestal.
        h3 (float): Height of the pedestal.

        Returns:
            float: Volume of the pedestal.
            """
        return (d2**2 * np.pi / 4) * h3

    

    def vol_downstand(self, b, h5):
        """
        Calculate the volume of the square downstand.

        Parameters:
        b (float): Width of the downstand.
        h5 (float): Height of the downstand.

        Returns:
        float: Volume of the downstand.
        """
        return b**2 * h5

    

    def vol_conc(self, h1, h2, h3, h5, d1, d2, b):
        """
        Calculate the total volume of concrete used in the structure.

        Parameters:
        h1, h2, h3, h5 (float): Heights of base slab, haunch, pedestal, and downstand respectively.
        d1, d2 (float): Diameters of base and pedestal.
        b (float): Width of the downstand.

        Returns:
        float: Total concrete volume.
        """
        return (
            self.vol_base_slab(d1, h1) +
            self.vol_haunch(d1, d2, h2) +
            self.vol_pedestal(d2, h3) +
            self.vol_downstand(b, h5)
            )

    

    def vol_haunch_fill(self, d1, h2, vol_haunch):
        """
        Calculate the fill volume within the bounds of the haunch height and base slab circumference (difference between full cylinder and haunch volume).

        Parameters:
        d1 (float): Diameter of the base.
        h2 (float): Height of the haunch.
        vol_haunch (float): Volume of the haunch.

        Returns:
        float: Fill volume in the haunch.
        """
        return (d1**2 * np.pi / 4) * h2 - vol_haunch

    
    def vol_pedestal_fill(self, d1, d2, h3, h4):
        """
        Calculate the fill volume within the bounds of the pedestal height and base slab circumference (annular volume between two cylinders).

        Parameters:
        d1 (float): Outer diameter.
        d2 (float): Inner diameter.
        h3 (float): Total height of the pedestal.
        h4 (float): Height of the inner core.

        Returns:
        float: Fill volume in the pedestal.
        """

        return (d1**2*np.pi/4)*(h3-h4)-(d2**2*np.pi/4)*(h3-h4)
    
    def vol_water(self, d1, hwt, vol_downstand):
        
        """
        Calculate the volume of water including the downstand.

        Parameters:
        d1 (float): Diameter of the base slab.
        hwt (float): Height of water.
        vol_downstand (float): Volume of the downstand.

        Returns:
        float: Total water volume.
        """
        if self.submerged:
            vol = (d1**2*np.pi/4)*hwt+vol_downstand
        else:
            vol = 0
        return vol
    
    def plot_foundation(self, d1, d2, h1, h2, h3, h4, h5, b, hwt):
        """
        Visualises and analyses a foundation concept design based on geometric parameters.
    
        This method performs the following tasks:
        - Validates and adjusts input parameters to ensure geometric feasibility and design best practices.
        - Reads load case data from a CSV file and estimates minimum section thickness using regression.
        - Generates a 2D section view with annotated dimensions and warnings.
        - Creates a 3D visualisation of the foundation geometry.
        - Plots a bar chart of calculated volumes for each structural component.
        - Stores all input parameters and calculated volumes in the `foundation_params` dictionary.
    
        Parameters:
        d1 (float): Diameter of the base slab.
        d2 (float): Diameter of the pedestal.
        h1 (float): Thickness of the base slab.
        h2 (float): Height of the haunch.
        h3 (float): Height of the pedestal.
        h4 (float): Height of the upstand above finished ground level (FGL).
        h5 (float): Depth of the downstand below the base slab.
        b (float): Breadth of the downstand.
        hwt (float): Height of the groundwater table from the base.
    
        Returns:
        None. Displays plots and updates `self.foundation_params` with all relevant data.
        """
        warnings = []
   
        if d2 >= d1:
            d2 = d1
            warnings.append("Pedestal diameter (d2) adjusted to be less than base diameter (d1).")
    
        max_h2 = (np.tan(np.radians(12)) * (d1 - d2)) / 2
        max_d2 = (d1 * (np.tan(np.radians(12))) - 2 * h2)/(np.tan(np.radians(12)))
        #min_d1 = (2 * h2 + d2 * (np.tan(np.radians(12))))/(np.tan(np.radians(12)))
    
        if h2 >= max_h2:
            h2 = max_h2
            warnings.append("Haunch height (h2) adjusted to maintain ≤ 12° haunch angle.")
    
        if d2 >= max_d2:
            d2 = max_d2
            warnings.append("Pedestal diameter (d2) adjusted to maintain ≤ 12° haunch angle.")
    
        #if d1 >= min_d1:
        #    d1 = min_d1
        #    warnings.append("Base diameter (d1) adjusted to maintain ≤ 12° haunch angle.")
    
        if b >= d1:
            b = d1
            warnings.append("Downstand breadth (b) adjusted to be less than base diameter (d1).")
    
        if h4 >= h3:
            h4 = h3
            warnings.append("Upstand height above FGL (h4) adjusted to be less than pedestal height (h3).")
    
        if self.submerged:
            if hwt > h1 + h2 + h3 - h4:
                hwt = h1 + h2 + h3 - h4
                warnings.append("Height of ground water table (hwt) adjusted to be less than FGL (h1 + h2 + h3 - h4).")
        else:
            hwt = 0
        
        if d2<0:
            d2 = 0.1
            warnings.append("Pedestal diameter (d2) adjusted to be non-negative.")
            
        df_LC = self.read_LCs(filename="Loadcases_env_wout_factor.csv")
        df_LC['Fact_BM'] = df_LC['Resolved moment (kNm)'] * df_LC['ULS partial factor']
        
        def predict_min_section_thickness(max_factored_bm):
            # Regression coefficients from the best fit line
            slope = 0.0192
            intercept = 596.79
            # Calculate and return the predicted minimum section thickness
            return slope * max_factored_bm + intercept
        
        thk_min = predict_min_section_thickness(df_LC['Fact_BM'].max())/1000
        
        if h1+h2<thk_min:
            warnings.append(f"Base plus haunch thickness is lower than best practice minimum ({thk_min:.2f}m).")

    
        plt.close('all')
        fig = plt.figure(figsize=(10, 12))
    
        # 2D Section View
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title("2D Section View")
        ax1.set_xlabel("Width (m)")
        ax1.set_ylabel("Height (m)")
        ax1.grid(True)
    
        
        # Add dimension annotations
        def add_dimension(x1, y1, x2, y2, text, v_h, offset_v=4, offset_h=0.3):
            ax1.annotate('', xy=(x1, y1), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle='<->', color='black'))
            if v_h=='h':
                ax1.text((x1 + x2)/2, (y1 + y2)/2 + offset_h, text, ha='center', va='bottom', fontsize=9, backgroundcolor='white')
            else:
                ax1.text((x1 + x2)/2+offset_v, (y1 + y2)/2, text, ha='center', va='bottom', fontsize=9, backgroundcolor='white')
        
        # Horizontal dimensions
        add_dimension(-d1/2, -h1-h5-0.1, d1/2, -h1-h5-0.1, f'd1 = {d1:.3f} m', v_h='h')
        add_dimension(-d2/2, h1+h2+h3+0.5, d2/2, h1+h2+h3+0.5, f'd2 = {d2:.3f} m', v_h='h')
        add_dimension(-b/2, -h1-h5-1, b/2, -h1-h5-1, f'b = {b:.3f} m', v_h='h')
        
        # Vertical dimensions
        add_dimension(d1/2+0.5, 0, d1/2+0.5, h1, f'h1 = {h1:.3f} m', v_h='v')
        add_dimension(d2/2+0.5, h1, d2/2+0.5, h1+h2, f'h2 = {h2:.3f} m', v_h='v')
        add_dimension(d2/2+0.5, h1+h2, d2/2+0.5, h1+h2+h3, f'h3 = {h3:.3f} m', v_h='v')
        add_dimension(d2/2+d1/2, h1+h2+h3-h4, d2/2+d1/2, h1+h2+h3, f'h4 = {h4:.3f} m', v_h='v')
        add_dimension(d1/2+d1/4, 0, d1/2+d1/4, -h5, f'h5 = {h5:.3f} m', v_h='v')
        add_dimension(-d1/2-0.5, 0, -d1/2-0.5, hwt, f'hwt = {hwt:.3f} m', v_h='v')
    
    
        ax1.add_patch(plt.Rectangle((-d1/2, 0), d1, h1, color='lightgray', label='Base Slab'))
        ax1.add_patch(plt.Polygon([(-d1/2, h1), (-d2/2, h1+h2), (d2/2, h1+h2), (d1/2, h1)], color='gray', label='Haunch'))
        ax1.add_patch(plt.Rectangle((-d2/2, h1+h2), d2, h3, color='darkgray', label='Pedestal'))
        ax1.add_patch(plt.Rectangle((-b/2, -h5), b, h5, color='slategray', label='Downstand'))
    
        ax1.axhline(y=hwt, color='blue', linestyle='--', label='Water Table')
        ax1.axhline(y=h1+h2+h3-h4, color='brown', linestyle='--', label='FGL')
    
        for i, warning in enumerate(warnings):
            ax1.text(0, h1 + h2 + h3 + h4 + 1.5 + i * 0.5, f"⚠ {warning}", color='red', ha='center', fontsize=9)
    
        ax1.set_xlim(-d1, d1)
        ax1.set_ylim(- h5 - 4, h1 + h2 + h3 + h4 + 4)
        #ax1.set_aspect('equal')
        ax1.legend(loc='upper right')
    
        # 3D View
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        ax2.set_title("3D View")
    
        u = np.linspace(0, 2 * np.pi, 30)
        z = np.linspace(0, h1, 2)
        U, Z = np.meshgrid(u, z)
        X = (d1/2) * np.cos(U)
        Y = (d1/2) * np.sin(U)
        ax2.plot_surface(X, Y, Z, color='lightgray', alpha=0.8)
    
        z_haunch = np.linspace(h1, h1+h2, 2)
        R_haunch = np.linspace(d1 / 2, d2 / 2, 2)
        X_haunch = np.outer(np.cos(u), R_haunch)
        Y_haunch = np.outer(np.sin(u), R_haunch)
        Z_haunch = np.outer(np.ones_like(u), z_haunch)
        ax2.plot_surface(X_haunch, Y_haunch, Z_haunch, color='gray', alpha=0.8)
    
        z2 = np.linspace(h1+h2, h1 + h2 + h3, 2)
        U2, Z2 = np.meshgrid(u, z2)
        X2 = (d2/2) * np.cos(U2)
        Y2 = (d2/2) * np.sin(U2)
        ax2.plot_surface(X2, Y2, Z2, color='darkgray', alpha=0.9)
    
        
        # Downstand (rectangular prism)
        x_down = [-b/2, b/2]
        y_down = [-b/2, b/2]
        z_down = [-h5, 0]
        X_down, Y_down = np.meshgrid(x_down, y_down)
        for z_val in z_down:
            ax2.plot_surface(X_down, Y_down, z_val*np.ones_like(X_down), color='slategray', alpha=0.8)
        
        # Water Table (horizontal plane)
        #x_wt = np.linspace(-d1/2, d1/2, 2)
        #y_wt = np.linspace(-d1/2, d1/2, 2)
        #X_wt, Y_wt = np.meshgrid(x_wt, y_wt)
        #Z_wt = (hwt) * np.ones_like(X_wt)
        #ax2.plot_surface(X_wt, Y_wt, Z_wt, color='blue', alpha=0.3)
    
        # FGL (horizontal plane)
        #x_gl = np.linspace(-d1/2, d1/2, 2)
        #y_gl = np.linspace(-d1/2, d1/2, 2)
        #X_gl, Y_gl = np.meshgrid(x_gl, y_gl)
        #Z_gl = (h1+h2+h3-h4) * np.ones_like(X_gl)
        #ax2.plot_surface(X_gl, Y_gl, Z_gl, color='brown', alpha=0.3)
    
        ax2.set_xlim(-d1/2, d1/2)
        ax2.set_ylim(-d1/2, d1/2)
        ax2.set_zlim(-h1 - h5, h1 + h2 + h3 + h4)
        #ax2.set_box_aspect([1, 1, 1])
        
        
        # Set equal aspect ratio
        #max_range = max(d1, d1, h1 + h2 + h3 + h4 + h5)
        #ax2.set_box_aspect([d1, d1, h1 + h2 + h3 + h4 + h5]) #Equal scaling
        
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
    
        plt.tight_layout()
        plt.show()
    
        volumes = [
        self.vol_base_slab(d1, h1),
        self.vol_haunch(d1, d2, h2),
        self.vol_pedestal(d2, h3),
        self.vol_downstand(b, h5),
        self.vol_haunch_fill(d1, h2, self.vol_haunch(d1, d2, h2)),
        self.vol_pedestal_fill(d1, d2, h3, h4)
        ]
        labels = ['Base Slab', 'Haunch', 'Pedestal', 'Downstand', 'Haunch Fill', 'Pedestal Fill']
        
        
        # Define colors: skyblue for first 4, orange for last 2
        colours = ['grey'] * 4 + ['brown'] * 2

        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, volumes, color=colours)
        plt.title('Volumes')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylabel('Volume (m\u00b3)')
    
        # Annotate each bar with its volume value
        for bar, volume in zip(bars, volumes):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{volume:.2f}m\u00b3', ha='center', va='bottom')
    
        
        # Add legend with total volumes
        conc_total = sum(volumes[:4])
        rebar_total = conc_total * self.mat_props["rebar"]
        fill_total = sum(volumes[4:])
        main_patch = plt.Rectangle((0, 0), 1, 1, color='grey', label=f'Concrete Volume Total: {conc_total:.2f} m³')
        rebar_patch = plt.Rectangle((0, 0), 1, 1, color='white', label=f'Rebar Mass Total: {rebar_total:.2f} kg')
        fill_patch = plt.Rectangle((0, 0), 1, 1, color='brown', label=f'Fill Volume Total: {fill_total:.2f} m³')
        plt.legend(handles=[main_patch, rebar_patch, fill_patch])

        
        plt.tight_layout()
        plt.show()
           
        
        #global foundation_params
        self.foundation_params = {
            'd1': d1, 'd2': d2, 'h1': h1, 'h2': h2,
            'h3': h3, 'h4': h4, 'h5': h5, 'b': b, 'hwt': hwt,
            'V_bs': self.vol_base_slab(d1, h1), 'V_h': self.vol_haunch(d1, d2, h2),
            'V_p': self.vol_pedestal(d2, h3), 'V_d': self.vol_downstand(b, h5),
            'V_c': self.vol_conc(h1, h2, h3, h5, d1, d2, b),
            'V_h_f': self.vol_haunch_fill(d1, h2, self.vol_haunch(d1, d2, h2)),
            'V_p_f': self.vol_pedestal_fill(d1, d2, h3, h4),
            'V_w': self.vol_water(d1, hwt, self.vol_downstand(b, h5)),
            'Rebar_mass': self.vol_conc(h1, h2, h3, h5, d1, d2, b) * self.mat_props["rebar"]
        }
        
        return fig, plt.gcf()

    def interactive_foundations(self, d1_init, d2_init, h1_init, h2_init, h3_init, h4_init, h5_init, b_init, hwt_init):
        style = {'description_width': 'initial'}
        
        sliders = {
            'd1': FloatSlider(value=d1_init, min=0, max=100, step=0.001, description='Base Diameter (d1)', layout=widgets.Layout(width='100%'), style=style),
            'd2': FloatSlider(value=d2_init, min=0, max=100, step=0.001, description='Pedestal Diameter (d2)', layout=widgets.Layout(width='100%'), style=style),
            'h1': FloatSlider(value=h1_init, min=0, max=10, step=0.001, description='Base Depth (h1)', layout=widgets.Layout(width='100%'), style=style),
            'h2': FloatSlider(value=h2_init, min=0, max=10, step=0.001, description='Haunch Height (h2)', layout=widgets.Layout(width='100%'), style=style),
            'h3': FloatSlider(value=h3_init, min=0, max=10, step=0.001, description='Pedestal Height (h3)', layout=widgets.Layout(width='100%'), style=style),
            'h4': FloatSlider(value=h4_init, min=0, max=5, step=0.001, description='Height of Upstand above FGL (h4)', layout=widgets.Layout(width='100%'), style=style),
            'h5': FloatSlider(value=h5_init, min=0, max=5, step=0.001, description='Downstand Height (h5)', layout=widgets.Layout(width='100%'), style=style),
            'b': FloatSlider(value=b_init, min=0, max=10, step=0.001, description='Downstand Breadth (b)', layout=widgets.Layout(width='100%'), style=style),
            'hwt': FloatSlider(value=hwt_init, min=0, max=10, step=0.001, description='Water Table Height (hwt)', layout=widgets.Layout(width='100%'), style=style)
        }
        
        ui = VBox([sliders[k] for k in sliders])
        out = interactive_output(self.plot_foundation, sliders)
        
        display(ui, out)
    
    def read_LCs(self, filename="Loadcases.csv", index_col='Loadcase', map_file="column_map.csv"):
        df = pd.read_csv(filename, index_col=index_col)
        
        # Load column map from CSV
        map_df = pd.read_csv(map_file)
        column_map = dict(zip(map_df['Original name'], map_df['Standard name']))
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        return df
    
    def read_data(self, filename, index_col=0):
        df = pd.read_csv(filename, index_col=index_col)
        return df
    
    def ecc_top_plinth(self, LC_height, h4):
        return LC_height-h4
    
    def ecc_bot_slab(self, h1, h2, h3):
        return h1+h2+h3
    
    def M_Res(self, F_H, ecc):
        return F_H * ecc
    
    def M_top_bottom(self, df, h1, h2, h3, h4, LC_height = 'Assumed height above ground level (m)', F_R = 'Resolved shear (kN)', M_R = 'Resolved moment (kNm)'):
    
        df['Eccentricity Top of Plinth (m)'] = self.ecc_top_plinth(LC_height=df[LC_height], h4=h4)
    
        df['Eccentricity Bottom of Slab (m)'] = self.ecc_bot_slab(h1 = h1, h2 = h2, h3=h3)
    
        df['M_Res Top of Plinth (kNm)'] = df[M_R] + self.M_Res(F_H = df[F_R], ecc = self.ecc_top_plinth(LC_height=df[LC_height], h4=h4))
    
        df['M_Res Bottom of Slab (kNm)'] = df['M_Res Top of Plinth (kNm)'] + self.M_Res(F_H = df[F_R], ecc = self.ecc_bot_slab(h1 = h1, h2 = h2, h3 = h3))
        

    def mat_prop_input(self):
        # Define the properties with initial values, units, and descriptions
        properties = [
            {"name": "g ballast (dry)", "value": 18.00, "unit": "kN/m³", "description": "dry bulk density of ballast"},
            {"name": "g ballast (wet)", "value": 20.00, "unit": "kN/m³", "description": "saturated bulk density of ballast"},
            {"name": "g water", "value": -9.81, "unit": "kN/m³", "description": "specific weight of water (assumed at 4°C)"},
            {"name": "ø'", "value": 30.00, "unit": "degs", "description": "6N upfill / capping layer to gravity base"},
            {"name": "Allowable BP", "value": 250.00, "unit": "kPa", "description": "allow. peak edge bearing pressure; ref. Geotechnical Report"},
            {"name": "g conc.", "value": 24.50, "unit": "kN/m³", "description": "assumed density of RC WTG foundation"},
            {"name": "g steel", "value": 77.00, "unit": "kN/m³", "description": "assumed density for embedded structural steel"},
            {"name": "rebar kg (per m3 conc)", "value": 150, "unit": "kg/m3", "description": "assumed mass of rebar per cubic metre of concrete"}
        ]
        
              
        rows = []

        for prop in properties:
            name = prop["name"]
            value_widget = widgets.FloatText(value=prop["value"], layout=widgets.Layout(width='150px'))
            unit_label = widgets.Label(value=prop["unit"], layout=widgets.Layout(width='80px'))
            desc_label = widgets.Label(value=prop["description"], layout=widgets.Layout(flex='1'))

            self.value_widgets[name] = value_widget

            # Observe changes to update mat_props
            value_widget.observe(self.update_mat_props, names='value')

            row = widgets.HBox([
                widgets.Label(value=name, layout=widgets.Layout(width='150px')),
                value_widget, unit_label, desc_label
            ])
            rows.append(row)

        display(HTML("<h3>Geotechnical and Material Properties Input Form</h3>"))
        display(widgets.VBox(rows))

        # Initialise mat_props with default values
        self.update_mat_props(None)

    def update_mat_props(self, change):
        self.mat_props = {
            'g_ballast_dry': self.value_widgets['g ballast (dry)'].value,
            'g_ballast_wet': self.value_widgets['g ballast (wet)'].value,
            'g_water': self.value_widgets['g water'].value,
            "phi_prime": self.value_widgets["ø'"].value,
            "allowable_bp": self.value_widgets["Allowable BP"].value,
            "g_concrete": self.value_widgets["g conc."].value,
            "g_steel": self.value_widgets["g steel"].value,
            "rebar": self.value_widgets["rebar kg (per m3 conc)"].value
        }
        # Optional: print or log updates
        #print("Material properties updated:", self.mat_props)
        
    def foundation_perm_load(self, volume, density):
        return volume * density

    def no_gapping(self, d1, M_Res, F_z_turb, F_z_found, R_ratio=4):
        # Convert inputs to numpy arrays if they aren't already
        M_Res = pd.Series(M_Res)
        F_z_turb = pd.Series(F_z_turb, index=M_Res.index)
        F_z_found = pd.Series(F_z_found, index=M_Res.index)
    
        # Calculate scalar e_limit
        e_limit = d1 / (2 * R_ratio)
    
        # Calculate eccentricity
        e = M_Res / (F_z_turb + F_z_found)
    
        # Broadcast e_limit to match shape of e
        e_limit_array = np.full_like(e, e_limit)
    
        # Calculate utilisation and result
        utilisation = e / e_limit_array
        results = np.where(utilisation <= 1, 'Pass', 'Fail')
    
        # Create DataFrame using M_Res index
        df = pd.DataFrame({
            'e_limit (m)': e_limit_array,
            'M_Res (kNm)': M_Res,
            'F_z (kN)': F_z_turb + F_z_found,
            'e (m)': e,
            'Utilisation': utilisation,
            'Result': results
        }, index=M_Res.index)
    
        return df
    
    def soil_bearing_pressure(self, d1, M_Res, F_z_turb, F_z_found, Theta_allow=250):
        # Convert inputs to pandas Series
        M_Res = pd.Series(M_Res)
        F_z_turb = pd.Series(F_z_turb, index=M_Res.index)
        F_z_found = pd.Series(F_z_found, index=M_Res.index)
    
        # Calculate eccentricity and e/r
        e = M_Res / (F_z_turb + F_z_found)
        e_r = e / (d1 / 2)
    
        # Load e/r to K lookup table
        e_r_K = pd.read_csv('e_r_K.csv', index_col=0)
        e_r_K = e_r_K.sort_index()
    
        # Interpolation function
        def interpolate_K(e_r_value):
            if e_r_value <= e_r_K.index.min():
                return e_r_K.iloc[0]['K']
            elif e_r_value >= e_r_K.index.max():
                return e_r_K.iloc[-1]['K']
            else:
                lower_index = e_r_K.index[e_r_K.index <= e_r_value].max()
                upper_index = e_r_K.index[e_r_K.index >= e_r_value].min()
                lower_value = e_r_K.loc[lower_index, 'K']
                upper_value = e_r_K.loc[upper_index, 'K']
                return lower_value + (upper_value - lower_value) * (e_r_value - lower_index) / (upper_index - lower_index)
    
        # Apply interpolation to each e/r value
        K = e_r.apply(interpolate_K)
    
        alpha = 2 * np.degrees(np.arccos(e_r))
        A_eff = ((d1/2)**2)*(np.radians(alpha)-np.sin(np.radians(alpha)))
        Theta_mean = (F_z_turb + F_z_found)/A_eff
        
        Theta_max = K * (F_z_turb + F_z_found) / (np.pi * (d1 / 2) ** 2)
        
        utilisation_max = Theta_max / Theta_allow
        utilisation_mean = Theta_mean / Theta_allow
        results_max = np.where(utilisation_max <= 1, 'Pass', 'Fail')
        results_mean = np.where(utilisation_mean <= 1, 'Pass', 'Fail')
    
        # Create results DataFrame
        df = pd.DataFrame({
            'd1': d1,
            'M_Res (kNm)': M_Res,
            'F_z (kN)': F_z_turb + F_z_found,
            'e (m)': e,
            'e/r': e_r,
            'K': K,
            'Theta_max (kPa)': Theta_max,
            'Utilisation_max': utilisation_max,
            'Result_max': results_max,
            'alpha (degs)': alpha,
            'A_eff (m2)': A_eff,
            'Theta_mean (kPa)': Theta_mean,
            'Utilisation_mean': utilisation_mean,
            'Result_mean': results_mean
        }, index=M_Res.index)
    
        return df
    
    def overturning(self, d1, F_z_turb, F_z_found_conc, F_z_found_steel, F_z_found_ballast, F_z_found_buoyancy, M_Res, Q_fact, gamma_G_stb = 0.9, gamma_G_dst = 1.10):

        e = d1/2
        
        # Convert inputs to pandas Series
        M_Res = pd.Series(M_Res)
        F_z_turb = pd.Series(F_z_turb, index=M_Res.index)
        F_z_found_conc = pd.Series(F_z_found_conc, index=M_Res.index)
        F_z_found_steel = pd.Series(F_z_found_steel, index=M_Res.index)
        F_z_found_ballast = pd.Series(F_z_found_ballast, index=M_Res.index)
        F_z_found_buoyancy = pd.Series(F_z_found_buoyancy, index=M_Res.index)
        Q_fact = pd.Series(Q_fact, index=M_Res.index)
    
        #Apply partial factors
        F_z = gamma_G_stb * (F_z_found_conc + F_z_found_steel + F_z_found_ballast + F_z_turb) + gamma_G_dst * F_z_found_buoyancy
        M_Res = M_Res * Q_fact
    
        M_d_stb = F_z * e
        M_d_dst = M_Res
    
        utilisation = M_d_dst / M_d_stb
        results = np.where(utilisation <= 1, 'Pass', 'Fail')
    
        # Create DataFrame using M_Res index
        df = pd.DataFrame({
            'e (m)': e,
            'Fz (kN)': F_z,
            'Md,stb (kNm)': M_d_stb,
            'Md,dst (kNm)': M_d_dst,
            'Utilisation': utilisation,
            'Result': results
        }, index=M_Res.index)
        
        return df

    def sliding(self, d1, phi, F_z_turb, F_z_found_conc, F_z_found_steel, F_z_found_ballast, F_z_found_buoyancy, F_Res, M_z, M_Res, Q_fact, gamma_G_stb = 0.9, gamma_G_dst = 1.10, lambda_phi = 1.25):
        R = d1/2
    
        tan_phi_d = np.tan(np.radians(phi))/lambda_phi
        
        # Convert inputs to pandas Series
        M_Res = pd.Series(M_Res)
        F_z_turb = pd.Series(F_z_turb, index=M_Res.index)
        F_z_found_conc = pd.Series(F_z_found_conc, index=M_Res.index)
        F_z_found_steel = pd.Series(F_z_found_steel, index=M_Res.index)
        F_z_found_ballast = pd.Series(F_z_found_ballast, index=M_Res.index)
        F_z_found_buoyancy = pd.Series(F_z_found_buoyancy, index=M_Res.index)
        F_Res = pd.Series(F_Res, index=M_Res.index)
        M_z = pd.Series(M_z, index=M_Res.index)
        Q_fact = pd.Series(Q_fact, index=M_Res.index)
        
        #Apply partial factors
        F_z = gamma_G_stb * (F_z_found_conc + F_z_found_steel + F_z_found_ballast + F_z_turb) + gamma_G_dst * F_z_found_buoyancy
        F_Res = F_Res * Q_fact
        M_Res = M_Res * Q_fact
        M_z = M_z * Q_fact
        
        #M_d_stb = F_z * e
        M_d_dst = M_Res
    
        e = M_d_dst / F_z
    
        H_d_stb = F_z * tan_phi_d
    
        A_eff = 2*(((d1/2)**2)*(np.arccos(e/R))-(e*(R**2-e**2)**0.5))
    
        b_e = 2*(R-e)
    
        l_e = 2*R*(1-(1-(b_e/(2*R)))**2)**0.5
    
        l_eff = (A_eff*(l_e/b_e))**0.5
    
        H_dash = (2*M_z)/l_eff+((F_Res)**2+(2*M_z/l_eff)**2)**0.5
    
        H_d_dst = H_dash + F_Res
        
        utilisation = H_d_dst / H_d_stb
        results = np.where(utilisation <= 1, 'Pass', 'Fail')
        
        # Create DataFrame using M_Res index
        df = pd.DataFrame({
            'Fz (kN)': F_z,
            'M_Res (kNm)': M_Res,
            'M_z (kNm)': M_z,
            'e (m)': e,
            'phi (degs)': phi,
            'tan(phi_d)': tan_phi_d,
            'A_eff (m2)': A_eff,
            'b_e (m)': b_e,
            'l_e (m)': l_e,
            'l_eff (m)': l_eff,
            "H'(kN)": H_dash,
            'Hd,stb (kN)': H_d_stb,
            'Hd,dst (kN)': H_d_dst,
            'Utilisation': utilisation,
            'Result': results
        }, index=M_Res.index)
        
        return df
    



    #DO NOT USE THIS - USE PARALLEL VERSION INSTEAD!
    def optimise_foundation_geometry(self, LCs_wout_pf, LCs_w_pf, d1_min=20, d1_max=40, d_1_steps=20,
                                      h1_min=0.1, h1_max=2.5, h_1_steps=20,
                                      h2_min=0.1, h2_max=2.5, h_2_steps=20,
                                      h3_min=0.1, h3_max=1.0, h_3_steps=20,
                                      d2=7, b=7, h4=0.55, h5=0.15,
                                      h1_h2_thk_tol = 0.75,
                                      theta_min_deg=6, theta_max_deg=12):
        # Generate parameter ranges
        d1_range = np.linspace(d1_min, d1_max, d_1_steps)
        h1_range = np.linspace(h1_min, h1_max, h_1_steps)
        h2_range = np.linspace(h2_min, h2_max, h_2_steps)
        h3_range = np.linspace(h3_min, h3_max, h_3_steps)
    
        # Convert angles to radians
        theta_max_rad = np.radians(theta_max_deg)
        theta_min_rad = np.radians(theta_min_deg)
    
        # Material properties
        phi = self.mat_props['phi_prime']
        g_concrete = self.mat_props['g_concrete']
        g_ballast = self.mat_props['g_ballast_wet']
        g_water = self.mat_props['g_water']
    
        # Predict minimum section thickness
        def predict_min_section_thickness(max_factored_bm):
            slope = 0.0192
            intercept = 596.79
            return slope * max_factored_bm + intercept
    
        df_LC = LCs_wout_pf.copy()
        df_LC['Fact_BM'] = df_LC['Resolved moment (kNm)'] * df_LC['ULS partial factor']
        thk_min = predict_min_section_thickness(df_LC['Fact_BM'].max()) / 1000
    
        results = []
    
        combinations = list(itertools.product(d1_range, h1_range, h2_range, h3_range))
    
        for d1, h1, h2, h3 in tqdm(combinations, desc="Optimising geometry"):
            if h1 > h2 or h3 > h2:
                continue
            if h1 + h2 < thk_min - 0.75 or h1 + h2 > thk_min + 0.75:
                continue
            if h3 < h4:
                continue
        
            hwt = h1 + h2 + h3 - h4
            haunch_half_span = (d1 - d2) / 2
        
            if np.tan(h2 / haunch_half_span) > theta_max_rad or np.tan(h2 / haunch_half_span) < theta_min_rad:
                continue
    
    
            V_c = self.vol_conc(h1, h2, h3, h5, d1, d2, b)
            V_bs = self.vol_base_slab(d1, h1)
            V_h = self.vol_haunch(d1, d2, h2)
            V_p = self.vol_pedestal(d2, h3)
            V_d = self.vol_downstand(b, h5)
            V_h_f = self.vol_haunch_fill(d1, h2, V_h)
            V_p_f = self.vol_pedestal_fill(d1, d2, h3, h4)
            V_w = self.vol_water(d1, hwt, V_d)
    
            self.opt_params.update({'d1': d1, 'h1': h1, 'h2': h2, 'h3': h3, 'd2': d2, 'h4': h4, 'h5': h5, 'hwt': hwt,
                                  'V_c': V_c, 'V_bs': V_bs, 'V_h': V_h, 'V_p': V_p, 'V_d': V_d, 'V_h_f': V_h_f,
                                  'V_p_f': V_p_f, 'V_w': V_w})
    
            self.M_top_bottom(df=LCs_wout_pf, h1=h1, h2=h2, h3=h3, h4=h4)
            self.M_top_bottom(df=LCs_w_pf, h1=h1, h2=h2, h3=h3, h4=h4)
    
            Conc_DL = self.foundation_perm_load(volume=V_c, density=g_concrete)
            Ballast_Sub = self.foundation_perm_load(volume=V_h_f + V_p_f, density=g_ballast)
            Hydrostatic_Uplift = self.foundation_perm_load(volume=V_w, density=g_water)
    
            try:
                no_gap = self.no_gapping(d1, LCs_wout_pf['M_Res Bottom of Slab (kNm)'],
                                       LCs_wout_pf['Axial (kN)'], Conc_DL + Ballast_Sub + Hydrostatic_Uplift, R_ratio=4)
                if all(no_gap['Result'] == 'Pass'):
                    ground_contact = self.no_gapping(d1, LCs_wout_pf['M_Res Bottom of Slab (kNm)'],
                                                   LCs_wout_pf['Axial (kN)'], Conc_DL + Ballast_Sub + Hydrostatic_Uplift, R_ratio=1/0.59)
                    if all(ground_contact['Result'] == 'Pass'):
                        sbp = self.soil_bearing_pressure(d1, LCs_wout_pf['M_Res Bottom of Slab (kNm)'],
                                                       LCs_wout_pf['Axial (kN)'], Conc_DL + Ballast_Sub + Hydrostatic_Uplift, Theta_allow=250)
                        if all(sbp['Result_max'] == 'Pass') and all(sbp['Result_mean'] == 'Pass'):
                            overturning = self.overturning(d1, LCs_w_pf['Axial (kN)'], Conc_DL, 0, Ballast_Sub, Hydrostatic_Uplift,
                                                         LCs_w_pf['M_Res Bottom of Slab (kNm)'], LCs_w_pf['ULS partial factor'], 0.9, 1.10)
                            if all(overturning['Result'] == 'Pass'):
                                sliding = self.sliding(d1, phi, LCs_w_pf['Axial (kN)'], Conc_DL, 0, Ballast_Sub, Hydrostatic_Uplift,
                                                     LCs_w_pf['Resolved shear (kN)'], LCs_w_pf['Torsional moment (kNm)'],
                                                     LCs_w_pf['M_Res Bottom of Slab (kNm)'], LCs_w_pf['ULS partial factor'], 0.9, 1.10, 1.25)
                                if all(sliding['Result'] == 'Pass'):
                                    results.append({
                                        'd1': d1, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'h5': h5, 'd2': d2, 'b': b, 'hwt': hwt,
                                        'V_c': V_c, 'V_bs': V_bs, 'V_h': V_h, 'V_p': V_p, 'V_d': V_d
                                    })
            except:
                continue
    
        if results:
            df_results = pd.DataFrame(results)
            optimal = df_results.loc[df_results['V_c'].idxmin()]
            return optimal, df_results
        else:
            return None, pd.DataFrame()
        
    def optimise_foundation_geometry_parallel(self, LCs_wout_pf, LCs_w_pf,
                                              d1_min=20, d1_max=40, d_1_steps=20,
                                              h1_min=0.1, h1_max=2.5, h_1_steps=20,
                                              h2_min=0.1, h2_max=2.5, h_2_steps=20,
                                              h3_min=0.1, h3_max=1.0, h_3_steps=20,
                                              d2=7, b=7, h4=0.55, h5=0.15,
                                              h1_h2_thk_tol=0.75,
                                              theta_min_deg=6, theta_max_deg=12
                                              ):

        # Generate parameter ranges
        d1_range = np.linspace(d1_min, d1_max, d_1_steps)
        h1_range = np.linspace(h1_min, h1_max, h_1_steps)
        h2_range = np.linspace(h2_min, h2_max, h_2_steps)
        h3_range = np.linspace(h3_min, h3_max, h_3_steps)
    
        # Convert angles to radians
        theta_max_rad = np.radians(theta_max_deg)
        theta_min_rad = np.radians(theta_min_deg)
    
        # Material properties
        phi = self.mat_props['phi_prime']
        g_concrete = self.mat_props['g_concrete']
        if self.submerged:
            g_ballast = self.mat_props['g_ballast_wet']
        else:
            g_ballast = self.mat_props['g_ballast_dry']
        g_water = self.mat_props['g_water']
    
        # Predict minimum section thickness
        def predict_min_section_thickness(max_factored_bm):
            slope = 0.0192
            intercept = 596.79
            return slope * max_factored_bm + intercept
    
        df_LC = LCs_wout_pf.copy()
        df_LC['Fact_BM'] = df_LC['Resolved moment (kNm)'] * df_LC['ULS partial factor']
        thk_min = predict_min_section_thickness(df_LC['Fact_BM'].max()) / 1000
    
        # Pre-filter combinations
        combinations = [
            (d1, h1, h2, h3) for d1, h1, h2, h3 in itertools.product(d1_range, h1_range, h2_range, h3_range)
            if h1 <= h2 and h3 <= h2 and h3 >= h4 and thk_min - h1_h2_thk_tol <= h1 + h2 <= thk_min + h1_h2_thk_tol
        ]
    
        def evaluate_combination(d1, h1, h2, h3):
            haunch_half_span = (d1 - d2) / 2
            if haunch_half_span == 0 or not (theta_min_rad <= np.tan(h2 / haunch_half_span) <= theta_max_rad):
                return None
    
            if self.submerged:
                hwt = h1 + h2 + h3 - h4
            else:
                hwt = 0
    
            V_c = self.vol_conc(h1, h2, h3, h5, d1, d2, b)
            V_bs = self.vol_base_slab(d1, h1)
            V_h = self.vol_haunch(d1, d2, h2)
            V_p = self.vol_pedestal(d2, h3)
            V_d = self.vol_downstand(b, h5)
            V_h_f = self.vol_haunch_fill(d1, h2, V_h)
            V_p_f = self.vol_pedestal_fill(d1, d2, h3, h4)
            V_w = self.vol_water(d1, hwt, V_d)
    
            self.opt_params.update({'d1': d1, 'h1': h1, 'h2': h2, 'h3': h3, 'd2': d2, 'h4': h4, 'h5': h5, 'hwt': hwt,
                                  'V_c': V_c, 'V_bs': V_bs, 'V_h': V_h, 'V_p': V_p, 'V_d': V_d, 'V_h_f': V_h_f,
                                  'V_p_f': V_p_f, 'V_w': V_w})
    
            self.M_top_bottom(df=LCs_wout_pf, h1=h1, h2=h2, h3=h3, h4=h4)
            self.M_top_bottom(df=LCs_w_pf, h1=h1, h2=h2, h3=h3, h4=h4)
    
            Conc_DL = self.foundation_perm_load(volume=V_c, density=g_concrete)
            Ballast_Sub = self.foundation_perm_load(volume=V_h_f + V_p_f, density=g_ballast)
            Hydrostatic_Uplift = self.foundation_perm_load(volume=V_w, density=g_water)
    
            try:
                no_gap = self.no_gapping(d1, LCs_wout_pf['M_Res Bottom of Slab (kNm)'],
                                       LCs_wout_pf['Axial (kN)'], Conc_DL + Ballast_Sub + Hydrostatic_Uplift, R_ratio=4)
                if not all(no_gap['Result'] == 'Pass'):
                    return None
    
                ground_contact = self.no_gapping(d1, LCs_wout_pf['M_Res Bottom of Slab (kNm)'],
                                               LCs_wout_pf['Axial (kN)'], Conc_DL + Ballast_Sub + Hydrostatic_Uplift, R_ratio=1/0.59)
                if not all(ground_contact['Result'] == 'Pass'):
                    return None
    
                sbp = self.soil_bearing_pressure(d1, LCs_wout_pf['M_Res Bottom of Slab (kNm)'],
                                               LCs_wout_pf['Axial (kN)'], Conc_DL + Ballast_Sub + Hydrostatic_Uplift, Theta_allow=250)
                if not (all(sbp['Result_max'] == 'Pass') and all(sbp['Result_mean'] == 'Pass')):
                    return None
    
                overturning = self.overturning(d1, LCs_w_pf['Axial (kN)'], Conc_DL, 0, Ballast_Sub, Hydrostatic_Uplift,
                                             LCs_w_pf['M_Res Bottom of Slab (kNm)'], LCs_w_pf['ULS partial factor'], 0.9, 1.10)
                if not all(overturning['Result'] == 'Pass'):
                    return None
    
                sliding = self.sliding(d1, phi, LCs_w_pf['Axial (kN)'], Conc_DL, 0, Ballast_Sub, Hydrostatic_Uplift,
                                     LCs_w_pf['Resolved shear (kN)'], LCs_w_pf['Torsional moment (kNm)'],
                                     LCs_w_pf['M_Res Bottom of Slab (kNm)'], LCs_w_pf['ULS partial factor'], 0.9, 1.10, 1.25)
                if not all(sliding['Result'] == 'Pass'):
                    return None
    
                return {
                    'd1': d1, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'h5': h5, 'd2': d2, 'b': b, 'hwt': hwt,
                    'V_c': V_c, 'V_bs': V_bs, 'V_h': V_h, 'V_p': V_p, 'V_d': V_d
                }
            except:
                return None
    
        results = Parallel(n_jobs=-1)(delayed(evaluate_combination)(d1, h1, h2, h3) for d1, h1, h2, h3 in tqdm(combinations, desc="Optimising geometry"))
    
        results = [res for res in results if res is not None]
    
        if results:
            df_results = pd.DataFrame(results)
            optimal = df_results.loc[df_results['V_c'].idxmin()]
            return optimal, df_results
        else:
            return None, pd.DataFrame()
        


    def visualise_design_space(self, df_results, optimal):
        figs = []
    
        # 3D Scatter Plot: d1, h1, h2 colored by V_c
        fig1 = plt.figure(figsize=(10, 7))
        ax1 = fig1.add_subplot(111, projection='3d')
        sc1 = ax1.scatter(df_results['d1'], df_results['h1'], df_results['h2'],
                          c=df_results['V_c'], cmap='viridis', s=50)
        
        
        ax1.set_xlabel('d1 (m)')
        ax1.set_ylabel('h1 (m)')
        ax1.set_zlabel('h2 (m)')
        ax1.set_title('3D design space: d1, h1, h2 coloured by V_c')
    
        # Adjust colorbar position
        cbar = fig1.colorbar(sc1, ax=ax1, pad=0.15)
        cbar.set_label('V_c (m3)')
    
    
        # Plot the optimal point with detailed label in legend
        label = f"Optimal design\nh1 = {optimal['h1']:.2f}m, h2 = {optimal['h2']:.2f}m, d1 = {optimal['d1']:.2f}m, V_c = {optimal['V_c']:.2f}m³"
        ax1.scatter(optimal['d1'], optimal['h1'], optimal['h2'], color='red', s=100, label=label)

    
        #ax1.scatter(optimal['d1'], optimal['h1'], optimal['h2'], color='red', s=100, label='Optimal Design')
        
        # Add guide lines to each axis from the optimal point
        # ax1.plot([optimal['d1'], optimal['d1']], [optimal['h1'], optimal['h1']], [ax1.get_zlim()[0], optimal['h2']], 'r--', linewidth=1)
        # ax1.plot([optimal['d1'], optimal['d1']], [ax1.get_ylim()[0], optimal['h1']], [optimal['h2'], optimal['h2']], 'r--', linewidth=1)
        # ax1.plot([ax1.get_xlim()[0], optimal['d1']], [optimal['h1'], optimal['h1']], [optimal['h2'], optimal['h2']], 'r--', linewidth=1)
        

        # Annotate the optimal point

        #annotation_text = f"h1 = {optimal['h1']}m, h2 = {optimal['h2']}m, d1 = {optimal['d1']}m, V_c = {optimal['V_c']}m³"
        #ax1.text(optimal['d1'] + 0.05, optimal['h1'] + 0.05, optimal['h2'] + 0.05, annotation_text, color='black')

        
        ax1.legend()
        figs.append(fig1)
    
        # Add spacing between plots
        plt.subplots_adjust(hspace=1)
    
        # 2D Plot: h2 vs V_c
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(df_results['h2'], df_results['V_c'], c='blue', label='Designs')
        ax2.scatter(optimal['h2'], optimal['V_c'], color='red', s=100, label='Optimal design')
        ax2.set_xlabel('h2 (m)')
        ax2.set_ylabel('V_c (m3)')
        ax2.set_title('h2 vs V_c')
        ax2.grid(True, which='both', linestyle='--')
        ax2.minorticks_on()
        ax2.legend()
        figs.append(fig2)
    
        return figs
    




    def visualise_design_space_frontier(self, df_results, optimal):
        figs = []
    
        # Extract points for convex hull
        points = df_results[['d1', 'h1', 'h2']].values
        hull = ConvexHull(points)
    
        # 3D Plot of the convex hull (frontier)
        fig1 = plt.figure(figsize=(10, 7))
        ax1 = fig1.add_subplot(111, projection='3d')
    
        # Plot the convex hull
        for simplex in hull.simplices:
            triangle = [points[simplex[0]], points[simplex[1]], points[simplex[2]]]
            poly = Poly3DCollection([triangle], alpha=0.3, facecolor='cyan', edgecolor='k')
            ax1.add_collection3d(poly)
    
        # Plot the optimal point
        ax1.scatter(optimal['d1'], optimal['h1'], optimal['h2'], color='red', s=100, label=(
            f"Optimal design\n"
            f"h1 = {optimal['h1']:.2f}m, h2 = {optimal['h2']:.2f}m,\n"
            f"d1 = {optimal['d1']:.2f}m, V_c = {optimal['V_c']:.2f}m³"
        ))
    
        ax1.set_xlabel('d1 (m)')
        ax1.set_ylabel('h1 (m)')
        ax1.set_zlabel('h2 (m)')
        ax1.set_title('Design space frontier (convex hull)')
        
        #fig1.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        ax1.set_box_aspect(None, zoom=0.90)
        plt.tight_layout()

        ax1.legend()
        figs.append(fig1)
    
        return figs








    

        
        
