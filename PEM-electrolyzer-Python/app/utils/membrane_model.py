# model.py
import numpy as np

class PEMModel:
    def __init__(self):
        # Thermodynamic & performance constants
        self.HHV_H2 = 285000.0  # Higher heating value of hydrogen (J/mol)
        self.alpha = 1e-6       # Production rate scaling factor (mol/(s·(A/cm²)))
        self.beta = 100.0       # Scaling factor for production rate dependence on thickness
        
        # Voltage model parameters (simplified)
        self.V_base = 1.3       # Base cell voltage (V)
        self.k1 = 0.005         # Coefficient for current density contribution (V per (A/cm²))
        self.k2 = 0.1           # Coefficient for inverse thickness term (V·m)
        
        # Durability parameters (lifetime in hours)
        self.L_base = 10000.0   # Baseline lifetime (hours)
        self.k3 = 10.0          # Additional lifetime per micron of thickness (h/µm)
        
        # Cost and environmental parameters
        self.c_ionomer = 300.0  # Cost of ionomer ($/kg)
        self.rho = 2000.0       # Density of membrane material (kg/m³)
        self.c_manuf = 20.0     # Manufacturing cost ($/m²)
        self.c_E = 10.0         # Environmental impact factor (kg CO₂-eq per kg)
        
        # Mechanical durability constraint parameters
        # For demonstration, assume that the effective mechanical constraint gives:
        self.t_mech_min = 158e-6  # Minimum allowable thickness (m), e.g., 158 microns

    def evaluate_objectives(self, x):
        """
        Evaluate the four objective functions at decision vector x.
        x[0] = t: Membrane thickness (m)
        x[1] = j: Current density (A/cm²)
        
        Returns a numpy array [f1, f2, f3, f4] where:
          f1 = -η_energy   (we minimize the negative of energy efficiency)
          f2 = -L          (we minimize the negative of lifetime)
          f3 = C_capital   (capital cost per unit area)
          f4 = E_material  (environmental impact per unit area)
        """
        t = x[0]
        j = x[1]
        
        # Simplified cell voltage model (V)
        V_cell = self.V_base + self.k1 * j + self.k2 / t
        
        # Hydrogen production rate (mol/s), decreasing with increasing t (due to ohmic effects)
        r_H2 = self.alpha * j / (1.0 + self.beta * t)
        
        # Energy efficiency: η = (HHV_H2 * r_H2) / (V_cell * I_cell)
        # For simplicity, assume I_cell = j (with area normalized to 1 cm²)
        eta_energy = (self.HHV_H2 * r_H2) / (V_cell * j)
        # Since r_H2 is proportional to j, j cancels out:
        # η_energy = (HHV_H2 * alpha) / ((1+beta*t) * V_cell)
        
        # Lifetime: assume lifetime increases linearly with thickness (converted to microns)
        L = self.L_base + self.k3 * (t * 1e6)
        
        # Capital cost per unit area ($/m²): material cost + manufacturing cost.
        # Material cost = c_ionomer * density * thickness
        C_capital = self.c_ionomer * self.rho * t + self.c_manuf
        
        # Environmental impact per unit area (kg CO₂-eq/m²)
        E_material = self.c_E * self.rho * t
        
        # For maximization objectives, we minimize their negatives.
        f1 = -eta_energy  # maximize efficiency -> minimize negative efficiency
        f2 = -L           # maximize lifetime -> minimize negative lifetime
        
        return np.array([f1, f2, C_capital, E_material])
    
    def evaluate_constraints(self, x):
        """
        Evaluate the constraint functions at decision vector x.
        Here we enforce the mechanical durability constraint:
          t - t_mech_min >= 0.
        Returns a numpy array of constraint values.
        (A solution is feasible if all constraint values are >= 0.)
        """
        t = x[0]
        g1 = t - self.t_mech_min  # Must be non-negative
        return np.array([g1])
