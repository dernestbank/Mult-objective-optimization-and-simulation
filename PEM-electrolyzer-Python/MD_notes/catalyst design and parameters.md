alpha: charge transfer coeffcents
	The average charge transfer coefficient for PEM electrolysers typically ranges from 0.5 to 1[2](https://www.ijee.net/article_184897_e8d1ff3e5ebdd469806b0ad46f29d754.pdf)[7](https://www.researchgate.net/figure/Charge-transfer-coefficients-estimated-from-PEMFC-polarization-curves_tbl1_238118590). For the anode (oxidation reaction), the charge transfer coefficient is often around 2, while for the cathode (reduction reaction), it is approximately 0.5[1](https://discovery.ucl.ac.uk/id/eprint/10140633/3/Maier_ReviewMassTransportPEMWE_Reviewed_Unmarked.pdf). 
	![[Pasted image 20250113041757.png]]
	 # An alternative way of estimating anodic and cathodic transfer coefficients from PEMFC polarization curves: http://dx.doi.org/10.1016/j.cej.2009.09.022
		1. It affects the activation overpotential, which is crucial for initiating electrochemical reactions[     5 ](https://www.mdpi.com/2072-666X/14/12/2234).
		2. Changes in the charge transfer coefficient can impact the overall voltage and power output of the electrolyser[   2      ](https://www.ijee.net/article_184897_e8d1ff3e5ebdd469806b0ad46f29d754.pdf).
		3. The coefficient influences the kinetics of the electrode processes and can be affected by electrode and catalyst microstructure and morphology[    1    ](https://discovery.ucl.ac.uk/id/eprint/10140633/3/Maier_ReviewMassTransportPEMWE_Reviewed_Unmarked.pdf).
		[Investigation of the effect of charge transfer coefficient (CTC) on the operating voltage of polymer electrolyte membrane (PEM) electrolyzer - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360319918309017?via%3Dihub)
		Increase in charge transfer coefficient (CTC) reduced activation [overvoltage](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/overvoltage).- •  Higher operating temperature enhances electro-kinetics of the reactions.
		- •      CTC remains the same even at balanced and unbalanced pressure.
		- •      CTC also depends on the properties of the electrode
		From <[https://www.sciencedirect.com/science/article/abs/pii/S0360319918309017?via%3Dihub](https://www.sciencedirect.com/science/article/abs/pii/S0360319918309017?via%3Dihub)>![[Pasted image 20250113131644.png]]


C_bulk_a: bulk concentration" in PEM electrolyzers

	**Anode (OER Side):**

	- **Reactant:** Water (H₂O)
	**Bulk Concentration:** Here, "bulk" refers to the concentration of water within the porous transport layer (PTL) adjacent to the catalyst layer. Since liquid water is fed to the anode, the concentration is relatively high. Approximating it as pure water at room temperature:
	 Density of water ≈ 1 g/cm³   	Molar mass of water ≈ 18 g/mol
		- Concentration ≈ (1 g/cm³) / (18 g/mol) ≈ **0.056 mol/cm³** (or 56 mol/L)
    
	- **Important Note:** This is a simplification. The actual concentration can vary slightly due to factors like temperature, pressure, and water transport phenomena within the PTL.


	**Cathode (HER Side):**

	- **Reactant:** Protons (H⁺) transported across the membrane.
	-**Bulk" Concentration:** Defining a true "bulk" concentration of protons on the cathode side is tricky. Protons are not present in a traditional "bulk" solution like water on the anode side. Instead, they are transported through the solid polymer electrolyte membrane and become available at the catalyst surface for the HER

	**Key Differences and Considerations:**

	- **Phase:** The "bulk" species on the anode side is liquid water, while on the cathode side, we're concerned with protons within the solid membrane and gaseous hydrogen in the pores.

	**In Summary:**

	- **Anode (Water):** Approximately **0.056 mol/cm³** (with the caveat of slight variations).
	- **Cathode (Protons):** Typically **0.001-0.002 mol/cm³** within the membrane, but not a "bulk" concentration in the same way as on the anode side.

**Proton Concentration in the Membrane:** 
	The proton concentration within the membrane itself is a crucial factor. This depends on the membrane material (e.g., Nafion), its equivalent weight (EW), and its water content (λ). A typical range for proton concentration in Nafion membranes is **1-2 mol/L** (or **0.001-0.002 mol/cm³**). However, this is the concentration within the membrane, not in a distinct "bulk" phase on the cathode side.
	
	DOI:[10.1016/j.coche.2016.02.006](http://dx.doi.org/10.1016/j.coche.2016.02.006)

A_cell: active area of cell

	**Small-Scale/Laboratory Cells:**
	
	- These typically have active areas ranging from **a few cm² to around 100 cm²**.
	- Examples: 1 cm², 5 cm², 25 cm², 50 cm², 100 cm²
	- Used for research, material testing, and small-scale hydrogen production.
	
	**Commercial/Industrial Stacks:**
	
	- These have much larger active areas, typically ranging from **several hundred cm² to over 1000 cm²**.
	- Examples: 200 cm², 500 cm², 1000 cm², 1250 cm²
	- Used for larger-scale hydrogen production for industrial applications, hydrogen refueling stations, and grid-scale energy storage


D_a, Bulk difusivity in water 
	**Bulk Diffusivity of Water Vapor (D):** 0.26 cm²/s at 80°C and 1 atm. This value can be found in literature or calculated using kinetic theory. Note that this is the diffusivity of water _vapor_ in air, as water is transported in both liquid and vapor phases in the anode PTL. PEM electrolysers varies depending on the specific conditions and materials used
	https://vbn.aau.dk/ws/portalfiles/portal/293736112/energies_11_03273_v2.pdf

	**Fuller-Schettler-Giddings equation:**

	**D_AB = (0.00143 * T^1.75) / (P * M_AB^0.5 * [(Σv_A)^(1/3) + (Σv_B)^(1/3)]^2)**

	Where:
	
	- **D_AB:** Bulk diffusivity of species A (water vapor) in species B (air) (cm²/s)
	- **T:** Temperature (K)
	- **P:** Pressure (atm)
	- **M_AB:** Molecular weight of the A-B mixture, calculated as: M_AB = 2 / [(1/M_A) + (1/M_B)]
	- **Σv_A and Σv_B:** Diffusion volumes for species A and B, respectively. These are tabulated values that account for the molecular size and shape of the diffusing species.
	
	**Calculation Example (at 80°C and 1 atm):**
	
	- **T:** 80°C + 273.15 = 353.15 K
	- **P:** 1 atm
	- **M_A (Water):** 18.015 g/mol
	- **M_B (Air):** Approximately 28.97 g/mol (average molecular weight of air)
	- **M_AB:** 2 / [(1/18.015) + (1/28.97)] ≈ 22.1 g/mol
	- **Σv_A (Water):** 13.1 (from Fuller-Schettler-Giddings tables)
	- **Σv_B (Air):** 19.7 (from Fuller-Schettler-Giddings tables)
	
	Plugging these values into the Fuller-Schettler-Giddings equation:
	
	D_AB = (0.00143 * 353.15^1.75) / (1 * 22.1^0.5 * [(13.1)^(1/3) + (19.7)^(1/3)]^2) ≈ **0.29 cm²/s**

	Bulk diffusion
	Bulk diffusion at the anode" refers to how easily water molecules can move through the porous materials on the anode side of the electrolyzer to reach the catalyst sites where they are split into oxygen, protons, and electrons. 2×10−5cm2​/s

Tau: tortousity
	## Anode Tortuosity
	
	For the anode of PEM electrolysers, the tortuosity values typically range from 1.27 to 2.13[1]https://pmc.ncbi.nlm.nih.gov/articles/PMC8564452/). This range is based on measurements of different porous transport layers (PTLs) used in PEM electrolysers. Specifically:
		- Lower tortuosity values (1.27-1.59) are associated with higher porosity PTLs (60-75% porosity)
	- Higher tortuosity values (1.75-2.13) are found in PTLs with lower porosity (54-55% porosity)
	
	## Cathode Tortuosity
		The search results do not provide specific tortuosity values for the cathode of PEM electrolysers. However, we can infer that the cathode tortuosity would likely be in a similar range to the anode, as both electrodes use porous transport layers.


- # Properties of catalysts

- # Anode
	- ### IrO2
		- Density
			- - The density of IrO2 is approximately **11.7 g/cm³**[  1  ](https://www.nature.com/articles/s41598-023-30960-x)[  3 ](https://www.riyngroup.com/iridiumiv-oxide-12030-49-8-iro2-15393098000223863.html)[ 7  ](https://pmc.ncbi.nlm.nih.gov/articles/PMC7341534/).
		-  Cost
			- The cost of IrO2 varies depending on purity and supplier. Prices range from **$20 to $80 per kilogram**, with some listings indicating prices as low as **$40 to $150** for lower quantities[2](https://www.alibaba.com/showroom/iridium-oxide.html)[3](https://www.riyngroup.com/iridiumiv-oxide-12030-49-8-iro2-15393098000223863.html)[4](https://www.indiamart.com/proddetail/iridium-oxide-catalyst-iro2-2850547598033.html).
		-  Exchange Current Density (j₀)
		- - The exchange current density (j₀) for IrO2 catalysts can vary based on specific formulations and operating conditions.  j₀ typically ranges from **0.1 to 10 mA/cm²** for high-performance IrO2 electrocatalysts used in PEM electrolysis applications.
	- Ruo2
		-  Density
			- The density of RuO2 is approximately **6.97 g/cm³** at 25 °C [ 7  ](https://www.samaterials.com/ruthenium-oxide-nanopowder-catalyst.html).
		 - Cost
			 The cost of RuO2 varies based on purity and supplier. Prices typically range from **$42.65 for 0.5g** to **$1869 for 25g** of high-purity RuO2 [ 8 ](https://www.chemicalbook.com/Price/Ruthenium-dioxide.htm). For larger quantities, prices can vary significantly depending on the supplier and specific product specifications.
		- Exchange Current Density (j₀) 
			- for RuO2 typically ranges from **0.1 to 10 mA/cm²**
- ## Cathode
	- Pt
		-  Density
	
			- The density of platinum is approximately **21.45 g/cm³** at room temperature [  4    ](https://pubchem.ncbi.nlm.nih.gov/compound/Platinum).
	    -  Cost
		    - - The cost of platinum varies significantly based on purity and market conditions. As of recent data, prices for platinum range from **$900 to $1,900 per kilogram** [  2    ](https://www.alibaba.com/showroom/platinum-catalyst-price.html). For smaller quantities, prices can be around **$27.30 to $59.00 per gram** [  2 ](https://www.alibaba.com/showroom/platinum-catalyst-price.html).
		-  Exchange Current Density (j₀)
			- The exchange current density (j₀) for platinum is typically around **3.1 A/cm²** (or expressed in logarithmic form, approximately −log⁡10(A/cm2)=3.1−log10​(A/cm2)=3.1) [3 ](https://en.wikipedia.org/wiki/Exchange_current_density).


	- Pt-Ru 
		- Density
			The density of Pt-Ru catalysts can vary depending on the specific composition and preparation method, but it typically ranges from 10 to 15 g/cm³
		-  Cost
			- The cost of Pt-Ru catalysts varies based on the ratio of platinum to ruthenium and the supplier. For example, a common formulation of Pt-Ru (50% Pt, 50% Ru) is priced around **$315 to $377 per 25g**. Additionally, the price of ruthenium is approximately ***$17.80 per gram***, which can influence the overall cost of the catalyst depending on its composition.
	
		-  Exchange Current Density (j₀)
			- The exchange current density (j₀) for Pt-Ru catalysts can vary widely based on specific formulations and conditions. Generally, j₀ values for Pt-Ru catalysts used in electrochemical applications can range from ***0.1 to 450 mA/cm²**,*

# Constraints Bounds parameters
-  Porosity (ε)

	Definition: The fraction of void space in the catalyst layer, allowing for transport of reactants and products. 
	**Rationale:**
		Sufficient porosity is needed for water transport at the anode and hydrogen removal at the cathode.
		Too high porosity can compromise mechanical stability and electrical conductivity.
		Cathode layers can often tolerate slightly higher porosity due to less demanding liquid water transport.
	**Practical Ranges:**

	- **Anode:** 0.3 - 0.6 (30% - 60%)
	- **Cathode:** 0.3 - 0.7 (30% - 70%)
- Catalyst Layer Thickness (L)
- 
	Definition: The thickness of the active catalyst layer where the electrochemical reactions occur.
	**Units:** Micrometers (µm)
	**Rationale:**
	 Thicker layers provide more active sites but increase transport resistance.
	 Thinner layers have better transport but may limit the reaction rate due to fewer active sites.
	 Cathode layers can sometimes be thicker due to faster kinetics of the HER compared to the OER.
	
    - **Anode:** 5 - 20 µm
    - **Cathode:** 5 - 30 µm
- ## Catalyst Layer Porosity
Definition: The fraction of void space in the catalyst layer, allowing for transport of reactants and products.
Rationale:
Sufficient porosity is needed for water transport at the anode and hydrogen removal at the cathode.
Too high porosity can compromise mechanical stability and electrical conductivity.
Cathode layers can often tolerate slightly higher porosity due to less demanding liquid water transport.
- Range: 35-50%[1](https://www.mdpi.com/1996-1073/15/18/6657)[ 4](https://elib.dlr.de/186997/1/2022_03_14-CCL_structure_submitted.pdf)
- Unit: Percentage (%)Practical Ranges:
- Anode: 0.3 - 0.6 (30% - 60%)
- Cathode: 0.3 - 0.7 (30% - 70%)

## Catalyst Layer Thickness,  sigma
Rationale:
Thicker layers provide more active sites but increase transport resistance.
Thinner layers have better transport but may limit the reaction rate due to fewer active sites.
Cathode layers can sometimes be thicker due to faster kinetics of the HER compared to the OER.
- Range: 4-10 μm[4](https://elib.dlr.de/186997/1/2022_03_14-CCL_structure_submitted.pdf)
- Unit: Micrometers (μm)
- Anode: 5 - 20 µm
- Cathode: 5 - 30 µm

## Effective Surface Area
Definition: The electrochemically active surface area of the catalyst available for reactions, considering factors like particle size, shape, and agglomeration. 
Rationale:
Higher ESA means more active sites and potentially higher reaction rates.
ESA depends heavily on catalyst material, particle size, and layer fabrication.
Iridium (anode catalyst) is often used in higher surface area forms to maximize its utilization due to its cost.
- Range: This is typically expressed as ECSA (Electrochemically Active Surface Area)
- For cathode (Pt-based): 15-80 m²/g[ 4](https://elib.dlr.de/186997/1/2022_03_14-CCL_structure_submitted.pdf)[5](https://www.energy.gov/eere/fuelcells/technical-targets-proton-exchange-membrane-electrolysis)
- Unit: Square meters per gram of catalyst (m²/g)

## Specific Surface Area
Definition: The total surface area of the catalyst material per unit mass, including both internal and external surfaces.Rationale:
SSA is a material property, while ESA is a property of the catalyst layer as fabricated.
Higher SSA is desirable but doesn't always translate directly to higher ESA due to factors like particle agglomeration and pore blockage in the layer.
- Range: This depends on the type of carbon support used
- For high surface area carbon supports: 200-800 m²/g
- For low surface area carbon supports: 50-100 m²/g[ 4](https://elib.dlr.de/186997/1/2022_03_14-CCL_structure_submitted.pdf)
- Unit: Square meters per gram of support material (m²/g)
- Anode: 100 - 300 m²/g <sub>Ir</sub>
- Cathode: 50 - 200 m²/g <sub>Pt</sub>

## Additional Considerations

1. Platinum Group Metal (PGM) Content:
    
    - Total PGM content (both electrodes combined): 0.125-3.0 mg/cm²[ 5](https://www.energy.gov/eere/fuelcells/technical-targets-proton-exchange-membrane-electrolysis)
    - Unit: Milligrams per square centimeter (mg/cm²)
    
2. Catalyst Loading:
    
    - Typical range: 0.025-0.2 mg/cm²[4](https://elib.dlr.de/186997/1/2022_03_14-CCL_structure_submitted.pdf)[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC11082850/)
    - Unit: Milligrams of catalyst per square centimeter (mg/cm²)
    
3. Ionomer to Carbon (I/C) Ratio:
    
    - Range: 0.35-0.69[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC11082850/)
    - Unit: Mass ratio (dimensionless)
    
4. Platinum Weight Percentage in Pt/C Catalyst:
    
    - Range: 10-40 wt%[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC11082850/)
    - Unit: Weight percentage (%)






