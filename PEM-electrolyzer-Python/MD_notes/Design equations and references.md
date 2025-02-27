


# Electrochemical Module

$$
V_{cell}=E_{0} +η_{act}+η_{Ω}+η_{conc}
$$
where ηact, and ηconc are activation, ohmic, and concentration overpotentials,
Where:

- Vcell​ is the cell voltage
- Erev​ is the reversible cell voltage
- ηactηact​ is the activation overpotential
- ηohmηohm​ is the ohmic overpotential [4](https://www.mdpi.com/1996-1073/13/24/6556)


## Open circuit Voltage
A simplified Nerst equation

$$V_{oc}=\frac{∆G_R^O }{z.F}+\frac{R.T}{z.F} ln(\frac{p_{H_2}  {p_{O_{2}}^0.5}}{α_{H_2 O} })$$
- E0 is the standard cell potential
- R is the gas constant
- T is temperature
- z is the number of electrons transferred (2 for water electrolysis)
- F is Faraday's constant
- pH2​​ and pO2​​ are partial pressures of hydrogen and oxygen
- aH<sub>2​</sub>O​ is the activity of water[ 6 ](https://www.mdpi.com/2311-5629/6/2/29)

is the standard Gibs free energy of the electrolysis reaction z is the number of electrons involved in the reaction(2 for hydrogen), F is Faraday’s constant, R is the gas constant, 𝛼 is the activity coefficient.


## The activation (Faradaic losses) voltages
The ηact has contributions from both anode and cathode i.e.$$ ηact=η_{act}^{anode}+η_{act}^{anode} $$
Using _the Butler–Volmer equation_ , η±act (– for anode and + for cathode)
$$
η_{act}^±=  \frac{RT}{(2α_± )} ln⁡(\frac{j}{j_{o,±}} )
$$
where R is the ideal gas constant, T is the operating temperature, α± is the charge transfer coefficient (assumed to be 0.5 ), j is the cell current density (i.e. the cell current normalized with respect to the electrode cross–sectional area), and j0,± is the exchange current density
[ref][https://www.mdpi.com/2311-5629/6/2/29]

the cathode is significantly faster than the kinetics of the oxygen evolution reaction at the anode (Espinosa-López et al., 2018; García-Valverde et al., 2012)
for exchange current density:

  $$i_{o,an} =K_{io, an}.exp⁡(\frac {E_{act,an}}{R.T})$$

Kio is the pre-exponential factor and Eact is the activation energy of the reaction at the anode.=2160000A/cm2 or 

the activation energy for the anode (Eact,anode) is equal to 76,000 J/mol (Crespi et al., 2023)

Pressure and Temperature dependence

$$
i_0​=i_{0,ref}​ (\frac{P{O_2}}​{P_{O_2​​​,ref}} )^γ exp(\frac{-ΔG}{RT_c​​}(1−\frac{T_c​​}{T_{ref}​}))
$$
![[Pasted image 20250115154149.png]]
Where:

- P_{O_2} is the oxygen partial pressure
- γ is a pressure coefficient
- ΔG is the Gibbs free energy change[ 5](https://digital.csic.es/bitstream/10261/304811/1/Intl%20J%20of%20Energy%20Research%20-%202022%20-%20Aguilar%20-%20Control%E2%80%90oriented%20estimation%20of%20the%20exchange%20current%20density%20in%20PEM%20fuel%20cells.pdf) 
4. Catalyst Layer Properties:  
    The exchange current density is influenced by catalyst layer properties:

$$ i_{0}=i_{0,ref}⋅RF $$
Where RF is the roughness factor of the catalyst layer, representing the ratio of actual surface area to geometric surface area

```
Influence of Porosity and Thickness
In porous electrodes, the effective exchange current density can be modified by considering the porosity and thickness of the catalyst layer. The exchange current density per geometric area can be expressed as:
$$
i_{o,eff}= i_o.\frac{\epsilon}{\delta}
$$

- i0,eff​ = effective exchange current density per geometric area
- ϵϵ = porosity of the catalyst layer (fractional volume)
- δδ = thickness of the catalyst layer (cm)

**catalyst loading on Limiting Current Density**:
Increasing catalyst loading can enhance the limiting current density (ilimilim​) up to a certain point. This is because more catalyst increases the number of active sites available for reactions, thus allowing higher current densities

 -  For instance, in one study, it was noted that at lower ionomer-to-carbon (I/C) ratios, the thickness of the catalyst layer (δδ) was linearly dependent on carbon loading (LCLC​), which in turn is related to platinum loading (LPtLPt​) [1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11082850/).

δCL∝LC

```

$$
i_{o,eff}= i_o.\frac{\epsilon}{\delta}
$$


## **The concentration overpotential (ΔV conc),**

considering only the anode side, where the dominant contribution is present (Colbertaldo et al., 2017; García-Valverde et al., 2012)
https://youtu.be/9FsHJFowgAo


$$
V_{con}=\frac{R.T}{z.α_{an}.F} (ln( \frac{j_L}{(j_L-i)}))
$$
α_an is the charge transfer coefficient of the reaction occurring at the anode and iL the limiting current density.
The limiting current density is assumed equal to 6 A/cm2 (Bessarabov & Millet, 2018)

$$
V_{con}=\frac{R.T}{z.α_{an}.F} (ln( \frac{C_b}{C_s}))
$$
Where:

- R is the gas constant
- T is the temperature
- n is the number of electrons transferred
- F is Faraday's constant
- C_b is the bulk concentration of the reactant
- C_s is the surface concentration of the reactant

the **limiting current density**
j_lim: Limiting current density. This represents the maximum current density that can be achieved when the reaction rate is limited by mass transport. the limiting current density can be derived from Fick's laws of diffusion and is given by:

$$ i_{lim} = \frac{nFDc}{L} $$
Where:

- n = number of electrons transferred per reaction (typically 2 for water electrolysis)
- F = Faraday's constant (approximately 96485 C/mol)
- D = diffusion coefficient of the reactant (e.g., water)
- c = concentration of the reactant (e.g., water concentration)
- L = thickness of the porous transport layer (PTL)
[ref][[Experimental assessment and analysis of mass transport limiting current density in water vapor-fed polymer electrolyte membrane electrolyzers | Scientific Reports](https://www.nature.com/articles/s41598-024-79935-6)]

The binary diffusion coefficient of any two substances can be extrapolated from reference values for a given temperature and pressure,

$$D= D^{ref} (\frac{T}{_{ref}} )^{2.33} (\frac{P_(ref)}{P})$$ Where _D__ref_ is a reference diffusivity, _T_ is the cell temperature, _T__ref_ is the reference temperature, _p__ref_ is reference pressure, and _p_ is absolute pressure.


Porosity and tortuosity relate effective transport properties, K eff, with bulk transport properties, K through the following relation
![[Pasted image 20250113040600.png]]


[Insights into Interfacial and Bulk Transport Phenomena Affecting Proton Exchange Membrane Water Electrolyzer Performance at Ultra‐Low Iridium Loadings - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8564452/)






## **The ohmic overpotential (ΔVohm)**

The ohmic overpotential is related with the materials resistance to the protons flux. The magnitude of ohmic losses depends on the materials properties.

$$V_{ohm,an}=V_{ohm,an}   +V_{ohm,ca}   +V_{ohm,mem}$$$
$$∆V_{ohm}=(R_{el,an}+R_{el,cat}+R_{mem} )i.A_{cell}$$

$$ V_{ohm,an}=V_{ohm,cat}=ρ_{el}.t_{el}.I$$

$$V_{ohm,mem}=\frac{t_{mem}}{σ_{mem}} . i$$
$$R_{el,X}=\frac{t_{el,X}}{A_{cell}}  ρ_{el,X}$$

$$R_{mem}=\frac{t_{mem}}{σ_{mem}.A_cell }$$
ρ_el is the material resistivity of the electrode, σ_memis the membrane resistivity, t_el is the electrode thickness, and t_mem is the membrane thickness.
The thickness of the membrane (tmem), which a sensitivity analysis shows as the main term influencing the ohmic loss and therefore the cell polarization curve

σ_{mem}=(0.005139λ-0.00326).exp⁡(1268(1/303-1/T))

λ Represents the number of water molecules per sulfonic acid site in the membrane (SO3H)


![[Pasted image 20250115154402.png]]


**Calculation of the local volumetric current density in the cathode catalyst layer**

local volumetric current density(i<sub>v</sub>), A/cm3 
$$i_{v} = \frac{\eta_{HER}}{\delta .R_{K, HER}}$$
where δ is the cathode catalyst layer thickness, ηHER is the kinetic overpotential and RK, HER is the charge transfer resistance for the HER. [re][https://doi.org/10.1149/2.0641805jes.]

RK, HER =hcarge transfer resistance 
$$
R_{K,HER} = \frac{R.T}{(\alpha_{a}+\alpha_{c}.F.L_{Pt}.A_{Pt}.i_{o,HER})}$$
where R is the gas constant, T is the temperature, (αa + αc) is the sum of the anodic and cathodic transfer coefficients, F is the Faraday constant, LPt is the Pt loading, APt is the specific Pt surface area, and i0, HER is the exchange current density of the HER.

\eta
The kinetic overpotential ηHER
the derivations made by Thompson et al. for the hydrogen oxidation reaction: [ref][https://doi.org/10.1149/1.2943203.]
$$
\eta_{HER}|x = \frac{i}{s.k_{eff}}. \frac{cosh(s(\delta-x))}{sinh(s.\delta)}$$
i is the geometric current density, κeff is the effective proton conductivity of the cathode catalyst layer, and s is a kinetic parameter:

$$S= [\frac{1}{K_{eff}.\delta}.\frac{1}{R_{K,HER}}]^{0.5}$$
The effective proton conductivity κeff

$$K_eff =\frac{1}{\rho{H+,cath}}$$
where ρH+, cath is the effective proton resistivity of the cathode catalyst layer, and is estimated to be ~ 25 Ohm·cm for an I/C ratio of 0.69 and ~ 60 Ohm·cm for an I/C ratio of 0.35
[ref][https://doi.org/10.1149/1.3435323.]
![[Pasted image 20250120111654.png]]

Zhang, Z.; Baudy, A.; Testino, A.; Gubler, L. Cathode Catalyst Layer Design in PEM Water Electrolysis toward Reduced Pt Loading and Hydrogen Crossover. _ACS Appl Mater Interfaces_ **2024**, _16_ (18), 23265–23277. [https://doi.org/10.1021/acsami.4c01827](https://doi.org/10.1021/acsami.4c01827).



Bruggeman core

 the tortuosity factor model applies the tortuosity factor with includes the relation on porosity in this case  
 transport efficiency"  B  
 B=ϵ/τ
 =ϵ/ϵ−1/2=ϵ3/2 .


