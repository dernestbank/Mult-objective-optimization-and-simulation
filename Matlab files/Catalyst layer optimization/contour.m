%% Full Integrated PEM Electrolyser Optimization Script with Contour Plot

clear; clc; close all;

%% Parameters and Constants
R = 8.314;     % J/(mol*K)
F = 96485;     % C/mol
T = 353;       % K
n = 2;         % electrons transferred
alpha = 0.5;   % charge transfer coefficient

%% Catalyst and Electrode Properties
% Cathode (Pt)
rho_cat_cath = 21.45;   % g/cm^3 (Pt)
S_cat_cath = 50e4;      % cm^2_active/g (Pt)
c_cat_cath = 30;        % $/g (Pt)

% Anode (IrO2)
rho_cat_an = 11.66;     % g/cm^3 (IrO2)
S_cat_an = 20e4;        % cm^2_active/g (IrO2)
c_cat_an = 50;          % $/g (IrO2)

% Mass transport
D = 2.5e-5;    % cm^2/s
tau = 2;        % dimensionless
C_bulk = 0.0555; % mol/cm^3
A_cell = 100;   % cm^2

% Bounds on design variables (for optimization)
delta_min = 0.5e-4;
delta_max = 5e-4;
epsilon_min = 0.3;
epsilon_max = 0.7;
x0 = [(delta_min+delta_max)/2, (epsilon_min+epsilon_max)/2];
lb = [delta_min, epsilon_min];
ub = [delta_max, epsilon_max];

% Overpotential constraint
eta_max = 0.1;

% Current densities for optimization
J_fixed = 1.0;  % Example current density for contour plotting

% For demonstration, let's generate the data grid over delta and epsilon
num_delta = 100;
num_epsilon = 100;
delta_array = linspace(delta_min, delta_max, num_delta);
epsilon_array = linspace(epsilon_min, epsilon_max, num_epsilon);

Overpot_list = [];
Loading_list = [];
Cost_list = [];

%% Parameter Sweep over Delta and Epsilon
for i = 1:num_epsilon
    for j = 1:num_delta
        delta_test = delta_array(j);
        epsilon_test = epsilon_array(i);

        [C_test, eta_test, ~, ~] = calculate_cost_overpotential([delta_test, epsilon_test], ...
            R, T, alpha, n, F, S_cat_cath, S_cat_an, J_fixed, c_cat_cath, c_cat_an, ...
            A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk);

        % Compute total catalyst loading
        L_cath = rho_cat_cath * delta_test * (1 - epsilon_test);
        L_an = rho_cat_an * delta_test * (1 - epsilon_test);
        L_total = L_cath + L_an;

        % Store the scattered data
        Overpot_list(end+1,1) = eta_test;
        Loading_list(end+1,1) = L_total;
        Cost_list(end+1,1) = C_test;
    end
end

%% Define a Rectangular Grid in Overpotential-Loading Space
OP_min = min(Overpot_list);
OP_max = max(Overpot_list);
Load_min = min(Loading_list);
Load_max = max(Loading_list);

% Create a full rectangular grid
num_OP_points = 100;
num_Load_points = 100;
OP_lin = linspace(OP_min, OP_max, num_OP_points);
Load_lin = linspace(Load_min, Load_max, num_Load_points);
[OP_grid, Load_grid] = meshgrid(OP_lin, Load_lin);

%% Interpolate Cost onto the (Overpotential, Loading) Grid
% We have scattered data (Overpot_list, Loading_list, Cost_list)
Cost_grid = griddata(Overpot_list, Loading_list, Cost_list, OP_grid, Load_grid, 'linear');

%% Contour Plot
figure;
contourf(OP_grid, Load_grid, Cost_grid, 20, 'LineColor','none');
colorbar;
colormap(parula);
xlabel('Total Overpotential (V)');
ylabel('Catalyst Loading (g/cm²)');
title(sprintf('Cost Contour at J=%.2f A/cm^2', J_fixed));
axis tight;
set(gca, 'FontSize', 14, 'FontWeight', 'bold');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to Calculate Cost and Overpotential
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [C, eta_total, C_cath, C_an] = calculate_cost_overpotential(x, R, T, alpha, n, F, ...
                                                       S_cat_cath, S_cat_an, J, c_cat_cath, c_cat_an, ...
                                                       A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk)
    % Extract design variables (thickness and porosity)
    delta = x(1);   % Catalyst layer thickness (cm)
    epsilon = x(2); % Catalyst layer porosity (fraction)

    % Calculate catalyst loading for cathode and anode
    L_cath = rho_cat_cath * delta * (1 - epsilon); % Cathode catalyst loading (g/cm²)
    L_an   = rho_cat_an   * delta * (1 - epsilon); % Anode catalyst loading (g/cm²)

    % Calculate individual costs for cathode and anode
    C_cath = L_cath * A_cell * c_cat_cath; % Cost of cathode catalyst ($)
    C_an = L_an * A_cell * c_cat_an; % Cost of anode catalyst ($)

    % Total cost
    C = C_cath + C_an;

    % Exchange current densities for cathode and anode (simplified)
    j0_base_cath = 1e-6; % Base exchange current for cathode (A/cm²)
    j0_cath = j0_base_cath * L_cath * S_cat_cath; % Exchange current for cathode (A/cm²)

    j0_base_an = 5e-7;   % Base exchange current for anode (A/cm²)
    j0_an = j0_base_an * L_an * S_cat_an; % Exchange current for anode (A/cm²)

    % Effective exchange current density (geometric mean of cathode and anode)
    J0_eff = sqrt(j0_cath * j0_an);
    if J0_eff <= 0
        J0_eff = 1e-14; % Avoid zero or negative values
    end

    % Activation overpotential (Tafel equation)
    eta_act = (R*T)/(alpha*n*F) * log(J / J0_eff);

    % Effective diffusivity (assuming it's affected by porosity)
    D_eff = D * (epsilon / tau);
    if D_eff <= 0
        D_eff = 1e-14; % Avoid zero or negative values
    end

    % Concentration overpotential (simplified)
    C_surface = C_bulk - (J * delta) / (n * F * D_eff); % Surface concentration
    if C_surface <= 0
        C_surface = 1e-10; % Avoid negative or zero surface concentration
    end
    eta_conc = (R*T) / (n*F) * log(C_bulk / C_surface);

    % Total overpotential (sum of activation and concentration overpotentials)
    eta_total = eta_act + eta_conc;
end
