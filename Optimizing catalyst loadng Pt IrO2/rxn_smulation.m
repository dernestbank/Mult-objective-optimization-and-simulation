% MATLAB Script for Multiobjective Optimization of Catalyst Loading in PEM Electrolyser


% Clear workspace and command window
clear; clc; close all;

%% Parameters and Constants

% Physical Constants
R = 8.314;            % Universal gas constant (J/mol·K)
F = 96485;            % Faraday's constant (C/mol)
T = 353;              % Operating temperature (K)
n = 2;                % Number of electrons transferred
alpha = 0.5;          % Charge transfer coefficient

% Catalyst Properties
rho_cat = 21.45;      % Catalyst density (g/cm^3), e.g., Platinum
S_cat = 50e4;         % Specific surface area (cm^2_active/g)
c_cat = 30;           % Catalyst cost ($/g)
j0 = 1e-6;            % Specific exchange current density (A/cm^2_active)

% Mass Transport Properties
D = 2.5e-5;           % Molecular diffusivity (cm^2/s)
tau = 2;              % Tortuosity factor (dimensionless)
C_bulk = 0.0555;      % Bulk concentration of water (mol/cm^3)

% Operating Conditions
A_cell = 100;         % Active cell area (cm^2)
J = 1.0;              % Current density (A/cm^2)

%% Decision Variables Bounds

% Catalyst layer thickness (cm)
delta_min = 0.5e-4;
delta_max = 5e-4;

% Porosity (dimensionless)
epsilon_min = 0.3;
epsilon_max = 0.7;

% Tortuosity (assumed constant)
tau = 2;

% Catalyst loading (g/cm^2)
% Calculated based on delta and epsilon during optimization

%% Constraints

eta_max = 0.1;        % Maximum allowable overpotential (V)

%% Normalization Factors for Objective Functions

% Maximum expected cost (for normalization)
L_max = rho_cat * delta_max * (1 - epsilon_min);
C_max = L_max * A_cell * c_cat;

% Maximum overpotential (for normalization)
eta_norm = eta_max;

%% Optimization Setup

% Initial Guess for [delta, epsilon]
x0 = [(delta_min + delta_max)/2, (epsilon_min + epsilon_max)/2];

% Variable Bounds
lb = [delta_min, epsilon_min];
ub = [delta_max, epsilon_max];

% Optimization Options
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');

%% Weighting Factors for Multiobjective Optimization

% Number of points on Pareto front
num_points = 20;
w1_values = linspace(0, 1, num_points);
w2_values = 1 - w1_values;

% Arrays to store optimization results
cost_values = zeros(num_points, 1);
overpotential_values = zeros(num_points, 1);
delta_values = zeros(num_points, 1);
epsilon_values = zeros(num_points, 1);

%% Optimization Loop to Generate Pareto Front

for i = 1:num_points
    w1 = w1_values(i);
    w2 = w2_values(i);
    
    % Composite Objective Function
    fun = @(x) objective_function(x, w1, w2, R, T, alpha, n, F, j0, S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk, C_max, eta_norm);
    
    % Nonlinear Constraint Function
    nonlcon = @(x) overpotential_constraint(x, R, T, alpha, n, F, j0, S_cat, J, D, tau, C_bulk, eta_max);
    
    % Run Optimization
    [x_opt, ~] = fmincon(fun, x0, [], [], [], [], lb, ub, nonlcon, options);
    
    % Store Results
    delta_opt = x_opt(1);
    epsilon_opt = x_opt(2);
    
    % Calculate Cost and Overpotential
    [C_opt, eta_total_opt] = calculate_cost_overpotential(x_opt, R, T, alpha, n, F, j0, S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk);
    
    cost_values(i) = C_opt;
    overpotential_values(i) = eta_total_opt;
    delta_values(i) = delta_opt;
    epsilon_values(i) = epsilon_opt;
    
    fprintf('Iteration %d: w1=%.2f, w2=%.2f, Cost=$%.2f, Overpotential=%.4f V\n', i, w1, w2, C_opt, eta_total_opt);
end

%% Plot Pareto Front

figure;
plot(cost_values, overpotential_values, '-o', 'LineWidth', 2);
xlabel('Cost ($)');
ylabel('Overpotential (V)');
title('Pareto Front for Catalyst Loading Optimization');
grid on;

%% Functions

function Z = objective_function(x, w1, w2, R, T, alpha, n, F, j0, S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk, C_norm, eta_norm)
    % Decision Variables
    delta = x(1);
    epsilon = x(2);
    
    % Calculate Cost and Overpotential
    [C, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk);
    
    % Normalize Objectives
    C_norm_factor = C_norm;
    eta_norm_factor = eta_norm;
    
    C_normalized = C / C_norm_factor;
    eta_normalized = eta_total / eta_norm_factor;
    
    % Composite Objective Function
    Z = w1 * C_normalized + w2 * eta_normalized;
end

function [C, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk)
    % Decision Variables
    delta = x(1);
    epsilon = x(2);
    
    % Catalyst Loading (g/cm^2)
    L = rho_cat * delta * (1 - epsilon);
    
    % Cost Calculation
    C = L * A_cell * c_cat;
    
    % Activation Overpotential
    J0 = j0 * L * S_cat;  % Exchange current density (A/cm^2)
    eta_act = (R * T) / (alpha * n * F) * log(J / J0);
    
    % Effective Diffusivity
    D_eff = D * (epsilon / tau);
    
    % Concentration Overpotential
    term = J * delta / (n * F * D_eff);
    C_surface = C_bulk - term;
    if C_surface <= 0
        C_surface = 1e-10;  % Prevent negative concentration
    end
    eta_conc = (R * T) / (n * F) * log(C_bulk / C_surface);
    
    % Total Overpotential
    eta_total = eta_act + eta_conc;
end

function [c, ceq] = overpotential_constraint(x, R, T, alpha, n, F, j0, S_cat, J, D, tau, C_bulk, eta_max)
    % Nonlinear Inequality Constraints (c <= 0)
    % Overpotential should be less than or equal to eta_max
    
    % Calculate Overpotential
    [~, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, S_cat, J, 0, 0, 0, D, tau, C_bulk);
    
    % Constraint: eta_total - eta_max <= 0
    c = eta_total - eta_max;
    ceq = [];
end





%%%%%%plot

%% Visualization of Simulation Optimization

% 1. Pareto Front (Cost vs. Overpotential)
figure;
plot(cost_values, overpotential_values, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
xlabel('Cost ($)', 'FontSize', 12);
ylabel('Overpotential (V)', 'FontSize', 12);
title('Pareto Front for Catalyst Loading Optimization', 'FontSize', 14);
grid on;

% 2. 3D Scatter Plot: Visualize Cost, Overpotential, and Thickness (delta)
figure;
scatter3(cost_values, overpotential_values, delta_values, 70, 'filled');
xlabel('Cost ($)', 'FontSize', 12);
ylabel('Overpotential (V)', 'FontSize', 12);
zlabel('Thickness \delta (cm)', 'FontSize', 12);
title('3D Visualization of Pareto Front (Thickness)', 'FontSize', 14);
grid on;
view(135, 30);  % Adjust view angle for better 3D perspective

% 3. Color-Coded Scatter: Cost vs. Overpotential Colored by Porosity
figure;
scatter(cost_values, overpotential_values, 70, epsilon_values, 'filled');
xlabel('Cost ($)', 'FontSize', 12);
ylabel('Overpotential (V)', 'FontSize', 12);
title('Pareto Front Colored by Porosity \epsilon', 'FontSize', 14);
colorbar;  % Show porosity color scale
colormap(jet);
grid on;

% Optional: Enhance readability
set(gcf, 'Color', 'w');  % Set figure background to white for clarity
