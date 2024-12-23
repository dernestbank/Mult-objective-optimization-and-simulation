%% rxn_simulation.m
% Multiobjective optimization of PEM electrolyser catalyst loading
% Balances cost (due to catalyst loading) and performance (overpotential).
%
% Author: [Your Name]
% Date: [Date]

clear; clc; close all;

%% Parameters and Constants

% Physical constants
R = 8.314;              % Universal gas constant (J/(mol*K))
F = 96485;              % Faraday's constant (C/mol)
T = 353;                % Temperature (K)
n = 2;                  % Electrons transferred per reaction
alpha = 0.5;            % Charge transfer coefficient

% Catalyst properties
rho_cat = 21.45;        % Catalyst density (g/cm^3), e.g., Pt
S_cat = 50e4;           % Specific surface area (cm^2_active/g)
c_cat = 30;             % Catalyst cost ($/g)
j0 = 1e-6;              % Specific exchange current density (A/cm^2_active)

% Mass transport properties
D = 2.5e-5;             % Diffusivity (cm^2/s)
tau = 2;                % Tortuosity factor (dimensionless)
C_bulk = 0.0555;        % Bulk concentration (mol/cm^3)

% Operating conditions
A_cell = 100;           % Cell active area (cm^2)
J = 1.0;                % Current density (A/cm^2)

% Decision variable bounds
delta_min = 0.5e-4;     % Minimum catalyst layer thickness (cm)
delta_max = 5e-4;       % Maximum catalyst layer thickness (cm)
epsilon_min = 0.3;      % Minimum porosity
epsilon_max = 0.7;      % Maximum porosity

% Overpotential constraint
eta_max = 0.1;          % Maximum allowable overpotential (V)

% Initial guess (midpoint of bounds)
x0 = [ (delta_min + delta_max)/2 , (epsilon_min + epsilon_max)/2 ];

lb = [delta_min, epsilon_min];
ub = [delta_max, epsilon_max];

%% Normalization Factors for Objectives

% Maximum expected cost for normalization:
L_max = rho_cat * delta_max * (1 - epsilon_min);
C_max = L_max * A_cell * c_cat;  % Reference max cost
eta_norm = eta_max;              % Reference max overpotential

%% Optimization Settings
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');

%% Generate Pareto Front by varying weights
num_points = 20;
w1_values = linspace(0, 1, num_points);
w2_values = 1 - w1_values;

cost_values = zeros(num_points, 1);
overpotential_values = zeros(num_points, 1);
delta_values = zeros(num_points, 1);
epsilon_values = zeros(num_points, 1);

for i = 1:num_points
    w1 = w1_values(i);
    w2 = w2_values(i);

    % Objective Function
    fun = @(x) objective_function(x, w1, w2, R, T, alpha, n, F, j0, S_cat, J, ...
                                  c_cat, A_cell, rho_cat, D, tau, C_bulk, C_max, eta_norm);

    % Nonlinear Constraint
    nonlcon = @(x) overpotential_constraint(x, R, T, alpha, n, F, j0, S_cat, J, ...
                                            c_cat, A_cell, rho_cat, D, tau, C_bulk, eta_max);

    % Solve optimization
    [x_opt, ~] = fmincon(fun, x0, [], [], [], [], lb, ub, nonlcon, options);

    % Extract results
    delta_opt = x_opt(1);
    epsilon_opt = x_opt(2);

    [C_opt, eta_total_opt] = calculate_cost_overpotential(x_opt, R, T, alpha, n, F, j0, ...
                                                          S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk);

    cost_values(i) = C_opt;
    overpotential_values(i) = eta_total_opt;
    delta_values(i) = delta_opt;
    epsilon_values(i) = epsilon_opt;

    fprintf('Iteration %d: w1=%.2f, w2=%.2f, Cost=$%.2f, Overpotential=%.4f V\n', ...
            i, w1, w2, C_opt, eta_total_opt);
end

%% Visualization

% 1. Pareto Front (Cost vs. Overpotential)
figure;
plot(cost_values, overpotential_values, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
xlabel('Cost ($)', 'FontSize', 12);
ylabel('Overpotential (V)', 'FontSize', 12);
title('Pareto Front for Catalyst Loading Optimization', 'FontSize', 14);
grid on;

% 2. 3D Scatter Plot: Cost, Overpotential, and Thickness (delta)
figure;
scatter3(cost_values, overpotential_values, delta_values, 70, 'filled');
xlabel('Cost ($)', 'FontSize', 12);
ylabel('Overpotential (V)', 'FontSize', 12);
zlabel('Thickness \delta (cm)', 'FontSize', 12);
title('3D Visualization of Pareto Front (Thickness)', 'FontSize', 14);
grid on;
view(135, 30);  % Adjust view angle

% 3. Color-Coded Scatter: Cost vs. Overpotential Colored by Porosity (epsilon)
figure;
scatter(cost_values, overpotential_values, 70, epsilon_values, 'filled');
xlabel('Cost ($)', 'FontSize', 12);
ylabel('Overpotential (V)', 'FontSize', 12);
title('Pareto Front Colored by Porosity \epsilon', 'FontSize', 14);
colorbar;  
colormap(jet);
grid on;

set(gcf, 'Color', 'w'); % Make figure background white

%% Functions

function Z = objective_function(x, w1, w2, R, T, alpha, n, F, j0, S_cat, J, ...
                                c_cat, A_cell, rho_cat, D, tau, C_bulk, C_norm, eta_norm)
    % Calculate cost and overpotential
    [C, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, S_cat, ...
                                                  J, c_cat, A_cell, rho_cat, D, tau, C_bulk);

    C_normalized = C / C_norm;
    eta_normalized = eta_total / eta_norm;

    % Weighted sum objective
    Z = w1 * C_normalized + w2 * eta_normalized;
end

function [C, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, ...
                                                       S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk)
    delta = x(1);
    epsilon = x(2);

    % Catalyst loading
    L = rho_cat * delta * (1 - epsilon);

    % Cost
    C = L * A_cell * c_cat;

    % Exchange current density
    J0 = j0 * L * S_cat;  
    if J0 <= 0
        J0 = 1e-14; % Prevent log(J/J0) from being invalid
    end

    % Activation overpotential
    eta_act = (R * T) / (alpha * n * F) * log(J / J0);

    % Effective diffusivity
    D_eff = D * (epsilon / tau);
    if D_eff <= 0
        D_eff = 1e-14; % Avoid division by zero
    end

    % Concentration overpotential
    C_surface = C_bulk - (J * delta) / (n * F * D_eff);
    if C_surface <= 0
        C_surface = 1e-10; % Small positive number to avoid log(<=0)
    end

    eta_conc = (R * T) / (n * F) * log(C_bulk / C_surface);

    eta_total = eta_act + eta_conc;
end

function [c, ceq] = overpotential_constraint(x, R, T, alpha, n, F, j0, S_cat, J, ...
                                              c_cat, A_cell, rho_cat, D, tau, C_bulk, eta_max)
    % Calculate overpotential
    [~, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, ...
                                                  S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk);

    % Constraint: eta_total should not exceed eta_max
    c = eta_total - eta_max;
    ceq = [];
end
