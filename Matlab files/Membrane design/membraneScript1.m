%% Multi-Objective Optimization for PEM Electrolyzer Membrane

% Clear workspace
clear; clc; close all;

%% Parameters (example values)
L0 = 175;           % µm
kappa0 = 0.09;      % S/cm
beta_x = 2;         % conductivity enhancement factor per unit x
gamma_L = 0.001;    % conductivity degradation factor per µm below L0
i = 1.0;            % A/cm^2
A = 25;             % cm^2
E_th = 1.48;        % V
eta_other = 0.25;   % V
D_gas = 2e-6;        % cm^2/s
Delta_p = 0.1;       % bar (example)
F = 96485;          
n = 2;             
t_base = 1.0;       
L_crit = 100;       % µm
alpha = 0.05;       % 1/µm
delta = 0.5;        % lifetime doping sensitivity

% Cost parameters (example)
c_base = 1e-3;      % $/µm
c_d = 5e-3;         % $/µm per doping fraction x
c_mg = 0.5;         
epsilon_x = 0.5;     
epsilon_L = 0.2;     

FE_req = 0.95;       % Faradaic efficiency requirement
t_req = 0.8 * t_base; % Lifetime requirement

L_min = 50; L_max = 300;
x_min = 0; x_max = 0.1;

%% Objective and Constraint Functions

% Objective function for gamultiobj:
% f = [ -Pi(L,x), C(L,x), -t_life(L,x) ]
fun = @(vars) objFun(vars, L0, kappa0, beta_x, gamma_L, i, A, ...
    E_th, eta_other, D_gas, Delta_p, F, n, t_base, L_crit, alpha, delta, ...
    c_base, c_d, c_mg, epsilon_x, epsilon_L);

% Nonlinear constraints: FE(L) >= FE_req and t_life(L,x) >= t_req
nonlcon = @(vars) nonlConstraints(vars, L0, kappa0, beta_x, gamma_L, i, A, ...
    E_th, eta_other, D_gas, Delta_p, F, n, t_base, L_crit, alpha, delta, ...
    c_base, c_d, c_mg, epsilon_x, epsilon_L, FE_req, t_req);

% Bounds
lb = [L_min, x_min];
ub = [L_max, x_max];

%% Run Multi-Objective Optimization
options = optimoptions('gamultiobj','Display','iter','PlotFcn',{@gaplotpareto});
[sol,fval] = gamultiobj(fun,2,[],[],[],[],lb,ub,nonlcon,options);

%% Visualization of Pareto Front
figure;
plot3(fval(:,1),fval(:,2),fval(:,3),'o');
xlabel('-Pi(L,x)'); ylabel('C(L,x)'); zlabel('-t_{life}(L,x)');
title('Pareto Front Solutions');
grid on;

%% Define Nested Functions

function f = objFun(vars, L0, kappa0, beta_x, gamma_L, i, A, ...
    E_th, eta_other, D_gas, Delta_p, F, n, t_base, L_crit, alpha, delta, ...
    c_base, c_d, c_mg, epsilon_x, epsilon_L)

L = vars(1);
x = vars(2);

% Conductivity
if L >= L0
    fL = 1;
else
    fL = 1 - gamma_L*(L0 - L);
end
kappa = kappa0*(1 + beta_x*x)*fL;

% Ohmic loss
R_mem = (L*1e-4)/(kappa*A); 
eta_ohm = i*R_mem;
eta_cell = E_th/(E_th + eta_other + eta_ohm);

% FE
J_crossover = (D_gas/(L*1e-4))*Delta_p; 
FE = 1 - ((n*F*J_crossover)/(i*A));

Pi = eta_cell*FE;

% Lifetime
t_life = t_base * exp(-alpha*(L_crit - L))*exp(-delta*x);

% Cost
C = c_base*L + c_d*x*L + c_mg*(1+epsilon_x*x)*(1+epsilon_L*(L0 - L)/L0);

% Objectives: minimize [-Pi, C, -t_life]
f = [-Pi, C, -t_life];
end

function [c,ceq] = nonlConstraints(vars, L0, kappa0, beta_x, gamma_L, i, A, ...
    E_th, eta_other, D_gas, Delta_p, F, n, t_base, L_crit, alpha, delta, ...
    c_base, c_d, c_mg, epsilon_x, epsilon_L, FE_req, t_req)

L = vars(1);

% Conductivity
if L >= L0
    fL = 1;
else
    fL = 1 - gamma_L*(L0 - L);
end
kappa = kappa0*(1 + beta_x*vars(2))*fL;

R_mem = (L*1e-4)/(kappa*A); 
eta_ohm = i*R_mem;
eta_cell = E_th/(E_th + eta_other + eta_ohm);

% FE
J_crossover = (D_gas/(L*1e-4))*Delta_p; 
FE = 1 - ((n*F*J_crossover)/(i*A));

% Lifetime
t_life = t_base * exp(-alpha*(L_crit - L))*exp(-delta*vars(2));

% Nonlinear constraints:
% FE(L) >= FE_req  =>  FE(L) - FE_req >= 0
% t_life(L,x) >= t_req => t_life(L,x) - t_req >= 0
c = [FE_req - FE; t_req - t_life];  % we move them to enforce FE >= FE_req and t_life >= t_req
ceq = [];
end
