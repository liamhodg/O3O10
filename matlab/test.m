t_alpha_plus = load('eta_zeta.mat','T_alpha').T_alpha;
t_omega_plus = load('eta_zeta.mat','T_omega').T_omega;
eps = load('eta_zeta.mat','eps').eps;

t_alpha = t_alpha_plus - eps;
t_omega = t_omega_plus - eps;

figure;
x = linspace(0,t_alpha,100);
plot(x, eta(x));
xlabel('t')
ylabel('eta(t)')
xlim([0, t_alpha])

figure;
x = linspace(0,t_omega,100);
plot(x, zeta(x));
xlabel('t')
ylabel('zeta(t)')
xlim([0, t_omega])