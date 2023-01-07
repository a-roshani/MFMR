
#--- STEP 1

MPLNLC.stan <- "
data { 
int<lower=0> A;
int<lower=0> TT;
int<lower=0> dxt[A,TT];
matrix[A,TT] ext;
vector[2]    gamma0;
matrix[2,2]  Sigma0;
vector[TT]   years;
vector[A]    ax;
int          h;
}
parameters {
vector<lower=0,upper=1>[A]     ex;
simplex[A]   Bx;
vector[TT]    Kt_1;
real<lower=0> sigma2_Bx;
real<lower=0> sigma2_Kt;
vector[2]     gam;
real          logit_rho;
}
transformed parameters {
vector<upper=0>[A]    Ax = log(ex);
real         rho = inv_logit(logit_rho);
vector[TT]   Kt= append_row(-sum(Kt_1[2:(TT)]),Kt_1[2:(TT)]);
}
model {
ex               ~ gamma(ax,0.01);
logit_rho        ~ normal(3,0.5);
sigma2_Bx        ~ inv_gamma(2.1,1);
sigma2_Kt        ~ inv_gamma(2.1,1);
Bx               ~ normal(inv(A),sqrt(sigma2_Bx));
gam              ~ multi_normal(gamma0, Sigma0);
Kt_1[1]          ~ normal(gam[1] + gam[2]*years[1] , sqrt(sigma2_Kt/(1-pow(rho,2))));
Kt_1[2:TT]       ~ normal(gam[1]+gam[2]*years[2:TT] + rho*(Kt_1[1:(TT-1)]-gam[1]-gam[2]*years[1:(TT-1)]) , sqrt(sigma2_Kt));
for(x in 1:A) {
for(t in 1:TT){
dxt[x,t]      ~ poisson(ext[x,t]*exp(Ax[x]+Bx[x]*Kt[t]));
}
}
}
generated quantities{
vector[h]   Kt_pred;
Kt_pred[1]  =   normal_rng(gam[1]+gam[2]*(years[TT]+1) + rho*(Kt_1[TT]-gam[1]-gam[2]*years[TT]) , sqrt(sigma2_Kt));
for(i in 2:h){
Kt_pred[i]  =   normal_rng(gam[1]+gam[2]*(years[TT]+i) + rho*(Kt_pred[(i-1)]-gam[1]-gam[2]*(years[TT]+i-1)) ,sqrt(sigma2_Kt));
}
}
"


#--- STEP 2

MPLNLC2.stan <- "
data {
int<lower=0> A;
int<lower=0> TT;
int<lower=0> Dxt[A,TT,2];
real<lower=0> Ext[A,TT,2];
int h;
vector<lower=0>[A] mu_alpha1;
vector<lower=0>[A] mu_alpha2;
vector[A] Ax ;
vector[A] Bx ;
vector[TT] Kt ;
vector[h] Kt_pred;
}
parameters {
vector<lower=0>[A] ex_m;
simplex[A] beta_m;
vector[TT] kappa_1_m;
real<lower=0> sigma2_beta_m;
real<lower=0> sigma2_kappa_m;
real logit_rho_m;
real<lower=0> sigma2_nu1;
real<lower=0> sigma2_nu2;
vector<lower=0>[A] ex_f;
simplex[A] beta_f;
vector[TT] kappa_1_f;
real<lower=0> sigma2_beta_f;
real<lower=0> sigma2_kappa_f;
real logit_rho_f;
real Lmu[A,TT,2] ;
}
transformed parameters {
vector[A]  alpha_m = log(ex_m);
vector[A]  alpha_f = log(ex_f);
real         rho_m = inv_logit(logit_rho_m);
real         rho_f = inv_logit(logit_rho_f);
vector[TT] kappa_m = append_row(-sum(kappa_1_m[2:(TT)]),kappa_1_m[2:(TT)]);
vector[TT] kappa_f = append_row(-sum(kappa_1_f[2:(TT)]),kappa_1_f[2:(TT)]);
}
model {
[sigma2_kappa_m,sigma2_kappa_f] ~ inv_gamma(2.1,1);
sigma2_nu1    ~ inv_gamma(2.1,2);
sigma2_nu2    ~ inv_gamma(2.1,2);
sigma2_beta_m ~ inv_gamma(2.1,0.01);
sigma2_beta_f ~ inv_gamma(2.1,0.1);
ex_m          ~ gamma(mu_alpha1,1);
ex_f          ~ gamma(mu_alpha2,1);
beta_m        ~ normal(inv(A),sqrt(sigma2_beta_m));
beta_f ~ normal(inv(A),sqrt(sigma2_beta_f));
logit_rho_m   ~ normal(0.5,0.5);
logit_rho_f   ~ normal(0.5,0.5);
kappa_1_m[1] ~ normal(0 , sqrt(sigma2_kappa_m/(1-(rho_m)^2)));
kappa_1_m[2:TT] ~ normal(rho_m*kappa_1_m[1:(TT-1)] , sqrt(sigma2_kappa_m));
kappa_1_f[1] ~ normal(0 , sqrt(sigma2_kappa_f/(1-(rho_f)^2)));
kappa_1_f[2:TT] ~ normal(rho_f*kappa_1_f[1:(TT-1)] , sqrt(sigma2_kappa_f));
for(x in 1:A) {
for(t in 1:TT){
Lmu[x,t,1] ~ normal(Ax[x]+Bx[x]*Kt[t]+alpha_m[x]+beta_m[x]*kappa_m[t] , sqrt(sigma2_nu1));
Lmu[x,t,2] ~ normal(Ax[x]+Bx[x]*Kt[t]+alpha_f[x]+beta_f[x]*kappa_f[t] , sqrt(sigma2_nu2));
Dxt[x,t,1] ~ poisson(Ext[x,t,1]*exp(Lmu[x,t,1]));
Dxt[x,t,2] ~ poisson(Ext[x,t,2]*exp(Lmu[x,t,2]));
}
}
}
generated quantities{
matrix[A,TT] log_lik_m;
vector[h] k_pred_m;
matrix[A,TT] log_lik_f;
vector[h] k_pred_f;
matrix[A,h] log_mu_m;
matrix[A,h] log_mu_f;
k_pred_m[1] = normal_rng(rho_m*kappa_1_m[TT] , sqrt(sigma2_kappa_m));
k_pred_f[1] = normal_rng(rho_f*kappa_1_f[TT] , sqrt(sigma2_kappa_f));
for(i in 2:h){
k_pred_m[i] = normal_rng(rho_m*k_pred_m[(i-1)],sqrt(sigma2_kappa_m));
k_pred_f[i] = normal_rng(rho_f*k_pred_f[(i-1)],sqrt(sigma2_kappa_f));
}
for(x in 1:A){
for(t in 1:TT){
log_lik_f[x,t] = poisson_lpmf(Dxt[x,t,2] | Ext[x,t,2]*exp(Lmu[x,t,2]));
log_lik_m[x,t] = poisson_lpmf(Dxt[x,t,1] | Ext[x,t,1]*exp(Lmu[x,t,1]));
}
}
for(x in 1:A){
for(t in 1:h){
log_mu_m[x,t] = normal_rng(Ax[x]+Bx[x]*Kt_pred[t]+alpha_m[x]+beta_m[x]*k_pred_m[t],sqrt(sigma2_nu1));
log_mu_f[x,t] = normal_rng(Ax[x]+Bx[x]*Kt_pred[t]+alpha_f[x]+beta_f[x]*k_pred_f[t],sqrt(sigma2_nu2));
}}
}
"
