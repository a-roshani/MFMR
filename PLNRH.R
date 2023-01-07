PLNRH.stan <- "
data { 
int<lower=0> A;
int<lower=0> TT;
int<lower=0> C;
int<lower=0> Dxt[A,TT];
real<lower=0> Ext[A,TT];
vector[2]    gamma0;
matrix[2,2]  Sigma0;
vector[TT]   years;
int          h;
vector<lower=0>[A]  mu_alpha;
}
parameters {
vector<lower=0>[A]     ex;
simplex[A]    Bx;
simplex[A]    lambda;
vector[TT]    Kt_1;
vector[C]     coh;
real<lower=0> sigma2_Bx;
real<lower=0> sigma2_Kt;
real<lower=0> sigma2_coh;
real<lower=0> sigma2_lambda;
real          omega0;
real          omega1;
vector[2]     gam;
real mean_rho;
real<lower=0> sigma2_rho;
real          logit_rho;
real<lower=0> sigma2_omega0;
real<lower=0> sigma2_omega1;
real<lower=0> sigma2_nu; 
real          Lmu[A,TT] ;
}
transformed parameters {
vector[A]    Ax = log(ex);
real         rho = inv_logit(logit_rho);
vector[TT]   Kt= append_row(-sum(Kt_1[2:(TT)]),Kt_1[2:(TT)]);//append_row(0,Kt_1[2:(TT)]);//
}
model {
ex              ~ gamma(mu_alpha,1);
mean_rho ~ normal(0,10);
sigma2_rho ~ inv_gamma(2.1,2);
logit_rho        ~ normal(mean_rho,sqrt(sigma2_rho));
[sigma2_Bx,sigma2_Kt,sigma2_coh,sigma2_lambda]  ~ inv_gamma(2.1,1);
Bx               ~ normal(inv(A),sqrt(sigma2_Bx));
lambda           ~ normal(inv(A),sqrt(sigma2_lambda));
gam              ~ multi_normal(gamma0, Sigma0);
Kt_1[1]          ~ normal(gam[1] + gam[2]*years[1] , sqrt(sigma2_Kt/(1-pow(rho,2))));
Kt_1[2:TT]       ~ normal(gam[1] + gam[2]*years[2:TT] + rho*(Kt_1[1:(TT-1)]-gam[1]-gam[2]*years[1:(TT-1)]) , sqrt(sigma2_Kt));
sigma2_nu        ~ inv_gamma(2.1,2);
[sigma2_omega0,sigma2_omega1] ~ inv_gamma(2.1,2);
omega0          ~ normal(0,sqrt(sigma2_omega0));
omega1          ~ normal(0,sqrt(sigma2_omega1));
coh[1]          ~ normal(omega0 , sqrt(sigma2_coh));
coh[2:C]        ~ normal(omega0 + omega1*coh[1:(C-1)], sqrt(sigma2_coh));
for(x in 1:A) {
for(t in 1:TT){
Lmu[x,t]      ~ normal(Ax[x]+Bx[x]*Kt[t]+lambda[x]*coh[t-x+A], sqrt(sigma2_nu));
Dxt[x,t]      ~ poisson(Ext[x,t]*exp(Lmu[x,t]));
}
}
}
generated quantities{
vector[h]   Kt_pred;
matrix[A,TT] log_lik;
matrix[A,h] log_mu;
vector[h]   coh_pred;
Kt_pred[1]   =   normal_rng(gam[1]+gam[2]*(years[TT]+1) + rho*(Kt_1[TT]-gam[1]-gam[2]*years[TT]) , sqrt(sigma2_Kt));
coh_pred[1]  =   normal_rng(omega0 + omega1*coh[C],sqrt(sigma2_coh));
for(i in 2:h){
Kt_pred[i]   =   normal_rng(gam[1]+gam[2]*(years[TT]+i) + rho*(Kt_pred[(i-1)]-gam[1]-gam[2]*(years[TT]+i-1)) ,sqrt(sigma2_Kt));
coh_pred[i]  =   normal_rng(omega0 + omega1*coh_pred[(i-1)],sqrt(sigma2_coh));
}
for(x in 1:A){
for(t in 1:TT){
log_lik[x,t] = poisson_lpmf(Dxt[x,t] | Ext[x,t]*exp(Lmu[x,t]));
}
}
for(x in 1:A){
for(t in 1:h){
if(t<x)
log_mu[x,t] = normal_rng(Ax[x]+Bx[x]*Kt_pred[t]+lambda[x]*coh[t-x+C+1],sqrt(sigma2_nu));
else
log_mu[x,t] = normal_rng(Ax[x]+Bx[x]*Kt_pred[t]+lambda[x]*coh_pred[t-x+1],sqrt(sigma2_nu));
}}
}
"
