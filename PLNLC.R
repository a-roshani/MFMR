PCFM0.stan <- "
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
vector[TT]    Kt_1;
real<lower=0> sigma2_Bx;
real<lower=0> sigma2_Kt;
vector[2]     gam;
real mean_rho;
real<lower=0> sigma2_rho;
real          logit_rho;
real<lower=0> sigma2_nu; 
real          Lmu[A,TT] ;
}
transformed parameters {
vector[A]    Ax = log(ex);
real         rho = inv_logit(logit_rho);
vector[TT]   Kt= append_row(-sum(Kt_1[2:(TT)]),Kt_1[2:(TT)]);
}
model {
ex          ~ gamma(mu_alpha,1);
mean_rho    ~ normal(0,10);
sigma2_rho  ~ inv_gamma(2.1,2);
logit_rho   ~ normal(mean_rho,sqrt(sigma2_rho));
[sigma2_Bx,sigma2_Kt]  ~ inv_gamma(2.1,1);
Bx          ~ normal(inv(A),sqrt(sigma2_Bx));
gam         ~ multi_normal(gamma0, Sigma0);
Kt_1[1]     ~ normal(gam[1] + gam[2]*years[1] , sqrt(sigma2_Kt/(1-pow(rho,2))));
Kt_1[2:TT]  ~ normal(gam[1] + gam[2]*years[2:TT] + rho*(Kt_1[1:(TT-1)]-gam[1]-gam[2]*years[1:(TT-1)]) , sqrt(sigma2_Kt));
sigma2_nu   ~ inv_gamma(2.1,2);
for(x in 1:A) {
for(t in 1:TT){
Lmu[x,t]    ~ normal(Ax[x]+Bx[x]*Kt[t], sqrt(sigma2_nu));
Dxt[x,t]    ~ poisson(Ext[x,t]*exp(Lmu[x,t]));
}
}
}
generated quantities{
vector[h]   Kt_pred;
matrix[A,TT] log_lik;
matrix[A,h] log_mu;
Kt_pred[1]   =   normal_rng(gam[1]+gam[2]*(years[TT]+1) + rho*(Kt_1[TT]-gam[1]-gam[2]*years[TT]) , sqrt(sigma2_Kt));
for(i in 2:h){
Kt_pred[i]   =   normal_rng(gam[1]+gam[2]*(years[TT]+i) + rho*(Kt_pred[(i-1)]-gam[1]-gam[2]*(years[TT]+i-1)) ,sqrt(sigma2_Kt));
}
for(x in 1:A){
for(t in 1:TT){
log_lik[x,t] = poisson_lpmf(Dxt[x,t] | Ext[x,t]*exp(Lmu[x,t]));
}
}
for(x in 1:A){
for(t in 1:h){
log_mu[x,t] = normal_rng(Ax[x]+Bx[x]*Kt_pred[t],sqrt(sigma2_nu));
}}
}
"
