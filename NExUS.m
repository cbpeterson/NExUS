clear all
rng(1)
no_iter = 200;
burnin  = 50; nmc = no_iter - burnin;
cut_off = 0.05;

%%% reading sample
%%% (In case of real data, note that it is recommended to normalize the 
%%% sample corresponding to each variable in the data to have mean 0 and
%%% variance 1.)

C = 4;
sample = cell(C,1);
sample{1} = csvread('Sample_1.csv');
sample{2} = csvread('Sample_2.csv');
sample{3} = csvread('Sample_3.csv');
sample{4} = csvread('Sample_4.csv');


p = size(sample{1},2);

sample_sizes = zeros(C,1);
for c = 1:C
    sample_sizes(c) = size(sample{c},1);
end

%%% Calculating sample matrix

S_mat = cell(C,1);
for c = 1:C
    S_mat{c} = transpose(sample{c})*sample{c};
end

    
%%% Precision matrices initial values

Precisions_est = zeros(p,p,C);
for c = 1:C
    Precisions_est(:,:,c) = eye(p);
end
Partial_corr_per_iter = cell(C,1);
for c = 1:C
    Partial_corr_per_iter{c} = zeros(p,p,no_iter);
    Partial_corr_per_iter{c}(:,:,1) = Precisions_est(:,:,c);
end

%%% delta, alpha_1,beta_1,alpha_2,beta_2 initial values

delta = 0.5;
alpha_1_default = 1;
beta_1_default = .1;
alpha_2_default = .1;
beta_2_default = 1;
n_bar = mean(sample_sizes);
transformed_sample_sizes= zeros(C,1);
for c = 1:C
    transformed_sample_sizes(c) = (n_bar^delta)*sample_sizes(c)^(1-delta);
end
alpha_1 = alpha_1_default;
beta_1  = beta_1_default*(n_bar^2)*ones(C,1)./(transformed_sample_sizes).^2;

beta_2_factor = zeros(C,C);
for c = 1:C
    for c_prime = 1:C
        beta_2_factor(c,c_prime) = 4*(transformed_sample_sizes(c)*transformed_sample_sizes(c_prime))^2/...
            ((transformed_sample_sizes(c)+transformed_sample_sizes(c_prime))^2);
    end
end
alpha_2 = alpha_2_default*ones(C,C);
beta_2  = beta_2_default*(n_bar^2)*ones(C,C)./beta_2_factor;


%%% lambda_1,lambda_2 initial values

lambda_1_square = alpha_1./beta_1;
lambda_2_square = alpha_2./beta_2;
lambda_1_square_graph = zeros(C,no_iter);
lambda_2_square_graph = zeros(C,C,no_iter);
lambda_1_square_graph(:,1) = lambda_1_square;
lambda_2_square_graph(:,:,1) = lambda_2_square;

%%% alpha_gamma,beta_gamma, gamma initial values

alpha_gamma = 1;
beta_gamma = 1;
gamma = 1;

%%% inv_tau_square and inv_omega_square initial values

INV_TAU_SQUARE = zeros(p,p,C);

for c = 1:C
    for i = 1:(p-1)
        for j = (i+1):p
            INV_TAU_SQUARE(i,j,c) = 1/exprnd(2/lambda_1_square(c));
            INV_TAU_SQUARE(j,i,c) = INV_TAU_SQUARE(i,j,c);
        end
    end
end

INV_OMEGA_SQUARE = zeros(p,p,C,C);

for c = 1:(C-1)
    for c_prime = (c+1):C
        for i = 1:(p-1)
            for j = (i+1):p
                INV_OMEGA_SQUARE(i,j,c,c_prime) = 1/exprnd(2/lambda_2_square(c,c_prime));
                INV_OMEGA_SQUARE(j,i,c,c_prime) = INV_OMEGA_SQUARE(i,j,c,c_prime);
            end
        end
        INV_OMEGA_SQUARE(:,:,c_prime,c) = INV_OMEGA_SQUARE(:,:,c,c_prime);
    end
end


%%% MCMC LOOP

for iter = 2:no_iter
    iter
    transformed_sample_sizes= zeros(C,1);
    for c = 1:C
        transformed_sample_sizes(c) = (n_bar^delta)*sample_sizes(c)^(1-delta);
    end
    
    beta_1  = beta_1_default*(n_bar^2)*ones(C,1)./(transformed_sample_sizes).^2;
    
    beta_2_factor = zeros(C,C);
    for c = 1:C
        for c_prime = 1:C
            beta_2_factor(c,c_prime) = 4*(transformed_sample_sizes(c)*transformed_sample_sizes(c_prime))^2/...
                ((transformed_sample_sizes(c)+transformed_sample_sizes(c_prime))^2);
        end
    end
    beta_2  = beta_2_default*(n_bar^2)*ones(C,C)./beta_2_factor;
    
    
    %%% Update Precisions
    
    sum_diags = 0;
    for c = 1:C
        theta_mat = Precisions_est(:,:,c);
        Precisions_est_here = Precisions_est;
        S_mat_here = S_mat{c};
        INV_TAU_SQUARE_here = INV_TAU_SQUARE(:,:,c);
        INV_OMEGA_SQUARE_here_for_c = zeros(p,p,C);
        for c_prime = 1:C
            INV_OMEGA_SQUARE_here_for_c(:,:,c_prime) = INV_OMEGA_SQUARE(:,:,c,c_prime);
        end
        
        for i = 1:p
            if(i == 1)
                perm = 1:p;
            else
                perm = [p,1:(p-1)];
            end
            
            % permutations
            theta_mat = theta_mat(perm,perm);
            Precisions_est_here = Precisions_est_here(perm,perm,:);
            S_mat_here = S_mat_here(perm,perm);
            INV_TAU_SQUARE_here = INV_TAU_SQUARE_here(perm,perm);
            INV_OMEGA_SQUARE_here_for_c = INV_OMEGA_SQUARE_here_for_c(perm,perm,:);
            
            
            % calculations
            B_vec = zeros(p-1,1);
            for jj = 1:(p-1)
                for cc = 1:C
                    if(cc ~= c)
                        B_vec(jj) = B_vec(jj) - 2*Precisions_est_here(jj,p,cc)*INV_OMEGA_SQUARE_here_for_c(jj,p,cc);
                    end
                end
            end
            A_vec = zeros(p-1,1);
            for jj = 1:(p-1)
                A_vec(jj) = INV_TAU_SQUARE_here(jj,p);
                for cc = 1:C
                    A_vec(jj) = A_vec(jj) + INV_OMEGA_SQUARE_here_for_c(jj,p,cc);
                end
            end
            D_A = diag(A_vec);
            Sigma = inv(D_A + (2*gamma+S_mat_here(p,p))*inv(theta_mat(1:(p-1),1:(p-1))));
            epsilon = gamrnd(sample_sizes(c)/2+1,2/(S_mat_here(p,p)+2*gamma));
            psi = mvnrnd(-Sigma*(2*S_mat_here(1:(p-1),p)+B_vec)/2,Sigma,1);
            theta_mat(1:(p-1),p) = psi;
            theta_mat(p, 1:(p-1)) = psi;
            theta_mat(p,p) = epsilon + psi*inv(theta_mat(1:(p-1),1:(p-1)))*transpose(psi);
        end
        perm = [p,1:(p-1)];
        theta_mat = theta_mat(perm,perm);
        if(min(eig(theta_mat))<0)
            stop
        end
        Precisions_est(:,:,c) = theta_mat;
        sum_diags = sum_diags + sum(diag(theta_mat));
        Partial_corr_per_iter{c}(:,:,iter) = theta_mat;
        for gg = 1:p
            for hh = gg:p
                Partial_corr_per_iter{c}(gg,hh,iter) = Partial_corr_per_iter{c}(gg,hh,iter)...
                    /sqrt(Partial_corr_per_iter{c}(gg,gg,iter)*Partial_corr_per_iter{c}(hh,hh,iter));
                Partial_corr_per_iter{c}(hh,gg,iter) = Partial_corr_per_iter{c}(gg,hh,iter);
            end
        end
    end
    
    
    
    %%% Update 1/T^2
    
    for c = 1:C
        for i = 1:(p-1)
            for j = (i+1):p
                mu = sqrt(lambda_1_square(c)/(Precisions_est(i,j,c))^2);
                lambda = lambda_1_square(c);
                pd = makedist('InverseGaussian','mu',mu,'lambda',lambda);
                random_here = random(pd);
                if(random_here>0)
                    INV_TAU_SQUARE(i,j,c) = random_here;
                    INV_TAU_SQUARE(j,i,c) = INV_TAU_SQUARE(i,j,c);
                end
            end
        end
    end
    
    %%% Update 1/W^2
    
    for c = 1:(C-1)
        for c_prime = (c+1):C
            for i = 1:(p-1)
                for j = (i+1):p
                    mu = sqrt(lambda_2_square(c,c_prime)/(Precisions_est(i,j,c)-Precisions_est(i,j,c_prime))^2);
                    lambda = lambda_2_square(c,c_prime);
                    pd = makedist('InverseGaussian','mu',mu,'lambda',lambda);
                    random_here = random(pd);
                    if(random_here>0)
                        INV_OMEGA_SQUARE(i,j,c,c_prime) = random_here;
                        INV_OMEGA_SQUARE(j,i,c,c_prime) = INV_OMEGA_SQUARE(i,j,c,c_prime);
                    end
                end
            end
            INV_OMEGA_SQUARE(:,:,c_prime,c) = INV_OMEGA_SQUARE(:,:,c,c_prime);
        end
    end
    
    
    %%% Update lambda_1_square
    for c = 1 : C
        A = 1./INV_TAU_SQUARE(:,:,c);
        sum_here = 0;
        for i = 1:(p-1)
            for j = (i+1):p
                sum_here = sum_here + A(i,j);
            end
        end
        lambda_1_square(c) = gamrnd(p*(p-1)/2 + alpha_1, 1/(sum_here/2 + beta_1(c)));
    end
    
    lambda_1_square_graph(:,iter) = lambda_1_square;
    
    %%% Update lambda_2_square
    for c = 1:(C-1)
        for c_prime = (c+1):C
            AA = 1./INV_OMEGA_SQUARE(:,:,c,c_prime);
            sum_here_2 = 0;
            for i = 1:(p-1)
                for j = (i+1):p
                    sum_here_2 = sum_here_2 + AA(i,j);
                end
            end
            lambda_2_square(c,c_prime) = gamrnd(p*(p-1)/2 + alpha_2(c,c_prime), 1/(sum_here_2/2 + beta_2(c,c_prime)));
            lambda_2_square(c_prime,c) = lambda_2_square(c,c_prime);
        end
    end
    
    lambda_2_square_graph(:,:,iter) = lambda_2_square;
    
    
    %%% Update gamma
    
    gamma = gamrnd(alpha_gamma+C*p, 1/(beta_gamma+sum_diags));
end



%%% Precisions mean (before cut off)

Partial_corr_mean_TEMP = zeros(p,p,C);
for c = 1:C
    for iter = (burnin+1) : no_iter
        Partial_corr_mean_TEMP(:,:,c) = Partial_corr_mean_TEMP(:,:,c) + Partial_corr_per_iter{c}(:,:,iter);
    end
    Partial_corr_mean_TEMP(:,:,c) = Partial_corr_mean_TEMP(:,:,c)/(no_iter - burnin);
end

%%%  Sparsity from prob matrices 

zero_one_array = cell(C,1);
prob_array_mean = cell(C,1);
prob_matrix = cell(C,1);
prob_matrix_array = cell(C,1);


%%% Final estimated Precision Matrices

Partial_corr_mean = Partial_corr_mean_TEMP;

for c = 1:C
    prob_array = zeros(p^2,no_iter);
    for iter = (burnin+1):no_iter
        temporary = reshape(Partial_corr_per_iter{c}(:,:,iter),[1,p^2]);
        for i = 1:(p^2)
            if(abs(temporary(i)) > cut_off)
                prob_array(i,iter) = 1;
            else
                prob_array(i,iter) = 0;
            end
        end
    end
    prob_array_mean{c} = mean(prob_array(:,(burnin+1):no_iter),2);
    
    
    %%% Probability matrix cut-off decision here %%%%%%%
    
    prob_matrix{c} = reshape(prob_array_mean{c},[p,p]);
    for i = 1:p
        for j = 1:p
            if(prob_matrix{c}(i,j)> 0.5 && abs(Partial_corr_mean_TEMP(i,j,c)) > cut_off)
                Partial_corr_mean(i,j,c) = Partial_corr_mean_TEMP(i,j,c);
            else
                Partial_corr_mean(i,j,c) = 0;
            end
        end
    end
end

%%% Lambda means

lambda_1_square_mean = zeros(C,1);
for c = 1:C
    lambda_1_square_mean(c) = mean(lambda_1_square_graph(c,(burnin+1):no_iter));
end

lambda_2_square_mean = zeros(C,C);
for c = 1:C
    for c_prime = 1:C
        lambda_2_square_mean(c,c_prime) = mean(lambda_2_square_graph(c,c_prime,(burnin+1):no_iter));
    end
    lambda_2_square_mean(c,c) = NaN;
end


Partial_corr_mean
lambda_2_square_mean

