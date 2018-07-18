% This script generates 4 precision matrices and data corresponding to those
% similar to those used in simulation 
clear all
rng(1)
% Create true precision matrices A1, A2, A3, and A4
K = 4;
p = 20;

% A1 is an AR(2) graph
A1 = toeplitz([1, 0.5, 0.4, zeros(1, p - 3)]);
n_edges = (sum(sum(A1 ~= 0)) - p) / 2;
n_possible = p * (p - 1) / 2;

% Locations of all nonzero entries of A1 above the diagonal
[rposA1, cposA1] = find(triu(A1) - eye(p));

% Locations of all zero entries of A1 above the diagonal
[rzeroA1, czeroA1] = find(triu(A1 == 0));

% Sample locations without replacement to perturb
pos_inds = randsample(size(rposA1, 1), n_edges);
zero_inds = randsample(size(rzeroA1, 1), n_possible - n_edges);
A2 = A1;

% Add 5 new and remove 5 edges to get A2 from A1
for j = 1:5
    % Generate random sign for new entry
    sign = 1;
    if binornd(1, .5) == 0
        sign = -1;
    end
    
    % Generate nonzero value for precision matrix entry
    nonzero_val = unifrnd(.4, .6) * sign;
    A2(rzeroA1(zero_inds(j)), czeroA1(zero_inds(j))) = nonzero_val;
    A2(czeroA1(zero_inds(j)), rzeroA1(zero_inds(j))) = nonzero_val;
    
    % Assign a current nonzero value to be 0
    A2(rposA1(pos_inds(j)), cposA1(pos_inds(j))) = 0;
    A2(cposA1(pos_inds(j)), rposA1(pos_inds(j))) = 0;
end

% Now add 10 and remove 10 more to get A3
A3 = A2;
for j = 6:15
    % Generate random sign for new entry
    sign = 1;
    if binornd(1, .5) == 0
        sign = -1;
    end
    
    % Generate nonzero value for precision matrix entry
    nonzero_val = unifrnd(.4, .6) * sign;
    A3(rzeroA1(zero_inds(j)), czeroA1(zero_inds(j))) = nonzero_val;
    A3(czeroA1(zero_inds(j)), rzeroA1(zero_inds(j))) = nonzero_val;
    
    % Assign a current nonzero value to be 0
    A3(rposA1(pos_inds(j)), cposA1(pos_inds(j))) = 0;
    A3(cposA1(pos_inds(j)), rposA1(pos_inds(j))) = 0;
end

% Now change remaining to get A4
A4 = A3;
for j = 16:20
    % Generate random sign for new entry
    sign = 1;
    if binornd(1, .5) == 0
        sign = -1;
    end
    
    % Generate nonzero value for precision matrix entry
    nonzero_val = unifrnd(.4, .6) * sign;
    A4(rzeroA1(zero_inds(j)), czeroA1(zero_inds(j))) = nonzero_val;
    A4(czeroA1(zero_inds(j)), rzeroA1(zero_inds(j))) = nonzero_val;
    
    % Assign a current nonzero value to be 0
    A4(rposA1(pos_inds(j)), cposA1(pos_inds(j))) = 0;
    A4(cposA1(pos_inds(j)), rposA1(pos_inds(j))) = 0;
end

% Still 37 edges, 5 new, 5 removed from old graph = 10 total different
(sum(sum(A2 ~= 0)) - p) / 2
sum(sum((A2 ~= 0) ~= (A1 ~= 0))) / 2

(sum(sum(A3 ~= 0)) - p) / 2
sum(sum((A3 ~= 0) ~= (A2 ~= 0))) / 2
sum(sum((A3 ~= 0) ~= (A1 ~= 0))) / 2

(sum(sum(A4 ~= 0)) - p) / 2
sum(sum((A4 ~= 0) ~= (A3 ~= 0))) / 2
sum(sum((A4 ~= 0) ~= (A2 ~= 0))) / 2
sum(sum((A4 ~= 0) ~= (A1 ~= 0))) / 2

% No common edges between A1 and A4
(sum(sum((A4 ~= 0) & (A1 ~= 0))) - p) / 2

% Adjust revised matrices to ensure positive definiteness
A2 = fix_matrix(A2, 1);
A3 = fix_matrix(A3, 1);
A4 = fix_matrix(A4, 1);

% Verify that results are positive definite
all(eig(A1) > 0)
all(eig(A2) > 0)
all(eig(A3) > 0)
all(eig(A4) > 0)


Precisions_true = ones(p,p,C);

Precisions_true(:,:,1) = A1;
Precisions_true(:,:,2) = A2;
Precisions_true(:,:,3) = A3;
Precisions_true(:,:,4) = A4;

% Generating sample 
sample = cell(C,1);
S_mat = cell(C,1);
sample_sizes = [20,40,60,80];
rng(1)
for c = 1:C
    sample{c} = zeros(sample_sizes(c),p);
    for kk = 1 : sample_sizes(c)
        sample{c} = mvnrnd(zeros(1,p),inv(Precisions_true(:,:,c)),sample_sizes(c));
        S_mat{c} = transpose(sample{c})*sample{c};
    end
end

csvwrite('Sample_1.csv',sample{1});
csvwrite('Sample_2.csv',sample{2});
csvwrite('Sample_3.csv',sample{3});
csvwrite('Sample_4.csv',sample{4});