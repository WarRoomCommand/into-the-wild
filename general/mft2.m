% declare hyperparameters
GAMMA = 1;

% load data
file_path = "../scotus/";
states = csvread(file_path + "processed_data.csv");
[num_actors, num_states] = size(states);

% construct p_vec and p_mat
p_vec = sum(transpose(states)) ./ num_states;
p_mat = (states * transpose(states)) ./ num_states;
p_vec_ext = repmat(p_vec, num_actors, 1);

% construct C matrix
corr = csvread(file_path + "correlations.csv");
C_inv = inv(C);

% gets the eigenvalues/eigenvectors
[~, S, V] = svd(C); % S is the eigenvalues, V is eigenvectors
[num_ev, ~] = size(S);

% normalizes S
for i = 1:num_ev
    S(i,i) = 0.5 * (S(i,i) - GAMMA + sqrt((S(i,i) - GAMMA)^2 + (4 * GAMMA)));
end

J = zeros([num_actors, num_actors]);
J_prime = V * S * inv(V);
for i = 1:num_actors
    for j = 1:num_actors
        if i < j
            J(i,j) = -1 * J_prime(i,j) / sqrt((p_vec(i) * p_vec(j))*(1-p_vec(i)) * (1-p_vec(j)));
        else
            J(i,j) = J(j,i);
        end
    end
end

h = heatmap(real(J));
csvwrite(file_path + "mft_roots_3.csv", real(J));




