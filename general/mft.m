% load data
file_path = "../scotus/";
states = csvread(file_path + "processed_data.csv");
[num_actors, num_states] = size(states);

% construct p_vec and p_mat
p_vec = sum(transpose(states)) ./ num_states;
p_mat = (states * transpose(states)) ./ num_states;
p_vec_ext = repmat(p_vec, num_actors, 1);

% construct C matrix
C = zeros(num_actors, num_actors);
for i = 1:num_actors
    for j = 1:num_actors
        C(i,j) = p_mat(i,j) - (p_vec(i) * p_vec(j));
        C(i,j) = C(i,j) / sqrt((p_vec(i) * p_vec(j))*(1-p_vec(i)) * (1-p_vec(j)));
    end
end
C = inv(C);

% construct J matrix
J = zeros([num_actors, num_actors]);
for i = 1:num_actors
    for j = 1:num_actors
        if i ~= j
            J(i,j) = -1 * C(i,j) / sqrt((p_vec(i) * p_vec(j))*(1-p_vec(i)) * (1-p_vec(j)));
        end
    end
end

% declare hyperparameters
THRESHOLD = 0.001; % determines convergence
BATCH_SIZE = 1000; % number of states in each mini-batch
REG_COEFF = 0.001; % unused regularization hyperparameter
LEARNING_RATE = 0.1; % initial learning rate
MOMENTUM_COEFF = 0.9; % momentum hyperparameter

% gradient descent
mag = 1; % the norm of the gradient
count = 0; % number of epochs
grad = zeros([num_actors, num_actors]); % gradient of J
while mag > THRESHOLD && count < 1000000
    % finds minibatch
    batch = (randi(2,num_actors, BATCH_SIZE) * 2) - 3;

    % finds gradient
    Z = 0;
    boltzmann_factors = zeros(1, BATCH_SIZE);
    for index = 1:BATCH_SIZE
        state = batch(:, index);
        state_energy_matrix = state * transpose(state) .* J;
        state_energy = sum(reshape(state_energy_matrix, [num_actors * num_actors, 1]));
        score = exp(0.5 * state_energy);
        Z = Z + score;
        boltzmann_factors(1,index) = score;
    end
    
    temp_grad = zeros([num_actors, num_actors]);
    for i = 1:BATCH_SIZE
        temp_grad = temp_grad + find_grad(batch(:,i), num_actors, Z, boltzmann_factors(1,i));
        reg_term = (2 * J) .* (p_vec_ext) .* (1-p_vec_ext) .* (transpose(p_vec_ext)) .* (1-transpose(p_vec_ext));
        temp_grad = temp_grad + (REG_COEFF * reg_term);
    end
    
    temp_grad = temp_grad / BATCH_SIZE;
    update_matrix = (MOMENTUM_COEFF * grad) + ((1 - MOMENTUM_COEFF) * temp_grad);
    grad = update_matrix;
    
    % updates J matrix 
    J = J - (LEARNING_RATE * update_matrix);

    % continuation checks and decrementing 
    mag = norm(temp_grad);
    count = count + 1;
    if mod(count, 1000) == 0
        disp(mag);
        disp(count);
        LEARNING_RATE = LEARNING_RATE / 2;
    end
end

% displays and saves heatmap
h = heatmap(J);
csvwrite(file_path + "mft_roots_2.csv", J);


% finds gradient            
function grad = find_grad(vec, num_actors, Z, energy)
grad = zeros([num_actors, num_actors]);
for i = 1:num_actors
    for j = 1:num_actors
        if i < j
            grad(i,j) = (-0.5 * (Z- energy) * vec(i) * vec(j)) / Z;
        else
            grad(i,j) = grad(j,i);
        end
    end
end
end

