global states 
global p_vec

% load data
file_path = "scotus/";
states = csvread("processed_data.csv");
[num_actors, num_states] = size(states);

% construct p_vec and p_mat
temp_states = (states + 1) ./ 2;
p_vec = sum(transpose(temp_states)) ./ num_states;
p_mat = (temp_states * transpose(temp_states)) ./ num_states;

% construct C matrix
C = csvread(file_path + "correlations.csv");
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

rem_diag = ones(num_actors, num_actors) - eye(num_actors);

J = real(J) + imag(J);
MAX_STEPS = 10000;
T = 0.1;
MAG = MAX_STEPS / 10;
step = T / MAX_STEPS;
curr_prob = get_prob(J);
o_prob = curr_prob;
reg_weight = 1;
cont_pert = 0;
for i = 1:MAX_STEPS
    pert1 = rand(num_actors, num_actors);
    pert1 = (pert1 .* transpose(pert1)) - 0.25;
    pert1 = (pert1 .* rem_diag) ./ MAG;
    
    pert2 =  (rand(num_actors, num_actors) .* (J .* -1));
    pert2 = triu(pert2) - (diag(pert2) .* eye(num_actors));
    pert2 = (pert2 + transpose(pert2)) ./ MAG;
    
    perturbation = ((1 - reg_weight) * pert1) +  (reg_weight * pert2);
    
    J = J + perturbation;
    poss_prob = get_prob(J);
    
    if poss_prob < curr_prob
        thresh = exp(T * (poss_prob - curr_prob));
        if rand > thresh
            J = J - perturbation;
        else
            curr_prob = poss_prob;
        end
    else
        curr_prob = poss_prob;
    end
    T = T - step;
    
    reg_weight = 1 - (i / MAX_STEPS);
    
end
disp(o_prob);
disp(curr_prob);
disp(o_prob/curr_prob);
h = heatmap(J);
csvwrite(file_path + "roots_mft.csv", J);



function prob = get_prob(J)
    % finds likelihood of this particular configuration
    global states
    [~, num_states] = size(states);
    
    Z = 0;
    boltzmann_factors = zeros(1, num_states);
    for index = 1:num_states
        state = states(:, index);
        state_energy_matrix = state * transpose(state) .* J;
        state_energy = sum(state_energy_matrix, 'all');
        score = exp(0.5 * state_energy);
        Z = Z + score;
        boltzmann_factors(1,index) = score;
    end
    
    boltzmann_factors = log(boltzmann_factors ./ Z);
    prob = sum(boltzmann_factors, 'all') + get_reg(J);
end

function reg = get_reg(J)
    global p_vec
    
    [~, num_actors] = size(p_vec);
    
    GAMMA = -2;
    reg = 0;
    for i = 1:num_actors
        for j = 1:i
            at = (J(i,j)^2 * p_vec(i) * (1 - p_vec(i)) * p_vec(j) * (1 - p_vec(j)));
            reg = reg + at;
        end
    end
    reg = GAMMA * reg;
    
end

