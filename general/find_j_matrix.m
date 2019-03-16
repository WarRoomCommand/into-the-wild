global corr
global data

% load data
file_path = "../scotus/";
corr = csvread(file_path + "correlations.csv");
data = csvread(file_path + "possible_states.csv");
[num_actors, ~] = size(corr);

% solve equations
roots = fsolve(@j_function, zeros(num_actors, num_actors));
h = heatmap(roots);
csvwrite(file_path + "roots.csv", roots);

% equations for fsolve
function equations = j_function(J)
    global data
    global corr

    Z = 0;
    [num_actors, num_states]  = size(data);
    boltzmann_factors = zeros(1, num_states);
    for index = 1:num_states
        state = data(:, index);
        state_energy_matrix = state * transpose(state) .* J;
        state_energy = sum(reshape(state_energy_matrix, [num_actors * num_actors, 1]));
        score = exp(0.5 * state_energy);
        Z = Z + score;
        boltzmann_factors(1,index) = score;
    end
    
    count = 1;
    equations = zeros(num_actors * num_actors, 1);
    for i = 1:num_actors
        for j = (i+1):num_actors
            total_corr = data(i,:) .* data(j,:) .* boltzmann_factors(1,:);
            total_corr = sum(total_corr) / Z;
            equations(count, 1) = total_corr - corr(i, j);
            count = count + 1;
            equations(count, 1) = J(i,j) - J(j,i);
            count = count + 1;
        end
        equations(count, 1) = J(i,i);
        count = count + 1;
    end
end
