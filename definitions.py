# LIBRARIES
import os
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# GRAPH SETTINGS
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
    
# LOADING DATA FROM CSV
def load_data(file_path, column):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data[column].values
    else:
        raise FileNotFoundError("The specified file does not exist.")

# SAVING DATA TO CSV
def save_csv(data, columns, destination_folder, csv_file):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    df = pd.DataFrame(data, columns=columns)
    save_path = os.path.join(destination_folder, csv_file)
    df.to_csv(save_path, index=False) 

# GENERATING THE HAMILTONIAN MATRIX ASSOCIATED WITH CERTAIN PARAMETERS
def create_matrix(S, A, B, C, n):
    N = 2 * S
    dim = (N + 1) * (n + 1)
    matrix = np.zeros((dim, dim))

    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(n + 1):
                for l in range(n + 1):
                    m_j = j - S
                    if i == j and k == l:
                        matrix[i * (n + 1) + k][j * (n + 1) + l] = np.sqrt(l) * np.sqrt(l + 1) + m_j
                    elif i == j and k == l - 1:
                        matrix[i * (n + 1) + k][j * (n + 1) + l] = -C * np.sqrt(l)
                    elif i == j and k == l + 1:
                        matrix[i * (n + 1) + k][j * (n + 1) + l] = -C * np.sqrt(l + 1)
                    elif i == j + 1 and k == l - 1:
                        matrix[i * (n + 1) + k][j * (n + 1) + l] = A * np.sqrt(l) * np.sqrt(B - m_j * (m_j + 1))
                    elif i == j-1 and k == l+1:
                        matrix[i * (n + 1) + k][j * (n + 1) + l] = A * np.sqrt(l + 1) * np.sqrt(B - m_j * (m_j - 1))

    return matrix
    
# OBTAINING THE EIGENVALUES OF THE HAMILTONIAN BY DIAGONALIZATION
def diagonalize_hamiltonian(S, mu, landa, n):
    A = landa * S**(-1/2)
    B = S * (S+1)
    C = mu * np.sqrt(S)
    H = create_matrix(S, A, B, C, n)
    
    eigenvalues = eigvalsh(H)
    eigenvalues_mod = eigenvalues - min(eigenvalues)
    
    save_csv(eigenvalues_mod.reshape(-1, 1), ['Eigenvalue'], f'EIGENVALUES_S-{S}_n-{n}', 
                f'raw_eigenvalues_S-{S}_mu-{round(mu, 2)}_lambda-{round(landa, 2)}_n-{n}.csv')
    
# TRUNCATING EIGENVALUES BASED ON THE NUMBER OF INCIDENT PHOTONS
def truncate_eigenvalues(S, mu, landa, n_lower, n_higher):
    file_path_1 = f'EIGENVALUES_S-{S}_n-{n_lower}/raw_eigenvalues_S-{S}_mu-{round(mu, 2)}_lambda-{round(landa, 2)}_n-{n_lower}.csv'
    file_path_2 = f'EIGENVALUES_S-{S}_n-{n_higher}/raw_eigenvalues_S-{S}_mu-{round(mu, 2)}_lambda-{round(landa, 2)}_n-{n_higher}.csv'
    eigenvalues_1 = load_data(file_path_1, 'Eigenvalue')
    eigenvalues_2 = load_data(file_path_2, 'Eigenvalue')
    
    eigenvalues = []
    for val1, val2 in zip(eigenvalues_1, eigenvalues_2):
        if abs(val1 - val2) <= 0.01 * val2:
            eigenvalues.append(val2)
        else:
            break
    
    save_csv(np.array(eigenvalues).reshape(-1, 1), ['Eigenvalue'], 
                f'REAL_EIGENVALUES_S-{S}_n-{n_higher}', 
                f'eigenvalues_S-{S}_mu-{round(mu, 2)}_lambda-{round(landa, 2)}_n-{n_higher}.csv')

# RESCALING THE LEVELS FOR s STATISTIC CALCULATION
def rescale_eigenvalues(S, mu, landa, n):
    file_path = f'REAL_EIGENVALUES_S-{S}_n-{n}/eigenvalues_S-{S}_mu-{round(mu, 2)}_lambda-{round(landa, 2)}_n-{n}.csv'
    eigenvalues = load_data(file_path, 'Eigenvalue')
    
    cumulative_density = np.arange(1, len(eigenvalues) + 1)
    coefficients = np.polyfit(eigenvalues, cumulative_density, 8)
    polynomial = np.poly1d(coefficients)
    eigenvalues_rescaled = polynomial(eigenvalues) - min(polynomial(eigenvalues))
    
    save_csv(eigenvalues_rescaled.reshape(-1, 1), ['Eigenvalue'], 
                f'RESCALED_EIGENVALUES_S-{S}_n-{n}', 
                f'eigenvalues_S-{S}_mu-{round(mu, 2)}_lambda-{round(landa, 2)}_n-{n}.csv')

# GENERATING THE LEVELS HISTOGRAM WITH THE OBTAINED EIGENVALUES
def histogram(S, mu, landa, n):
    file_path = f'RESCALED_EIGENVALUES_S-{S}_n-{n}/eigenvalues_S-{S}_mu-{round(mu,2)}_lambda-{round(landa,2)}_n-{n}.csv'
    data = pd.read_csv(file_path)
    eigenvalues = data["Eigenvalue"].values

    plt.figure()
    plt.hist(eigenvalues, bins=100, density=False, color='black', edgecolor='black')
    plt.xlabel('Energy level')
    plt.ylabel('Frequency')

    destination_folder = f'HISTOGRAM_LEVELS_S-{S}_n-{n}'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    filename = f'histogram_S-{S}_mu-{round(mu,2)}_lambda-{round(landa,2)}_n-{n}.png'
    save_path = os.path.join(destination_folder, filename)
    plt.savefig(save_path)
    plt.close()

# OBTAINING THE STATISTIC ASSOCIATED WITH A HAMILTONIAN
def process_statistic(S, mu, landa, n, statistic_name, statistic_type):
    file_path = f'{statistic_type}_EIGENVALUES_S-{S}_n-{n}/eigenvalues_S-{S}_mu-{round(mu,2)}_lambda-{round(landa,2)}_n-{n}.csv'
    if os.path.exists(file_path):
        eigenvalues = load_data(file_path, 'Eigenvalue')
        statistics = []
        if statistic_name.lower() == 's':
            for k in range(len(eigenvalues) - 1):
                statistic = eigenvalues[k+1] - eigenvalues[k]
                statistics.append(statistic)
        elif statistic_name.lower() == 'r':
            for k in range(len(eigenvalues) - 2):
                statistic = (eigenvalues[k+2]-eigenvalues[k+1]) / (eigenvalues[k+1]-eigenvalues[k])
                statistics.append(statistic)
        elif statistic_name.lower() == 'gap_ratio':
            for k in range(len(eigenvalues) - 2):
                gap_num = min(eigenvalues[k+2]-eigenvalues[k+1], eigenvalues[k+1]-eigenvalues[k])
                gap_den = max(eigenvalues[k+2]-eigenvalues[k+1], eigenvalues[k+1]-eigenvalues[k])
                statistic = gap_num / gap_den
                statistics.append(statistic)
    
        save_csv(statistics, [f'Statistic {statistic_name}'], f'STATISTIC_{statistic_name}_S-{S}_n-{n}', 
                    f'statistic_{statistic_name}_S-{S}_mu-{round(mu,2)}_lambda-{round(landa,2   )}_n-{n}.csv')
    else:
        raise FileNotFoundError("The specified file does not exist.")

# OBJECTIVE FOR MINIMIZATION
def objective(params, x, y1, y2, hist):
    alpha, beta = params
    y_fit = alpha * y1 + beta * y2
    return np.sum((hist - y_fit) ** 2)

# FITTING LINEAR COMBINATION FOR LEVELS IN TRANSITION
def fit_histogram(x_values, y_values, y_values_2, hist):
    params_init = [0.5, 0.5]
    result = minimize(objective, params_init, args=(x_values, y_values, y_values_2, hist))
    alpha_opt, beta_opt = result.x
    return alpha_opt, beta_opt

# REPRESENTATION OF THE FITTED STATISTIC AND THEORETICAL DISTRIBUTIONS
def statistic_vs_optimal_theory(S, mu, landa, n, statistic_name):
    file_path = f'STATISTIC_{statistic_name}_S-{S}_n-{n}/statistic_{statistic_name}_S-{S}_mu-{round(mu,2)}_lambda-{round(landa,2)}_n-{n}.csv'
    data = pd.read_csv(file_path)
    statistic_data = data[f'Statistic {statistic_name}'].values
    
    plt.figure()
    if statistic_name == 's':
        max_value = 4
        bins_considered = 100
        x_values = np.linspace(0.01, max_value, bins_considered)
        y_values = np.exp(-x_values)
        y_values_2 = (3.1415/2) * x_values * np.exp(-3.1415*(x_values**2)/4)
        colors=['lightblue', 'blue', 'green']
        labels=[r'Poisson: $e^{-s}$', r'GOE: $\frac{\pi}{2}\cdot s\cdot e^{-\frac{\pi}{4}\cdot s^2}$']
    elif statistic_name == 'r':
        max_value = 4
        bins_considered = 100
        x_values = np.linspace(0.01, max_value, bins_considered)
        y_values = 1 / (x_values + 1) ** 2
        y_values_2 = (27 * (x_values+x_values**2)**1) / (8*(1+x_values+x_values**2)**(5/2))
        colors=['lightcoral', 'red', 'royalblue']
        labels=[r'Poisson: $\frac{1}{{(1+r)}^2}$', r'GOE: $\frac{27}{8}\cdot \frac{r+r^2}{(1+r+r^2)^{5/2}}$']
    elif statistic_name == 'gap_ratio':
        max_value = 1
        bins_considered = 25
        x_values = np.linspace(0.01, max_value, bins_considered)
        y_values = 2 / (x_values + 1) ** 2
        y_values_2 = 2*(27 * (x_values+x_values**2)**1) / (8*(1+x_values+x_values**2)**(5/2))
        colors=['lightgreen', 'green', 'red']
        labels=[r'Poisson: $\frac{2}{{(1+r)}^2}$', r'GOE: $\frac{27}{4}\cdot \frac{r+r^2}{(1+r+r^2)^{5/2}}$']

    plt.xlabel('Statistic {statistic_name}')
    plt.ylabel('Probability density')
    
    plt.hist(statistic_data, bins=bins_considered, range=(0, max_value), color=colors[0], edgecolor=colors[1], density=True, alpha=0.7, label=r'Experimental data')
    plt.plot(x_values, y_values, color='black', linestyle='--', linewidth=2, label=labels[0])
    plt.plot(x_values, y_values_2, color=colors[2], linestyle='--', linewidth=2, label=labels[1])

    hist_statistic, _ = np.histogram(statistic_data, bins=bins_considered, range=(0, max_value), density=True)
    alpha_opt, beta_opt = fit_histogram(x_values, y_values, y_values_2, hist_statistic)

    y_fit = alpha_opt * y_values + beta_opt * y_values_2
    plt.plot(x_values, y_fit, color='darkmagenta', linestyle='--', linewidth=2, label=str(round(alpha_opt,3))+'·Poisson + '+str(round(beta_opt,3))+'·GOE')
    plt.legend()

    destination_folder = f'HISTOGRAM_STATISTIC_{statistic_name}_S-{S}_n-{n}'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    name = '{statistic_name}_vs_theory_S-'+str(S)+'_mu-'+str(mu)+'_lambda-'+str(landa)+'_n-'+str(n)+'.png'
    save_path = os.path.join(destination_folder, name)
    plt.savefig(save_path)
    plt.close()   

# REPRESENTATION OF THE EVOLUTION OF GAP RATIO FOR A FIXED VALUE OF MU OR LAMBDA
def gap_ratio_comparison_fixed_variable(S, fixed_variable, fixed_value, other_value_min, other_value_max, step, n):
    gaps = []
    for value in np.arange(other_value_min, other_value_max, step):
        if fixed_variable == 'mu':
            file_path = f'STATISTIC_gap_ratio_S-{S}_n-{n}/statistic_gap_ratio_S-{S}_mu-{round(fixed_value,2)}_lambda-{round(value,2)}_n-{n}.csv'
        elif fixed_variable == 'lambda':
            file_path = f'STATISTIC_gap_ratio_S-{S}_n-{n}/statistic_gap_ratio_S-{S}_mu-{round(value,2)}_lambda-{round(fixed_value,2)}_n-{n}.csv'
        data = pd.read_csv(file_path)
        gap = data["Statistic gap_ratio"].values
        mean_gap = np.mean(gap)
        gaps.append(mean_gap)

    plt.scatter(np.arange(other_value_min, other_value_max, step), gaps, color='black', label=r'Experimental data')

    if fixed_variable == 'mu':
        plt.title(f'$\lambda$ comparison, $\mu$={round(fixed_value,2)} fixed')
        plt.xlabel('$\lambda$')
        name = f'gap_vs_theory_S-{S}_lambda_fixed-{round(fixed_value,2)}_mu_from-{round(other_value_min,2)}-to-{round(other_value_max,2)}_n-{n}.png'
    elif fixed_variable == 'lambda':
        plt.title(f'$\mu$ comparison, $\lambda$={round(fixed_value,2)} fixed')
        plt.xlabel('$\mu$')
        name = f'gap_vs_theory_S-{S}_mu_fixed-{round(fixed_value,2)}_lambda_from-{round(other_value_min,2)}-to-{round(other_value_max,2)}_n-{n}.png'
    plt.ylabel('Gap')

    x_values = np.linspace(other_value_min, other_value_max, 400)
    y_values = 0.39 * np.ones(400)  # Poisson values
    y_values_2 = 0.53 * np.ones(400)  # GOE values
    plt.plot(x_values, y_values, color='red', linestyle='--', label='Poisson')
    plt.plot(x_values, y_values_2, color='green', linestyle='--', label='GOE')
    plt.legend()

    destination_folder = f'HISTOGRAM_GAP_RATIO_S-{S}_n-{n}'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)    
    save_path = os.path.join(destination_folder, name)
    plt.savefig(save_path)
    plt.close()

# EVOLUTION OF THE OPTIMAL METRIC FOR AN STATISTIC  
def process_optimal_metrics(statistic_name, S, n, mu_min, mu_max, step_mu, landa_min, landa_max, step_landa):
    df = pd.DataFrame(columns=['mu', 'lambda', 'alpha_opt', 'beta_opt'])

    for mu in np.arange(mu_min, mu_max, step_mu):
        for landa in np.arange(landa_min, landa_max, step_landa):
            try:
                processed_file_path = f'STATISTIC_{statistic_name}_S-{S}_n-{n}/statistic_{statistic_name}_S-{S}_mu-{round(mu,2)}_lambda-{round(landa,2)}_n-{n}.csv'
                processed_data = pd.read_csv(processed_file_path)
                statistic = processed_data[f'Statistic {statistic_name}'].values

                if statistic_name == "s":
                    max_value = 4
                    bins_considered = 100
                    x_values = np.linspace(0.01, max_value, bins_considered)
                    y_values = np.exp(-x_values)
                    y_values_2 = (3.1415/2) * x_values * np.exp(-3.1415*(x_values**2)/4)
                elif statistic_name == "r":
                    max_value = 4
                    bins_considered = 100
                    x_values = np.linspace(0.01, max_value, bins_considered)
                    y_values = 1 / (x_values + 1) ** 2
                    y_values_2 = (27 * (x_values+x_values**2)**1)/(8*(1+x_values+x_values**2)**(5/2))
                elif statistic_name == "gap_ratio":
                    max_value = 1
                    bins_considered = 25
                    x_values = np.linspace(0.01, max_value, bins_considered)
                    y_values = 2 / (x_values + 1) ** 2
                    y_values_2 = 2*(27 * (x_values+x_values**2)**1)/(8*(1+x_values+x_values**2)**(5/2))

                hist_data, _ = np.histogram(statistic, bins=bins_considered, range=(0, max_value), density=True)
                alpha_opt, beta_opt = fit_histogram(x_values, y_values, y_values_2, hist_data)

                df.loc[len(df)] = [round(mu, 2), round(landa, 2), alpha_opt, beta_opt]

            except FileNotFoundError as e:
                print(f"Error FileNotFoundError: {e}. Ignoring and moving to the next iteration.")
                continue

    archivo_csv = f'{statistic_name}_S-{S}_n-{n}.csv'
    df.to_csv(archivo_csv, index=False)
