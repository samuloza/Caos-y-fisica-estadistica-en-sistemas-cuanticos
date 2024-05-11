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

# IMPORT DEFINITIONS
import definitions
    
def obtain_eigenvalues(S, mu, landa, n):
    definitions.diagonalize_hamiltonian(S, mu, landa, n)
    definitions.diagonalize_hamiltonian(S, mu, landa, int(n*0.9))
    definitions.truncate_eigenvalues(S, mu, landa, int(n*0.9), n)
    definitions.rescale_eigenvalues(S, mu, landa, n)

def execute_statistics(S, n, mu_min, mu_max, step_mu, landa_min, landa_max, step_landa): 
    print("Executing...")
    for mu in np.arange(mu_min, mu_max, step_mu):
        for landa in np.arange(landa_min, landa_max, step_landa):
            definitions.histogram(S, mu, landa, n)
            
            definitions.process_statistic(S, mu, landa, n, 'r', 'REAL')
            definitions.process_statistic(S, mu, landa, n, 'gap_ratio', 'REAL')
            definitions.process_statistic(S, mu, landa, n, 's', 'RESCALED')
            
            definitions.statistic_vs_optimal_theory(S, mu, landa, n, 'r')
            definitions.statistic_vs_optimal_theory(S, mu, landa, n, 'gap_ratio')
            definitions.statistic_vs_optimal_theory(S, mu, landa, n, 's')
    
    definitions.process_optimal_metrics('r', S, n, mu_min, mu_max, step_mu, landa_min, landa_max, step_landa)
    definitions.process_optimal_metrics('gap_ratio', S, n, mu_min, mu_max, step_mu, landa_min, landa_max, step_landa)
    definitions.process_optimal_metrics('s', S, n, mu_min, mu_max, step_mu, landa_min, landa_max, step_landa)     
    
    if (mu_max - mu_min) >= step_mu:
        definitions.gap_ratio_comparison_fixed_variable(S, 'lambda', landa, mu_min, mu_max, step_mu, n)
            
    if (landa_max - landa_min) >= step_landa:
        definitions.gap_ratio_comparison_fixed_variable(S, 'mu', mu, landa_min, landa_max, step_landa, n)

def main():
    current_directory = os.getcwd()
    output_folder = os.path.join(current_directory, "Results")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    
    S = int(input("Enter the number of spins (remember it must be an integer greater than or equal to 2): "))
    n = int(input("Enter the number of photons (remember it must be an integer greater than or equal to 1): "))
    
    mu_response = ""
    while mu_response.lower() not in ["yes", "no"]:
        mu_response = input("Do you have a fixed value for the parameter 𝜇? (yes/no): ")
        if mu_response.lower() == "yes":
            mu_min = float(input("Enter the value of 𝜇 (remember it is greater than or equal to 0): "))
            mu_max = mu_min + 1
            step_mu = 2
        
        elif mu_response.lower() == "no":
            
            mu_min = float(input("Enter the minimum value of 𝜇 (remember it is greater than or equal to 0): "))
            mu_max = float(input("Enter the maximum value of 𝜇 (remember it is greater than or equal to 0): "))
            step_mu = float(input("Enter the step size of 𝜇 (remember it is greater than or equal to 0): "))
        
        else:
            print("Invalid response for 𝜇. Please enter 'yes' or 'no'.")
    
    landa_response = ""
    while landa_response.lower() not in ["yes", "no"]:
        landa_response = input("Do you have a fixed value for the parameter 𝜆? (yes/no): ")
        if landa_response.lower() == "yes":
            landa_min = float(input("Enter the value of 𝜆 (remember it is greater than or equal to 0): "))
            landa_max = landa_min + 1
            step_landa = 2
        
        elif landa_response.lower() == "no":
            landa_min = float(input("Enter the minimum value of 𝜆 (remember it is greater than or equal to 0): "))
            landa_max = float(input("Enter the maximum value of 𝜆 (remember it is greater than or equal to 0): "))
            step_landa = float(input("Enter the step size of 𝜆 (remember it is greater than or equal to 0): "))         
        
        else:
            print("Invalid response for 𝜆. Please enter 'yes' or 'no'.")

    eigenvalues_response = ""
    while eigenvalues_response.lower() not in ["yes", "no"]:
        eigenvalues_response = input("Do you already have the eigenvalues calculated? (yes/no): ")
        if eigenvalues_response.lower() == "no":
            print("Obtaining eigenvalues, this may take some time.")
            for mu in np.arange(mu_min, mu_max, step_mu):
                for landa in np.arange(landa_min, landa_max, step_landa):
                    obtain_eigenvalues(S, round(mu, 2), round(landa, 2), n)
            
        elif eigenvalues_response.lower() != "yes":
            print("Invalid response. Please enter 'yes' or 'no'.")
    
    execute_statistics(S, n, mu_min, mu_max, step_mu, landa_min, landa_max, step_landa)
    
    os.chdir(current_directory)
    print('Execution correctly finished')

if __name__ == "__main__":
    main()
