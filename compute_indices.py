import SALib.analyze.sobol
import numpy as np
import os
import sys

def reading_bounds_for_employment(txt_file):
    bounds = []
    with open(txt_file) as data:
        for line in data:
            bounds.append([float(line.strip().split()[0]), float(line.strip().split()[1])])
    return bounds

def bounds_for_enrolment(file):
    arr = np.loadtxt(file).tolist()
    return(arr)

def calculate_indices(problem, keys, key, Y, averaging=1):
    index = keys.index(key)  
    dict_indices = SALib.analyze.sobol.analyze(problem, np.array(Y[index]), keep_resamples=True, num_resamples=1000, calc_second_order=False)
    return dict_indices

def write_indices(experiment, dict_indices, key, identificator):
    np.savetxt(f'{experiment}_indices/{key}_{identificator}', dict_indices[identificator])

def read_points(experiment, number_of_points, number_of_groups, averaging=1):
    Y = np.array([[0. for i in range(number_of_points * (number_of_groups + 2))] for k in range(6)])
    arr_names = os.listdir(experiment)
    for i in range(0, averaging * number_of_points * (number_of_groups + 2), averaging):
        arr_peak, arr_dead, arr_cases, arr_severe, arr_critical, arr_mortality = [], [], [], [], [], []
        for j in range(0, averaging):
            if str(i + j) in arr_names:
                with open(f'{experiment}/{i + j}') as data:
                    curve, peak, dead, severe, critical = data.read().strip().split('\n')
                arr_peak.append(float(peak))
                arr_dead.append(float(dead))
                arr_cases.append(sum(list(map(float, curve.strip().split()))))
                arr_severe.append(float(severe))
                arr_critical.append(float(critical))
                arr_mortality.append(float(dead) / sum(list(map(float, curve.strip().split()))))
            else:
                arr_peak.append(np.nan)
                arr_dead.append(np.nan)
                arr_cases.append(np.nan)
                arr_severe.append(np.nan)
                arr_critical.append(np.nan)
                arr_mortality.append(np.nan)
        if np.nan in arr_peak:
            Y[0][i // averaging] =  np.nan # height of the peak
            Y[1][i // averaging] =  np.nan # number of deaths
            Y[2][i // averaging] =  np.nan # total number of cases
            Y[3][i // averaging] =  np.nan # peak number of severe cases
            Y[4][i // averaging] =  np.nan # peak number of critical cases
            Y[5][i // averaging] =  np.nan # mortality
        else:
            Y[0][i // averaging] =  np.mean(arr_peak) # height of the peak
            Y[1][i // averaging] =  np.mean(arr_dead) # number of deaths
            Y[2][i // averaging] =  np.mean(arr_cases) # total number of cases
            Y[3][i // averaging] =  np.mean(arr_severe) # peak number of severe cases
            Y[4][i // averaging] =  np.mean(arr_critical) # peak number of critical cases
            Y[5][i // averaging] =  np.mean(arr_mortality) # mortality
    return Y

def execute_analysis(experiment, problem, number_of_points, averaging=1):
    if not os.path.exists(f'{experiment}_indices'):
        os.makedirs(f'{experiment}_indices')
    number_of_groups = len(set(problem['groups']))
    Y = read_points(experiment, number_of_points, number_of_groups, averaging=averaging)
    keys = ['peak', 'deaths', 'cases', 'severe', 'critical', 'mortality']
    identificators = ['S1', 'S1_conf', 'ST', 'ST_conf', 'S1_conf_all', 'ST_conf_all']
    for key in keys:
        dict_indices = calculate_indices(problem, keys, key, Y)
        for identificator in identificators:
            write_indices(experiment, dict_indices, key, identificator)

# problem should match the one which was defined in the sampling file (sample_parameters_autoencoders.py or sample_parameters_multipliers.py)    
problem = {'names': ['choice_of_age_distribution', 'average_student_teacher_ratio', 'average_class_size', 'inter_grade_mixing',  'average_teacher_teacher_degree', 'The average number of students per staff members at school', 'average_additional_staff_degree', 'cv_contacts'],
           'groups': ['choice_of_age_distribution', 'average_student_teacher_ratio', 'average_class_size', 'inter_grade_mixing',  'average_teacher_teacher_degree', 'The average number of students per staff members at school', 'average_additional_staff_degree', 'cv_contacts'],
           'bounds': [[0, 86]] + [[0.5, 2] for i in range(7)],
           'num_vars': 8
}

dict_for_matrices = {'coefs_for_household_size_distribution': [1, 7, [[0.5, 2]]], 'coefs_for_school_size_distribution': [3, 14, [[0.5, 2]]], 'coefs_for_workplace_distribution': [1, 9, [[0.5, 2]]], 'coefs_for_household_contacts': [16, 16, [[0.5, 2]]], 'coefs_for_school_contacts': [16, 16, [[0.5, 2]]], 'coefs_for_work_contacts': [16, 16, [[0.5, 2]]], 'coefs_for_random_contacts': [16, 16, [[0.5, 2]]],  'employment_rate': [1, 85, reading_bounds_for_employment('employment_bounds.txt')]}

for key, value in dict_for_matrices.items():
    for i in range(value[0]):
        for j in range(value[1]):
            if len(value[2]) == 1:
                problem['names'].append(key + '_' + str(i) + '_' + str(j))
                problem['bounds'].append(value[2][0])
                problem['groups'].append(key)
                problem['num_vars'] += 1
            else:
                problem['names'].append(key + '_' + str(i) + '_' + str(j))
                problem['bounds'].append(value[2][j])
                problem['groups'].append(key)
                problem['num_vars'] += 1

# comment this for loop if you want to perform the analysis on the full set of parameters

for i in range(len(problem['groups'])):
    if problem['groups'][i] not in ['choice_of_age_distribution', 'average_student_teacher_ratio', 'average_class_size', 'coefs_for_household_size_distribution', 'coefs_for_workplace_distribution', 'coefs_for_household_contacts', 'employment_rate']:
        problem['groups'][i] = 'the_rest'

execute_analysis(f'{sys.argv[1]}_results', problem, int(sys.argv[2])) 