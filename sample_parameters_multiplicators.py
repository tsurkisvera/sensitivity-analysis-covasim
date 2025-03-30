import concurrent.futures
import SALib.sample.sobol
import numpy as np
import json
import location_preprocessor as lp
import time
import pandas as pd
import os
import sys

t1 = time.time()

SYNTHPOPS_MAIN_PATH = "synthpops/synthpops/data/"
folder_name = sys.argv[1]
number_of_points = int(sys.argv[2])

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

if not os.path.exists(f'{SYNTHPOPS_MAIN_PATH}{folder_name}'):
    os.makedirs(f'{SYNTHPOPS_MAIN_PATH}{folder_name}')

age_brackets = ['0_4', '5_9', '10_14', '15_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_100']

# read different options of age distribution

print(f'begin_reading_age_structure: {time.time() - t1}')

sheets = []
dict_amount = {'1': 18, '2': 12, '3': 8, '4': 7, '5': 14, '6': 7, '7': 10, '8': 11}
for i in range(1, 9):
    for j in range(1, dict_amount[str(i)] + 1):
        df = pd.read_excel('age_structures.xlsx', sheet_name = f'2.{i}.{j}.')
        sheets.append(df)

arr_fractions = [[0 for i in range(101)] for i in range(len(sheets))]
for i in range(101):
    for j in range(len(sheets)):
        arr_fractions[j][i] = sheets[j]['Unnamed: 1'][7 + i] / sum([sheets[j]['Unnamed: 1'][i] for i in range(7, 108) if isinstance(sheets[j]['Unnamed: 1'][i], int)]) if isinstance(sheets[j]['Unnamed: 1'][7 + i], int) else 0.0

arr_bins = [[sum(arr_fractions[i][int(age_brackets[j].split('_')[0]) : int(age_brackets[j].split('_')[0]) + 1]) for j in range(16)] for i in range(len(sheets))]

print(f'end_reading_age_structure: {time.time() - t1}')

# read default parameters
def reading_default_parameters(json_file='synthpops/synthpops/data/0.json', xlsx_file='0.xlsx'):
    with open(json_file) as file:
        data = file.read()
    dict = json.loads(data)

    school_dict = lp.get_other_parameters(xlsx_file)
    contact_dict = lp.get_dict_contact_matrices(xlsx_file)

    return [dict, school_dict, contact_dict] 

# read bounds for employment rate
def reading_bounds_for_employment(txt_file):
    bounds = []
    with open(txt_file) as data:
        for line in data:
            bounds.append([float(line.strip().split()[0]), float(line.strip().split()[1])])
    return bounds

# functions to write the final dictionaries

def write_school_contacts(file, arr):
    df1 = pd.DataFrame({'School contact matrix ': []})
    df2 = pd.DataFrame({key: element for key, element in zip(age_brackets, arr)})
    writer = pd.ExcelWriter(file, engine='xlsxwriter')
    df1.to_excel(writer, sheet_name='Education_Common', index=False)
    df2.to_excel(writer, sheet_name='Education_Common', startrow=1, index=age_brackets)
    writer.close()

def write_household_contacts(file, arr):
    df1 = pd.DataFrame({'Household contact matrix ': []})
    df2 = pd.DataFrame({key: element for key, element in zip(age_brackets, arr)})
    writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay')
    df1.to_excel(writer, sheet_name='Households', index=False)
    df2.to_excel(writer, sheet_name='Households', startrow=1, index=age_brackets)
    writer.close()

def write_work_contacts(file, arr):
    df1 = pd.DataFrame({'Workplace contact matrix ': []})
    df2 = pd.DataFrame({key: element for key, element in zip(age_brackets, arr)})
    writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay')
    df1.to_excel(writer, sheet_name='Work', index=False)
    df2.to_excel(writer, sheet_name='Work', startrow=1, index=age_brackets)
    writer.close()

def write_random_contacts(file, arr):
    df1 = pd.DataFrame({'Random contact matrix ': []})
    df2 = pd.DataFrame({key: element for key, element in zip(age_brackets, arr)})
    writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay')
    df1.to_excel(writer, sheet_name='Random', index=False)
    df2.to_excel(writer, sheet_name='Random', startrow=1, index=age_brackets)
    writer.close()

def write_scalars_for_school(file, arr_names, arr):
     arr_all = []
     for i in range(len(arr_names)):
        arr_all.append(arr_names[i])
        arr_all.append(arr[i])
     df = pd.DataFrame({'': arr_all})
     writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay')
     df.to_excel(writer, sheet_name='Education_Common', startrow=20, index=False)
     writer.close()

def write_parameters(file, school_dict):
    df = pd.DataFrame({'location_name': [school_dict['location']], 'population_count': [school_dict['n']]})
    writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay')
    df.to_excel(writer, sheet_name='Parameters', index=False)
    writer.close()

def write_covasim_parameters(file, covasim_dict):
    df = pd.DataFrame({key: [covasim_dict[key]] for key in covasim_dict.keys()})
    writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay')
    df.to_excel(writer, sheet_name='Covasim_Parameters', index=False)
    writer.close()

# list the parameters in problem with respectuve ranges

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

print(f'end_writing_problem: {time.time() - t1}')

print(f'begin_salib: {time.time() - t1}')

param_values = SALib.sample.sobol.sample(problem, N=number_of_points, calc_second_order=False)
print(len(param_values))
print(f'end_salib: {time.time() - t1}')

def write(i):
    dict, school_dict, contact_dict = reading_default_parameters()

    # change household distribution

    j = problem['names'].index('coefs_for_household_size_distribution_0_0')
    dict['household_size_distribution'] = [[dict['household_size_distribution'][p][0], dict['household_size_distribution'][p][1] * param_values[i][j + p]] for p in range(7)]

    # change school distribution
        
    j = problem['names'].index('coefs_for_school_size_distribution_0_0')
    dict['school_size_distribution_by_type'][0]['size_distribution'] = np.array(dict['school_size_distribution_by_type'][0]['size_distribution']) * param_values[i][j : j + 14]
    dict['school_size_distribution_by_type'][0]['size_distribution'] = list(dict['school_size_distribution_by_type'][0]['size_distribution'] / np.sum(dict['school_size_distribution_by_type'][0]['size_distribution']))
    dict['school_size_distribution_by_type'][1]['size_distribution'] = np.array(dict['school_size_distribution_by_type'][1]['size_distribution']) * param_values[i][j + 14: j + 28]
    dict['school_size_distribution_by_type'][1]['size_distribution'] = list(dict['school_size_distribution_by_type'][1]['size_distribution'] / np.sum(dict['school_size_distribution_by_type'][1]['size_distribution']))
    dict['school_size_distribution_by_type'][2]['size_distribution'] = np.array(dict['school_size_distribution_by_type'][2]['size_distribution']) * param_values[i][j + 28: j + 42]
    dict['school_size_distribution_by_type'][2]['size_distribution'] = list(dict['school_size_distribution_by_type'][2]['size_distribution'] / np.sum(dict['school_size_distribution_by_type'][2]['size_distribution']))

    # change age distribution

    j = problem['names'].index('choice_of_age_distribution')
    for k in range(16):
        dict['population_age_distributions'][0]['distribution'][k][2] = arr_bins[int(np.round(param_values[i][j]))][k]

    # change workplace distribution
    
    j = problem['names'].index('coefs_for_workplace_distribution_0_0')
    for k in range(9):
        dict['workplace_size_counts_by_num_personnel'][k][2] = dict['workplace_size_counts_by_num_personnel'][k][2] * param_values[i][j + k]

    # change household contacts
    
    j = problem['names'].index('coefs_for_household_contacts_0_0')
    for k in range(16):
        contact_dict['H'][k] = list(np.array(contact_dict['H'][k]) * param_values[i][j + 16 * k : j + 16 * (k + 1)])

    # change school contacts
    
    j = problem['names'].index('coefs_for_school_contacts_0_0')
    for k in range(16):
        contact_dict['S'][k] = list(np.array(contact_dict['S'][k]) * param_values[i][j + 16 * k : j + 16 * (k + 1)])

    # change work contacts
    
    j = problem['names'].index('coefs_for_work_contacts_0_0')
    for k in range(16):
        contact_dict['W'][k] = list(np.array(contact_dict['W'][k]) * param_values[i][j + 16 * k : j + 16 * (k + 1)])

    # change random contacts
    
    j = problem['names'].index('coefs_for_random_contacts_0_0')
    for k in range(16):
        contact_dict['C'][k] = list(np.array(contact_dict['C'][k]) * param_values[i][j + 16 * k : j + 16 * (k + 1)])

    # change school parameters

    j = problem['names'].index('The average number of students per staff members at school')
    o_s = 1 / school_dict['average_student_all_staff_ratio'] - 1 / school_dict['average_student_teacher_ratio']
    o_s *= param_values[i][j]

    for key in ['average_student_teacher_ratio', 'average_class_size', 'inter_grade_mixing',  'average_teacher_teacher_degree', 'average_additional_staff_degree']:
        j = problem['names'].index(key)
        school_dict[key] *= param_values[i][j]

    school_dict['average_student_all_staff_ratio'] = 1 / (o_s + 1 / school_dict['average_student_teacher_ratio'])

    # change параметры covasim
    covasim_dict = {}

    j = problem['names'].index('cv_contacts')
    covasim_dict['contacts'] = [param_values[i][j]]

    # change employment rate
    j = problem['names'].index('employment_rate_0_0')
    for k in range(85):
        dict['employment_rates_by_age'][16 + k][1] = param_values[i][j + k]
        
    # write everything

    with open(f'{SYNTHPOPS_MAIN_PATH}{folder_name}/{i + 1}.json', 'w') as json_file:
        json.dump(dict, json_file)

    write_school_contacts(f'{folder_name}/{i + 1}.xlsx', contact_dict['S'])
    write_household_contacts(f'{folder_name}/{i + 1}.xlsx', contact_dict['H'])
    write_work_contacts(f'{folder_name}/{i + 1}.xlsx', contact_dict['W'])
    write_random_contacts(f'{folder_name}/{i + 1}.xlsx', contact_dict['C'])

    write_scalars_for_school(f'{folder_name}/{i + 1}.xlsx', ['The average number of students per teacher', 'Average class size', 'The average fraction of edges required to create edges between grades in the same school', 'The average number of contacts per teacher with other teachers', 'Teacher age min', 'Teacher age max', 'The average number of students per staff members at school (including both teachers and non teachers)', 'The average number of contacts per additional non teaching staff in schools',  'Staff age min', 'Staff age max'], [school_dict[key] for key in school_dict.keys() if key not in ['location', 'n']])
    write_parameters(f'{folder_name}/{i + 1}.xlsx', school_dict)
    write_covasim_parameters(f'{folder_name}/{i + 1}.xlsx', covasim_dict)

max_workers = 12

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
    for i in range(0, len(param_values), max_workers):
        res_sims = pool.map(write, range(i, min(len(param_values), i + max_workers)))

t2 = time.time()
print(f"Total time {t2 - t1}")