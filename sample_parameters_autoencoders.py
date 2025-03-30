import concurrent.futures
import SALib.sample.sobol
import numpy as np
import json
import location_preprocessor as lp
import time
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import os
import numpy as np
import sys

t1 = time.time()

SYNTHPOP_MAIN_PATH = "synthpops/synthpops/data/"
folder_name = sys.argv[1]
number_of_points = int(sys.argv[2])

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

if not os.path.exists(f'{SYNTHPOPS_MAIN_PATH}{folder_name}'):
    os.makedirs(f'{SYNTHPOPS_MAIN_PATH}{folder_name}')

age_brackets = ['0_4', '5_9', '10_14', '15_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_100']

# read default parameters

def reading_default_parameters(json_file='synthpops/synthpops/data/0.json', xlsx_file='0.xlsx'):
    with open(json_file) as file:
        data = file.read()
    dict = json.loads(data)

    school_dict = lp.get_other_parameters(xlsx_file)
    contact_dict = lp.get_dict_contact_matrices(xlsx_file)

    return [dict, school_dict, contact_dict] 

# transform employment distribution 

def proper_employment(arr):
    final_arr = [i for i in range(101)]
    for i in range(15):
        final_arr[i] = 0
    for i in range(15, 65):
        final_arr[i] = arr[i // 5 - 3]
    for i in range(65, 101):
        final_arr[i] = arr[-1]
    return final_arr

# read boulds for enrolment

def bounds_for_enrolment(file):
    arr = np.loadtxt(file).tolist()
    return(arr)

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

# autoencoder structure for contact matrices

class Autoencoder1(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(16 * 16, 6 * 6),
            nn.ReLU(),
            nn.Linear(6 * 6, 1 * 1),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1 * 1, 6 * 6),
            nn.ReLU(),
            nn.Linear(6 * 6, 16 * 16),
            nn.ReLU()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return self.decoder(y)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# autoencoder structure for household size distribution

class Autoencoder_hh(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(1 * 7, 1 * 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1 * 1, 1 * 7),
            nn.ReLU()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return nn.functional.normalize(input=self.decoder(y)[0], p=1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = nn.functional.normalize(input=self.decoder(encoded)[0], p=1)
        return decoded

# autoencoder structure for age distribution

class Autoencoder_age(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(1 * 16, 1 * 6),
            nn.ReLU(),
            nn.Linear(1 * 6, 1 * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1 * 2, 1 * 6),
            nn.ReLU(),
            nn.Linear(1 * 6, 1 * 16),
            nn.ReLU()
        )
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return nn.functional.normalize(input=self.decoder(y)[0], p=1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = nn.functional.normalize(input=self.decoder(encoded)[0], p=1)
        return decoded

# autoencoder structire for hha_bysize matrix

class Autoencoder_hhabysize(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(7 * 11, 5 * 8),
            nn.ReLU(),
            nn.Linear(5 * 8, 1 * 1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1 * 1, 5 * 8),
            nn.ReLU(),
            nn.Linear(5 * 8, 7 * 11),
            nn.ReLU()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return self.decoder(y)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# autoencoder structure for employment rate      
   
class Autoencoder_employment(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(1 * 11, 1 * 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1 * 1, 1 * 11),
            nn.ReLU()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return self.decoder(y)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

home_model = Autoencoder1()
home_model.load_state_dict(torch.load('models/model_for_home_1D_28_08.txt',  weights_only=True))
home_model.double()

work_model = Autoencoder1()
work_model.load_state_dict(torch.load('models/model_for_work_1D_28_08.txt',  weights_only=True))
work_model.double()

school_model = Autoencoder1()
school_model.load_state_dict(torch.load('models/model_for_school_1D_28_08.txt',  weights_only=True))
school_model.double()

random_model = Autoencoder1()
random_model.load_state_dict(torch.load('models/model_for_random_1D_23_09.txt',  weights_only=True))
random_model.double()

hh_model = Autoencoder_hh()
hh_model.load_state_dict(torch.load('models/model_for_hh_1D_L2_23_09.txt', weights_only=True))
hh_model.double()

age_model = Autoencoder_age()
age_model.load_state_dict(torch.load('models/model_for_age_L1_2D_addit_layer_29_08.txt', weights_only=True))
age_model.double()

hhabysize_model = Autoencoder_hhabysize()
hhabysize_model.load_state_dict(torch.load('models/model_for_hhabysize_1D_L2_23_09.txt', weights_only=True))
hhabysize_model.double()

employment_model = Autoencoder_employment()
employment_model.load_state_dict(torch.load('models/model_for_employment_18_09.txt', weights_only=True))
employment_model.double()

transform = transforms.ToTensor()

# list the parameters in problem with respectuve ranges

problem = {'names': ['home_contacts', 'work_contacts', 'school_contacts', 'random_contacts', 'hh_sizes', 'hha_bysize', 'average_student_teacher_ratio', 'average_class_size',  'average_teacher_teacher_degree', 'The average number of students per staff members at school', 'average_additional_staff_degree', 'employment_rate', 'cv_contacts'],
           'groups': ['home_contacts', 'work_contacts', 'school_contacts', 'random_contacts', 'hh_sizes', 'hha_bysize', 'average_student_teacher_ratio', 'average_class_size',  'average_teacher_teacher_degree', 'The average number of students per staff members at school', 'average_additional_staff_degree', 'employment_rate', 'cv_contacts'],
           'bounds': [[-0.25, 0.54], [-0.73, 0.05], [-0.56, 0.78], [-0.76, 0.88], [-0.71, -0.02], [-0.24, 1.47], [6.3044, 27.199], [14, 30], [0.5, 2], [1, 2], [0.5, 2], [-66.04, -46.83], [4, 20]],
           'num_vars': 13
}

dict_for_matrices = {'coefs_for_age_distribution': [1, 2, [[-1.31, 0.24], [-0.25, 0.21]]], 'coefs_for_school_size_distribution': [3, 4, [[0.5, 2]]], 'workplace_distribution': [1, 5, [[28333, 3449178], [833, 214917], [523, 111188], [250, 50930], [54, 10870]]], 'enrolment': [1, 65, bounds_for_enrolment('bounds_for_enrolment.txt')]}

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
    if problem['groups'][i] not in ['home_contacts', 'work_contacts', 'coefs_for_age_distribution', 'hh_sizes', 'hha_bysize', 'average_student_teacher_ratio', 'average_class_size', 'workplace_distribution']:
        problem['groups'][i] = 'the_rest' 

# sampling from provided ranges using Sobol'sequences 

param_values = SALib.sample.sobol.sample(problem, N=number_of_points, calc_second_order=False)
print('number of runs:', len(param_values))

def write(i):
    dict, school_dict, contact_dict = reading_default_parameters()

    # sample age distribution, hh sizes distribution, head ages by sizes

    numbers_for_ages = [param_values[i][problem['names'].index('coefs_for_age_distribution_0_0')], param_values[i][problem['names'].index('coefs_for_age_distribution_0_0') + 1]]
    number_for_hhabysizes = param_values[i][problem['names'].index('hha_bysize')]
    number_for_hhsizes = param_values[i][problem['names'].index('hh_sizes')]

    def parabole(x, a, b, c):
        return a * x ** 2 + b * x + c
        
    params_for_parabole = np.array([-0.49611302, -0.48741903, -0.06646047])
    
    agedistr = age_model.decode(transform(np.array([[numbers_for_ages[0], numbers_for_ages[1] + parabole(numbers_for_ages[0], *params_for_parabole)]]))).detach().numpy()[0]
    hhabysizematrix = hhabysize_model.decode(transform(np.array([[number_for_hhabysizes]]))).detach().numpy()[0][0].reshape(7, 11)
    hhdistr = hh_model.decode(transform(np.array([[number_for_hhsizes]]))).detach().numpy()[0]
                   
    for k in range(16):
        dict['population_age_distributions'][0]['distribution'][k][2] = agedistr[k]
                   
    dict['household_size_distribution'] = [[dict['household_size_distribution'][p][0], hhdistr[p]] for p in range(7)]
    
    final_distr = np.zeros((7, 12))
    for k in range(1, 8):
        final_distr[k - 1, 0] = k
    for j in range(7):
        for k in range(1, 12):
            final_distr[j, k] = hhabysizematrix[j, k - 1]
    dict['household_head_age_distribution_by_family_size'] = final_distr.tolist()

    # sample contact matrices

    number_for_hhcontacts = param_values[i][problem['names'].index('home_contacts')]
    hhcontactmatrix = home_model.decode(transform(np.array([[number_for_hhcontacts]]))).detach().numpy()[0].reshape(16, 16)
    contact_dict['H'] = hhcontactmatrix

    number_for_wcontacts = param_values[i][problem['names'].index('work_contacts')]
    wcontactmatrix = work_model.decode(transform(np.array([[number_for_wcontacts]]))).detach().numpy()[0].reshape(16, 16)
    contact_dict['W'] = wcontactmatrix

    number_for_schcontacts = param_values[i][problem['names'].index('school_contacts')]
    schcontactmatrix = school_model.decode(transform(np.array([[number_for_schcontacts]]))).detach().numpy()[0].reshape(16, 16)
    contact_dict['S'] = schcontactmatrix

    number_for_rcontacts = param_values[i][problem['names'].index('random_contacts')]
    rcontactmatrix = random_model.decode(transform(np.array([[number_for_rcontacts]]))).detach().numpy()[0].reshape(16, 16)
    contact_dict['C'] = rcontactmatrix

    # sample employment rate

    number_for_employment = param_values[i][problem['names'].index('employment_rate')]
    employment_distr = employment_model.decode(transform(np.array([[number_for_employment]]))).detach().numpy()[0][0]
    final_employment = proper_employment(employment_distr)
    dict['employment_rates_by_age'] = [[k, final_employment[k]] for k in range(101)]

    # change school size distributions

    j = problem['names'].index('coefs_for_school_size_distribution_0_0')
    dict['school_size_distribution_by_type'][0]['size_distribution'] = np.array(dict['school_size_distribution_by_type'][0]['size_distribution']) * np.array([param_values[i][j] for k in range(4)] + [param_values[i][j + 1] for k in range(4)] + [param_values[i][j + 2] for k in range(4)] + [param_values[i][j + 3] for k in range(2)])
    dict['school_size_distribution_by_type'][0]['size_distribution'] = list(dict['school_size_distribution_by_type'][0]['size_distribution'] / np.sum(dict['school_size_distribution_by_type'][0]['size_distribution']))
    dict['school_size_distribution_by_type'][1]['size_distribution'] = np.array(dict['school_size_distribution_by_type'][1]['size_distribution']) * np.array([param_values[i][j + 4] for k in range(4)] + [param_values[i][j + 5] for k in range(4)] + [param_values[i][j + 6] for k in range(4)] + [param_values[i][j + 7] for k in range(2)])
    dict['school_size_distribution_by_type'][1]['size_distribution'] = list(dict['school_size_distribution_by_type'][1]['size_distribution'] / np.sum(dict['school_size_distribution_by_type'][1]['size_distribution']))
    dict['school_size_distribution_by_type'][2]['size_distribution'] = np.array(dict['school_size_distribution_by_type'][2]['size_distribution']) * np.array([param_values[i][j + 8] for k in range(4)] + [param_values[i][j + 9] for k in range(4)] + [param_values[i][j + 10] for k in range(4)] + [param_values[i][j + 11] for k in range(2)])
    dict['school_size_distribution_by_type'][2]['size_distribution'] = list(dict['school_size_distribution_by_type'][2]['size_distribution'] / np.sum(dict['school_size_distribution_by_type'][2]['size_distribution']))
    
    # sample school scalar parameters

    j = problem['names'].index('average_class_size')
    school_dict['average_class_size'] = int(np.round(param_values[i][j]))

    j = problem['names'].index('average_student_teacher_ratio')
    school_dict['average_student_teacher_ratio'] = param_values[i][j]

    j = problem['names'].index('The average number of students per staff members at school')
    school_dict['average_student_all_staff_ratio'] = school_dict['average_student_teacher_ratio'] / param_values[i][j]

    for key in ['average_teacher_teacher_degree', 'average_additional_staff_degree']:
        j = problem['names'].index(key)
        school_dict[key] *= param_values[i][j]

    # sample workplace size distribution

    dict['workplace_size_counts_by_num_personnel'] = [[0, 9, 0], [10, 19, 0], [20, 49, 0], [50, 249, 0], [250, 2000, 0]]
    j = problem['names'].index('workplace_distribution_0_0')
    for k in range(5):
        dict['workplace_size_counts_by_num_personnel'][k][2] = param_values[i][j + k]

    # sample enrolment rate by age

    j = problem['names'].index('enrolment_0_0')
    dict['enrollment_rates_by_age'] = [[k, param_values[i][j + k] / 100] for k in range(65)] + [[65 + k, 0] for k in range(36)]

    # sample covasim parameters

    covasim_dict = {}

    j = problem['names'].index('cv_contacts')
    covasim_dict['contacts'] = [param_values[i][j]]
        
    # write everything

    with open(f'synthpops/synthpops/data/{folder_name}/{i + 1}.json', 'w') as json_file:
        json.dump(dict, json_file)

    write_school_contacts(f'{folder_name}/{i + 1}.xlsx', contact_dict['S'])
    write_household_contacts(f'{folder_name}/{i + 1}.xlsx', contact_dict['H'])
    write_work_contacts(f'{folder_name}/{i + 1}.xlsx', contact_dict['W'])
    write_random_contacts(f'{folder_name}/{i + 1}.xlsx', contact_dict['C'])

    write_scalars_for_school(f'{folder_name}/{i + 1}.xlsx', ['The average number of students per teacher', 'Average class size', 'The average fraction of edges required to create edges between grades in the same school', 'The average number of contacts per teacher with other teachers', 'Teacher age min', 'Teacher age max', 'The average number of students per staff members at school (including both teachers and non teachers)', 'The average number of contacts per additional non teaching staff in schools',  'Staff age min', 'Staff age max'], [school_dict[key] for key in school_dict.keys() if key not in ['location', 'n']])
    write_parameters(f'{folder_name}/{i + 1}.xlsx', school_dict)
    write_covasim_parameters(f'{folder_name}/{i + 1}.xlsx', covasim_dict)

max_workers=150

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
    for i in range(0, len(param_values), max_workers):
        res_sims = pool.map(write, range(i, min(len(param_values), i + max_workers)))

t2 = time.time()
print(f"Total time {t2 - t1}")