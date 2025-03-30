import sys
import pandas as pd
import json
import synthpops.synthpops as sp
import sciris as sc
import numpy as np


SYNTHPOP_MAIN_PATH = "synthpops/synthpops/data/"

def get_index_by_key(df, key):
    index = df[df.iloc[:, 0] == key].index[0]
    return index


def treat_common(df, key, num_rows, num_columns, start_column=0):
    start_row = get_index_by_key(df, key) + 2
    return df.loc[start_row:start_row + num_rows - 1, start_column:start_column + num_columns]


def write_contact_matrix(df, filename):
    df.to_csv(filename, sep=' ', header=False, index=False)


def get_distribution(df):
    list_brackets = []
    for index, row in df.iterrows():
        bracket = list(map(int, row[0].split('_'))) + [float(row[1])]
        list_brackets.append(bracket)
    return list_brackets


def get_distribution_2(df):
    result_list = []
    for index, row in df.iterrows():
        bracket = [float(row[0]), float(row[1])]
        result_list.append(bracket)
    return result_list


def norm_dist(tmp_res):
    sum = 0
    for (_, num_f) in tmp_res:
        sum += num_f
    for i in range(len(tmp_res)):
        tmp_res[i][1] = tmp_res[i][1] / float(sum)

def main_households(df):
    def treat_ages_dist():
        key = "Age bracket distribution"
        tmp_df = treat_common(df=df, key=key, num_rows=16, num_columns=1)
        return get_distribution(tmp_df)

    def treat_household_dist():
        key = "Distribution of household sizes"
        tmp_df = treat_common(df=df, key=key, num_rows=7, num_columns=1)
        tmp_res = get_distribution_2(tmp_df)
        norm_dist(tmp_res)
        return tmp_res


    def treat_family_size_dist():
        key = "Age distribution of the reference person for each household size"
        tmp_df = treat_common(df=df, key=key, num_rows=8, num_columns=11)
        return list(map(list, tmp_df.values))

    def treat_household_contact_matrix():
        key = "Household contact matrix "
        tmp_df = treat_common(df=df, key=key, num_rows=16, num_columns=16, start_column=1)
        write_contact_matrix(tmp_df, "household_contact_matrix")

    result_dict = dict()
    ages_bracket = treat_ages_dist()
    result_dict["population_age_distributions"] = [{
        "num_bins": len(ages_bracket),
        "distribution": ages_bracket
    }]

    result_dict["household_head_age_brackets"] = [
        [18, 19],
        [20, 24],
        [25, 29],
        [30, 34],
        [35, 39],
        [40, 44],
        [45, 49],
        [50, 54],
        [55, 64],
        [65, 74],
        [75, 99]
    ]
    result_dict["household_head_age_distribution_by_family_size"] = treat_family_size_dist()
    result_dict["household_size_distribution"] = treat_household_dist()


    treat_household_contact_matrix()
    return result_dict


def main_education_place(df):
    key = "Size bracket distribution"
    tmp_df = treat_common(df=df, key=key, num_rows=14, num_columns=1)
    dist = get_distribution(tmp_df)
    result_list = []
    for d in dist:
        result_list.append(d[2])
    tmp_res = np.array(result_list)
    tmp_res = tmp_res / np.sum(tmp_res)
    return list(tmp_res)


def main_education_common(df):

    def treat_enrollment_by_age():
        key = "Enrollment by age"
        tmp_df = treat_common(df=df, key=key, num_rows=101, num_columns=1)
        return get_distribution_2(tmp_df)

    def treat_employment_rates_by_age():
        key = "Employment rates by age"
        tmp_df = treat_common(df=df, key=key, num_rows=101, num_columns=1)
        return get_distribution_2(tmp_df)

    def treat_school_contact_matrix():
        key = "School contact matrix "
        tmp_df = treat_common(df=df, key=key, num_rows=16, num_columns=16, start_column=1)
        write_contact_matrix(tmp_df, "school_contact_matrix")

    result_dict = dict()
    result_dict['enrollment_rates_by_age'] = treat_enrollment_by_age()
    result_dict['employment_rates_by_age'] = treat_employment_rates_by_age()
    treat_school_contact_matrix()

    return result_dict


def main_school(excel_file_location):
    df_education_common = pd.read_excel(excel_file_location, header=None, sheet_name="Education_Common")
    result_dict = main_education_common(df_education_common)

    # set school parameters by type of school
    result_dict['school_size_brackets'] = [
        [20, 50],
        [51, 100],
        [101, 300],
        [301, 500],
        [501, 700],
        [701, 900],
        [901, 1100],
        [1101, 1300],
        [1301, 1500],
        [1501, 1700],
        [1701, 1900],
        [1901, 2100],
        [2101, 2300],
        [2301, 2700]
      ]
    df_kindergarten = pd.read_excel(excel_file_location, header=None, sheet_name="Kindergarten")
    df_school = pd.read_excel(excel_file_location, header=None, sheet_name="School")
    df_university = pd.read_excel(excel_file_location, header=None, sheet_name="University")
    result_dict["school_size_distribution_by_type"] = [{
        "school_type": "pk",
        "size_distribution": main_education_place(df_kindergarten)
    }, {
        "school_type": "es",
        "size_distribution": main_education_place(df_school)
    }, {
        "school_type": "uv",
        "size_distribution": main_education_place(df_university)
    }]

    result_dict["school_types_by_age"] = [{
        "school_type": "pk",
        "age_range": [3, 6]
    }, {
        "school_type": "es",
        "age_range": [7, 17]
    }, {
        "school_type": "uv",
        "age_range": [18, 100]
    }]

    return result_dict


def main_work(df):
    """
    create working contact matrix and workplace_size_counts_by_num_personnel
    :param df:
    :return: dict that contain workplace_size_counts_by_num_personnel
    """
    def treat_workplace_size_dist():
        key = "Workplace size distribution"
        tmp_df = treat_common(df=df, key=key, num_rows=9, num_columns=1)
        return get_distribution(tmp_df)

    def treat_workplace_contact_matrix():
        key = "Workplace contact matrix "
        tmp_df = treat_common(df=df, key=key, num_rows=16, num_columns=16, start_column=1)
        write_contact_matrix(tmp_df, "work_contact_matrix")

    workplace_part_dict = dict()
    # treat first parameter
    workplace_part_dict["workplace_size_counts_by_num_personnel"] = treat_workplace_size_dist()
    # treat contact matrix
    treat_workplace_contact_matrix()

    return workplace_part_dict


def save_location_in_json(excel_file_location, json_file_location, location_name):
    # read data
    df_household = pd.read_excel(excel_file_location, header=None, sheet_name="Households")
    df_work = pd.read_excel(excel_file_location, header=None, sheet_name="Work")
    # operation with data
    result_dict = dict()
    school_dict = main_school(excel_file_location)
    household_dict = main_households(df_household)
    work_dict = main_work(df_work)

    result_dict.update(school_dict)
    result_dict.update(household_dict)
    result_dict.update(work_dict)

    result_dict['location_name'] = location_name

    with open(json_file_location, 'w') as json_file:
        json.dump(result_dict, json_file)

def get_other_parameters(excel_file_location):
    df_common = pd.read_excel(excel_file_location, header=None, sheet_name="Parameters")
    result_dict = dict()
    result_dict['location'] = df_common.iloc[0, 1]
    result_dict['n'] = df_common.iloc[1, 1]

    # treat other school parameters
    df_education = pd.read_excel(excel_file_location, header=None, sheet_name="Education_Common")

    def get_coef_by_key(key):
        return df_education.iloc[get_index_by_key(df_education, key) + 1, 0]
    result_dict['average_student_teacher_ratio'] = get_coef_by_key("The average number of students per teacher")
    result_dict['average_class_size'] = get_coef_by_key("Average class size")
    result_dict['inter_grade_mixing'] = get_coef_by_key("The average fraction of edges required to create edges between grades in the same school")
    result_dict['average_teacher_teacher_degree'] = get_coef_by_key("The average number of contacts per teacher with other teachers")
    result_dict['teacher_age_min'] = get_coef_by_key("Teacher age min")
    result_dict['teacher_age_max'] = get_coef_by_key("Teacher age max")
    result_dict['average_student_all_staff_ratio'] = get_coef_by_key("The average number of students per staff members at school (including both teachers and non teachers)")
    result_dict['average_additional_staff_degree'] = get_coef_by_key("The average number of contacts per additional non teaching staff in schools")
    result_dict['staff_age_min'] = get_coef_by_key("Staff age min")
    result_dict['staff_age_max'] = get_coef_by_key("Staff age max")

    return result_dict


def get_dict_contact_matrices(filename):
    result_dict = dict()
    df_education = pd.read_excel(filename, header=None, sheet_name="Education_Common")
    result_dict['S'] = treat_common(df=df_education, key="School contact matrix ", num_rows=16, num_columns=16, start_column=1).to_numpy(dtype=float)

    df_random = pd.read_excel(filename, header=None, sheet_name="Random")
    result_dict['C'] = treat_common(df=df_random, key="Random contact matrix ", num_rows=16, num_columns=16, start_column=1).to_numpy(dtype=float)

    df_work = pd.read_excel(filename, header=None, sheet_name="Work")
    result_dict['W'] = treat_common(df=df_work, key="Workplace contact matrix ", num_rows=16, num_columns=16, start_column=1).to_numpy(dtype=float)

    df_household = pd.read_excel(filename, header=None, sheet_name="Households")
    result_dict['H'] = treat_common(df=df_household, key="Household contact matrix ", num_rows=16, num_columns=16, start_column=1).to_numpy(dtype=float)

    return result_dict

def make_people_from_file(excel_filename, popfile):
    other_parameters = get_other_parameters(excel_filename)
    location_name = other_parameters['location']
    save_location_in_json(excel_filename, f"{SYNTHPOP_MAIN_PATH}{location_name}.json", location_name)
    contact_matrices = get_dict_contact_matrices(excel_filename)
    pars = sc.objdict(
        rand_seed                       = 123,
        country_location                = location_name,
        smooth_ages                     = True,
        window_length                   = 7,
        household_method                = 'infer_ages',
        with_non_teaching_staff         = 1,
        use_two_group_reduction         = 1,
        with_school_types               = 1,
        contact_matrices                = contact_matrices,
        school_mixing_type              = {'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'uv': 'age_and_class_clustered'},  # you should know what school types you're working with
    )
    pars.update(other_parameters)

    pop = sp.Pop(**pars)

    print(f"Initiation is successful!")
    if popfile is not None:
        pop.save(popfile)
        print(f"Population saved in {popfile}")
    return pop


class CommonParameters:
    def __init__(self,
                 rand_seed=123,
                 location="Novosibirsk",
                 n=10000
    ):
        self.rand_seed = rand_seed
        self.location_name = location
        self.n = n




class SchoolParameters:
    def __init__(self,
        school_size_brackets=None,
        average_student_teacher_ratio=None,
        average_class_size=None,
        inter_grade_mixing=None,
        average_teacher_teacher_degree=None,
        teacher_age_min=None,
        teacher_age_max=None,
        average_student_all_staff_ratio=None,
        average_additional_staff_degree=None,
        staff_age_min=None,
        staff_age_max=None,
        enrollment_rates_by_age=None,
        employment_rates_by_age=None,
        school_size_distribution_by_type=None,
        school_types_by_age=[{
            "school_type": "pk",
            "age_range": [3, 6]
        }, {
            "school_type": "es",
            "age_range": [7, 18]
        }, {
            "school_type": "uv",
            "age_range": [18, 100]
        }],
        contact_matrix=None
    ) -> None:
        self.school_size_brackets = school_size_brackets
        self.average_student_teacher_ratio = average_student_teacher_ratio
        self.average_class_size = average_class_size
        self.inter_grade_mixing = inter_grade_mixing
        self.average_teacher_teacher_degree = average_teacher_teacher_degree
        self.teacher_age_min = teacher_age_min
        self.teacher_age_max = teacher_age_max
        self.average_student_all_staff_ratio = average_student_all_staff_ratio
        self.average_additional_staff_degree = average_additional_staff_degree
        self.staff_age_min = staff_age_min
        self.staff_age_max = staff_age_max
        self.enrollment_rates_by_age = enrollment_rates_by_age
        self.employment_rates_by_age = employment_rates_by_age
        self.school_size_distribution_by_type = school_size_distribution_by_type
        self.school_types_by_age = school_types_by_age
        self.contact_matrix = contact_matrix

    @staticmethod
    def get_default_parameters(folder_name, i):
        with open(SYNTHPOP_MAIN_PATH + f'{folder_name}/{i}.json') as json_file:
            default_data = json.load(json_file)
            default_data_school = get_other_parameters(f'{folder_name}/{i}.xlsx')
            return SchoolParameters(
                average_student_teacher_ratio=default_data_school['average_student_teacher_ratio'],
                average_class_size=default_data_school['average_class_size'],
                inter_grade_mixing=default_data_school['inter_grade_mixing'],
                average_teacher_teacher_degree=default_data_school['average_teacher_teacher_degree'],
                teacher_age_min=default_data_school['teacher_age_min'],
                teacher_age_max=default_data_school['teacher_age_max'],
                average_student_all_staff_ratio=default_data_school['average_student_all_staff_ratio'],
                average_additional_staff_degree=default_data_school['average_additional_staff_degree'],
                staff_age_min=default_data_school['staff_age_min'],
                staff_age_max=default_data_school['staff_age_max'],
                school_size_brackets=default_data['school_size_brackets'],
                enrollment_rates_by_age=default_data['enrollment_rates_by_age'],
                employment_rates_by_age=default_data['employment_rates_by_age'],
                school_size_distribution_by_type=default_data['school_size_distribution_by_type'],
                school_types_by_age=default_data['school_types_by_age'],
                contact_matrix=get_dict_contact_matrices(f"{folder_name}/{i}.xlsx")['S']
            )

class HouseholdParameters:
    def __init__(self,
        household_size_distribution=None,
        household_head_age_distribution_by_family_size=None,
        population_age_distributions=None,
        contact_matrix=None
    ) -> None:
        self.contact_matrix = contact_matrix
        self.household_size_distribution = household_size_distribution
        self.household_head_age_distribution_by_family_size = household_head_age_distribution_by_family_size
        self.population_age_distributions = population_age_distributions
        self.household_head_age_brackets = [
            [18, 19],
            [20, 24],
            [25, 29],
            [30, 34],
            [35, 39],
            [40, 44],
            [45, 49],
            [50, 54],
            [55, 64],
            [65, 74],
            [75, 99]
        ]

    @staticmethod
    def get_default_parameters(folder_name, i):
        with open(SYNTHPOP_MAIN_PATH + f'{folder_name}/{i}.json') as json_file:
            default_data = json.load(json_file)
            # print(default_data)
            household_size_distribution = default_data['household_size_distribution']
            norm_dist(household_size_distribution)
            return HouseholdParameters(
                household_size_distribution=household_size_distribution,
                household_head_age_distribution_by_family_size=default_data['household_head_age_distribution_by_family_size'],
                population_age_distributions=default_data['population_age_distributions'],
                contact_matrix=get_dict_contact_matrices(f"{folder_name}/{i}.xlsx")['H']
            )

class WorkParameters:
    def __init__(self,
        workplace_size_counts_by_num_personnel=None,
        contact_matrix=None
    ) -> None:
        self.contact_matrix = contact_matrix
        self.workplace_size_counts_by_num_personnel = workplace_size_counts_by_num_personnel


    @staticmethod
    def get_default_parameters(folder_name, i):
        with open(SYNTHPOP_MAIN_PATH + f'{folder_name}/{i}.json') as json_file:
            default_data = json.load(json_file)
            return WorkParameters(
                workplace_size_counts_by_num_personnel=default_data['workplace_size_counts_by_num_personnel'],
                contact_matrix=get_dict_contact_matrices(f"{folder_name}/{i}.xlsx")['W']
            )


class RandomParameters:
    def __init__(self, contact_matrix=None) -> None:
        self.contact_matrix=contact_matrix

    @staticmethod
    def get_default_parameters(folder_name, i):
        return RandomParameters(
            contact_matrix=get_dict_contact_matrices(f"{folder_name}/{i}.xlsx")['C']
        )


def make_people_from_pars(i, folder_name,
    common_pars=None,
    household_pars=None,
    school_pars=None,
    work_pars=None,
    random_pars=None,
    filename=None
):
    household_pars=HouseholdParameters.get_default_parameters(folder_name, i)
    school_pars=SchoolParameters.get_default_parameters(folder_name, i)
    work_pars=WorkParameters.get_default_parameters(folder_name, i)
    random_pars=RandomParameters.get_default_parameters(folder_name, i)

    json_dict = dict()
    json_dict.update(vars(common_pars))
    json_dict.update(vars(household_pars))
    json_dict.update(vars(school_pars))
    json_dict.update(vars(work_pars))
    json_dict.pop('contact_matrix')


    # TODO (IvanKozlov98) very bad code
    json_file_location = f'{SYNTHPOP_MAIN_PATH}{common_pars.location_name}.json'
    with open(json_file_location, 'w') as json_file:
        json.dump(json_dict, json_file)

    contact_matrices = {
        'W': work_pars.contact_matrix,
        'H': household_pars.contact_matrix,
        'C': random_pars.contact_matrix,
        'S': school_pars.contact_matrix,
    }
    pars = sc.objdict(
        n                               = common_pars.n,
        rand_seed                       = common_pars.rand_seed,
        country_location                = common_pars.location_name,
        sheet_name                      = common_pars.location_name,
        smooth_ages                     = True,
        window_length                   = 7,
        household_method                = 'infer_ages',
        with_non_teaching_staff         = 1,
        use_two_group_reduction         = 1,
        with_school_types               = 1,
        average_student_teacher_ratio   = school_pars.average_student_teacher_ratio,
        average_class_size              = school_pars.average_class_size,
        inter_grade_mixing              = school_pars.inter_grade_mixing,
        average_teacher_teacher_degree  = school_pars.average_teacher_teacher_degree,
        teacher_age_min                 = school_pars.teacher_age_min,
        teacher_age_max                 = school_pars.teacher_age_max,
        average_student_all_staff_ratio = school_pars.average_student_all_staff_ratio,
        average_additional_staff_degree = school_pars.average_additional_staff_degree,
        staff_age_min                   = school_pars.staff_age_min,
        staff_age_max                   = school_pars.staff_age_max,
        contact_matrices                = contact_matrices,
        school_mixing_type              = {'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'uv': 'age_and_class_clustered'},  # you should know what school types you're working with
    )

    pop = sp.Pop(**pars)
    if filename is not None:
        pop.save(filename)
        print(f"Population saved in {filename}")

    return pop