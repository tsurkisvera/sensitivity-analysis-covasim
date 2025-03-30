import covasim as cv
import pandas as pd
import location_preprocessor as lp
import time
import concurrent.futures
import os
import sys 

folder_name = sys.argv[1]
number_of_points = int(sys.argv[2])
number_of_parameters = int(sys.argv[3])

if not os.path.exists(f'{folder_name}_results'):
    os.makedirs(f'{folder_name}_results')

def single_build_and_run(i):
    print(i)
    try:
        pop_size = 1e5
        pars = {"pop_size": pop_size, "pop_type": 'synthpops', 'n_days': 180}

        pop = lp.make_people_from_pars(i + 1, folder_name,
            common_pars=lp.CommonParameters(
                rand_seed=i,
                location=f"Novosibirsk_{i}",
                n=pop_size
            ),
        filename=None
        ) 

        class check_severe(cv.Analyzer):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.n = 0
                return

            def apply(self, sim):
                n = len(cv.true(sim.people.severe))
                if n > self.n:
                    self.n = n

        class check_critical(cv.Analyzer):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.n = 0
                return

            def apply(self, sim):
                n = len(cv.true(sim.people.critical))
                if n > self.n:
                    self.n = n

        sim = cv.Sim(pars=pars, rand_seed=i, verbose=False, variants=[cv.variant('wild', days=0)], analyzers=[check_severe(label='severe'), check_critical(label='critical')], pop_infected=30).init_people(prepared_pop=pop)

        df = pd.read_excel(f'{folder_name}/{i + 1}.xlsx', sheet_name='Covasim_Parameters')
        community = float(df['contacts'][0][1:-1])
        sim.pars['contacts']['c'] = community

        sim.run()
        with open(f'{folder_name}_results/{i}', 'w') as file:
            file.write(' '.join(list(map(str, sim.results['new_infections'].values))) + '\n')
            file.write(str(max(sim.results['new_infections'].values)) + '\n')
            file.write(str(len(cv.true(sim.people.dead))) + '\n')
            file.write(str(sim.get_analyzer('severe').n) + '\n')
            file.write(str(sim.get_analyzer('critical').n))
    except:
        print('there was an error somewhere')

cities_count = number_of_points * (number_of_parameters + 2)
max_workers = 120

if __name__ == '__main__':
    print("Start time")
    t1 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for i in range(0, cities_count, max_workers):
            res_sims = pool.map(single_build_and_run, range(i, min(cities_count, i + max_workers)))
            print(f"_______{i}________")

    # single_build_and_run(1)
    
    t2 = time.time()
    print(f"Total time {t2 - t1}")