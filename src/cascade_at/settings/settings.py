from cascade_at.settings import setting_components as sc
from cascade_at.settings.base_case import BASE_CASE


class Settings:
    def __init__(self):

        self.model = sc.Model()
        self.max_num_iter = sc.MaxNumIter()
        self.print_level = sc.PrintLevel()
        self.accept_after_max_steps = sc.AcceptAfterMaxSteps()
        self.students_dof = sc.StudentsDOF()
        self.log_students_dof = sc.LogStudentsDOF()
        self.eta = sc.Eta()
        self.tolerance = sc.Tolerance()

        self.config_version = None

        self.min_cv = []
        self.min_cv_by_rate = []
        self.covariate = []
        self.rate = []
        self.random_effect = []

    def configure(self):
        for att in BASE_CASE['model']:
            setattr(self.model, att, BASE_CASE['model'][att])
        for att in BASE_CASE['max_num_iter']:
            setattr(self.max_num_iter, att, BASE_CASE['max_num_iter'][att])
        for att in BASE_CASE['print_level']:
            setattr(self.print_level, att, BASE_CASE['print_level'][att])
        for att in BASE_CASE['accept_after_max_steps']:
            setattr(self.accept_after_max_steps, att, BASE_CASE['accept_after_max_steps'][att])
        for att in BASE_CASE['students_dof']:
            setattr(self.students_dof, att, BASE_CASE['students_dof'][att])
        for att in BASE_CASE['log_students_dof']:
            setattr(self.students_dof, att, BASE_CASE['log_students_dof'][att])
        for att in BASE_CASE['eta']:
            setattr(self.eta, att, BASE_CASE['eta'][att])
        if 'tolerance' in BASE_CASE.keys():
            for att in BASE_CASE['tolerance']:
                setattr(self.tolerance, att, BASE_CASE['tolerance'][att])

        self.config_version = BASE_CASE['config_version']

        for min_cv in BASE_CASE['min_cv']:
            mcv = sc.MinCV()
            for k in min_cv:
                setattr(mcv, k, min_cv[k])
            self.min_cv.append(mcv)

        for min_cv_by_rate in BASE_CASE['min_cv_by_rate']:
            mcv = sc.MinCVByRate()
            for k in min_cv_by_rate:
                setattr(mcv, k, min_cv_by_rate[k])
            self.min_cv_by_rate.append(mcv)

        for cov in BASE_CASE['country_covariate']:
            c = sc.Covariate()
            for k in cov:
                setattr(c, k, cov[k])
            c.configure()
            self.covariate.append(c)

        for rate in BASE_CASE['rate']:
            r = sc.Rate()
            for k in rate:
                setattr(r, k, rate[k])
            r.configure()
            self.rate.append(r)

        for random_effect in BASE_CASE['random_effect']:
            re = sc.RandomEffect()
            for k in random_effect:
                setattr(re, k, random_effect[k])
            re.configure()
            self.random_effect.append(re)


