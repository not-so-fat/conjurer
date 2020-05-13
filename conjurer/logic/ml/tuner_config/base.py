

class TunerConfig(object):
    def __init__(self, estimator_dict, distributions_rg, distributions_cl, grid_rg, grid_cl):
        self.estimator_dict = estimator_dict
        self.parameters_dict = self.get_default_parameters(
            distributions_rg, distributions_cl, grid_rg, grid_cl)

    @staticmethod
    def get_default_parameters(distributions, distributions_cl, grid, grid_cl):
        return {
            "grid": {
                **{ptype: grid_cl for ptype in ["cl", "mcl"]},
                **{"rg": grid}
            },
            "random": {
                **{ptype: distributions_cl for ptype in ["cl", "mcl"]},
                **{"rg": distributions}
            }
        }
