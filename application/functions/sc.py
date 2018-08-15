import brica

class SC(object):
    def __init__(self):
        self.timing = brica.Timing(6, 1, 0)

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('SC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('SC did not recieve from BG')
        
        fef_data = inputs['from_fef']
        bg_data = inputs['from_bg']

        action = self._decide_action(fef_data, bg_data)
        return dict(to_environment=action)

    def _decide_action(self, fef_data, bg_data):
        max_likelihoood = -1.0
        decided_ex = 0.0
        decided_ey = 0.0

        # TODO: デバッグで現在最大のlikelihoodを持つactionを反映している
        for data in fef_data:
            likelihood = data[0]
            ex = data[1]
            ey = data[2]
            if likelihood > 0.1 and likelihood > max_likelihoood:
                decided_ex = ex
                decided_ey = ey
                max_likelihoood = likelihood
                
        action = [decided_ex * 0.01, decided_ey * 0.01]
        return action
