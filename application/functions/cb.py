import brica
import numpy as np

ACTION_AMP_RATE = 4.0
ACTION_CUTOFF = 0.1

class CB(object):
    """ Cerebellum module.
    
    CB outputs action for smooth pursuit eye movment.
    """
    def __init__(self):
        self.timing = brica.Timing(5, 1, 0)

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('CB did not recieve from FEF')

        fef_data = inputs['from_fef']
        
        action = ACTION_AMP_RATE * fef_data # fef data is too small
        if np.linalg.norm(action) < ACTION_CUTOFF: # for rapid convergence
            action *= 0.0 # action = np.array([0.0, 0.0])

        # Action values should be within range [-1.0~1.0]
        action = np.clip(action, -1.0, 1.0)

        return dict(to_environment=action)
