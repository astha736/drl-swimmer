import numpy as np

class RobotInitialDomain:
    """ Simulation domain parameter options """

    def __init__(self, friction):
        self.friction = friction

    def reset_morphology_friction(morphology_links, friction):
        for elem in morphology_links:
            elem['friction'] = [friction, 0, 0]
    
    def setup(self, animat_options):
        RobotInitialDomain.reset_morphology_friction(morphology_links=animat_options.morphology.links, friction=self.friction)
