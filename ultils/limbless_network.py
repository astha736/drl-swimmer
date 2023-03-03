import numpy as np

class LimblessExperimentNetwork:
    """ Complete reset of the network 

        Aim: test different networks with different network types 
            in both open and closed loop
     """
    agnathax_links = ['head','body_0','body_1','body_2','body_3','body_4','body_5','body_6','body_7','body_8','body_9']

    def __init__(self, c_intra=10, c_inter=None, s_local=None, s_rostl=None, s_caudl=None, **kwargs):
        """_summary_

        Args:
            c_intra (int, optional): osc2osc connectivity. Defaults to 10.
            c_inter (int, optional): osc2osc connectivity. Defaults to None.
            s_local (int, optional): joint2osc connectivity. Defaults to None.
            s_rostl (int, optional): joint2osc connectivity. Defaults to None.
            s_caudl (int, optional): joint2osc connectivity. Defaults to None.
        
        kwargs:
        self.s_local_senstivity. Senstivity function to be used. Defaults to cos
        self.s_rostl_senstivity. Senstivity function to be used. Defaults to cos
        self.s_caudl_senstivity. Senstivity function to be used. Defaults to cos

        senstivity function decides the final equations. This is done via "conn_type"
        that each network map has.
        
        Note:
            c_intra: internal connection in each segment. Left-Right oscillator
            c_inter: nearest neighbour connectivity
            s_local: stretch local in the segment 
            s_rostl: stretch feedback towards the head. 1 segment above 
            s_caudl: stretch feedback towards the tail. 1 segment below
        
        In limbless experiment we repurpose the types of stretch feedback
            'cos'      -> 'STRETCH2AMPTEGOTAE'      -> freq equation
            'sin'      -> 'STRETCH2FREQTEGOTAE'     -> freq equation
            'nonperiod'-> 'STRETCH2FREQ'            -> freq equation
        """
        self.c_intra = c_intra
        self.c_inter = c_inter
        self.s_local = s_local
        self.s_rostl = s_rostl
        self.s_caudl = s_caudl

        # will contain shallow copy of the connectivity
        self.c_intra_array = None
        self.c_inter_array = None
        self.s_local_array = None
        self.s_rostl_array = None
        self.s_caudl_array = None
        self.contact_local_array = None


        def conn_type(senstivity):
            if senstivity == None:
                conn_type = None
            elif senstivity == 'cos':
                conn_type = 'STRETCH2AMPTEGOTAE'
            elif senstivity == 'sin':
                conn_type = 'STRETCH2FREQTEGOTAE'
            elif senstivity == 'nonperiod':
                conn_type = 'STRETCH2FREQ'
            elif senstivity == 'SignalNone':
                conn_type = 'STRETCH2AMP'
            else:
                raise ValueError("Can be cos, sin, nonperiod. Provided {}".format(senstivity))
            return conn_type

        # choose the senstivity for a given connection type
        self.s_local_senstivity = conn_type(kwargs.pop('s_local_senstivity', None))
        self.s_rostl_senstivity = conn_type(kwargs.pop('s_rostl_senstivity', None))
        self.s_caudl_senstivity = conn_type(kwargs.pop('s_caudl_senstivity', None))

        self.contact_local = kwargs.pop('contact_local', None)
        self.contact_local_senstivity = conn_type(kwargs.pop('contact_local_senstivity', 'REACTION2FREQ'))
    
    def setup(self, animat_options):
        """ setup the network desired by changing the animat_options

        Args:
            animat_options (AnimatOptions): object of class 
        """
        animat_options.control.network.osc2osc = []
        animat_options.control.network.joint2osc = []
        animat_options.control.network.contact2osc = []

        if self.c_intra is not None:
            self.c_intra_array = LimblessExperimentNetwork.add_intra_segmental_conn(
                network_osc2osc=animat_options.control.network.osc2osc,
                weight=self.c_intra,
                )
        if self.c_inter is not None:
            self.c_inter_array = LimblessExperimentNetwork.add_inter_segmental_conn(
                network_osc2osc=animat_options.control.network.osc2osc,
                weight=self.c_inter,
                )
        if self.s_local is not None:
            self.s_local_array = LimblessExperimentNetwork.add_stretch_conn_local(
                network_joint2osc=animat_options.control.network.joint2osc,
                conn_type=self.s_local_senstivity,
                weight=self.s_local,
                )
        if self.s_rostl is not None:
            self.s_rostl_array = LimblessExperimentNetwork.add_stretch_conn_rostral(
                network_joint2osc=animat_options.control.network.joint2osc,
                conn_type=self.s_rostl_senstivity,
                weight=self.s_rostl,
                )
        if self.s_caudl is  not None:
            self.s_caudl_array = LimblessExperimentNetwork.add_stretch_conn_caudal(
                network_joint2osc=animat_options.control.network.joint2osc,
                conn_type=self.s_caudl_senstivity,
                weight=self.s_caudl,
            )
        if self.contact_local is not None:
            self.contact_local_array = LimblessExperimentNetwork.add_contact_conn(
                network_contact2osc=animat_options.control.network.contact2osc,
                conn_type=self.contact_local_senstivity,
                weight=self.contact_local,
            )
        # animat_options.control.network.contact2osc
        return 

    def add_intra_segmental_conn(network_osc2osc, weight=10):
        """Add intra segmental connectivity
        using a shallow copy of the variables
        """
        intra_segmental_conn = []
        # left to right
        intra_segmental_conn += [
            {
                'in': 'osc_body_{}_R'.format(i),
                'out': 'osc_body_{}_L'.format(i),
                'type': 'OSC2OSC',
                'weight': weight,
                'phase_bias': np.pi,
            } for i in range(0, 10)
        ]
        # right to left
        intra_segmental_conn += [    
            {
                'in': 'osc_body_{}_L'.format(i),
                'out': 'osc_body_{}_R'.format(i),
                'type': 'OSC2OSC',
                'weight': weight,
                'phase_bias': np.pi,
            } for i in range(0, 10)
        ]
        network_osc2osc += intra_segmental_conn
        return intra_segmental_conn

    def add_inter_segmental_conn(network_osc2osc, weight=10):
        """ Add inter segmental connectivity
            Note: Since in farms, the motor orientation is 
            opposite of usual condideration. 
            Usual: Motor mounted on link i+1, and moves link i. 
            Hence, the signs of phase bias are reversed 

            This allows to get a forward travelling wave
        """
        inter_segmental_conn = []
        # downwards
        inter_segmental_conn += [
            {
                'in': 'osc_body_{}_L'.format(i+1),
                'out': 'osc_body_{}_L'.format(i),
                'type': 'OSC2OSC',
                'weight': weight,
                'phase_bias': (2*np.pi)/10,
            } for i in range(0, 9)
        ]
        inter_segmental_conn += [    
            {
                'in': 'osc_body_{}_R'.format(i+1),
                'out': 'osc_body_{}_R'.format(i),
                'type': 'OSC2OSC',
                'weight': weight,
                'phase_bias': (2*np.pi)/10,
            } for i in range(0, 9)
        ]
        # upwards
        inter_segmental_conn += [
            {
                'in': 'osc_body_{}_L'.format(i),
                'out': 'osc_body_{}_L'.format(i+1),
                'type': 'OSC2OSC',
                'weight': weight,
                'phase_bias': -(2*np.pi)/10,
            } for i in range(0, 9)
        ]
        inter_segmental_conn += [
            {
                'in': 'osc_body_{}_R'.format(i),
                'out': 'osc_body_{}_R'.format(i+1),
                'type': 'OSC2OSC',
                'weight': weight,
                'phase_bias': -(2*np.pi)/10,
            } for i in range(0, 9)
        ]
        network_osc2osc += inter_segmental_conn
        return inter_segmental_conn

    def add_stretch_conn_local(network_joint2osc, conn_type, weight=10):
        """ Stretch feedback in local segment

        Note: Use a shallow copy of 'stretch_conn_local'. 
                This will allow later access 
        """
        stretch_conn_local = []
        stretch_conn_local += [
            {
                'in': 'osc_body_{}_L'.format(i),
                'out': 'joint_{}'.format(i),
                'type': conn_type,
                'weight': weight,
            } for i in range(0, 10)
        ]
        stretch_conn_local += [
            {
                'in': 'osc_body_{}_R'.format(i),
                'out': 'joint_{}'.format(i),
                'type': conn_type,
                'weight': weight*-1,
            } for i in range(0, 10)
        ]

        network_joint2osc += stretch_conn_local
        return stretch_conn_local

    def add_stretch_conn_caudal(network_joint2osc, conn_type, weight=10):
        """Stretch feedback in caudal direction: head to tail"""
        stretch_conn_caudal = []
        stretch_conn_caudal += [
            {
                'in': 'osc_body_{}_L'.format(i+1),
                'out': 'joint_{}'.format(i),
                'type': conn_type,
                'weight': weight,
            } for i in range(0, 9)
        ]

        # right
        stretch_conn_caudal += [
            {
                'in': 'osc_body_{}_R'.format(i+1),
                'out': 'joint_{}'.format(i),
                'type': conn_type,
                'weight': weight*-1,
            } for i in range(0, 9)
        ]
        network_joint2osc += stretch_conn_caudal
        return stretch_conn_caudal

    def add_stretch_conn_rostral(network_joint2osc, conn_type, weight=10):
        """Stretch feedback in rostral direction: tail to head"""

        stretch_conn_rostral = []
        stretch_conn_rostral += [
            {
                'in': 'osc_body_{}_L'.format(i),
                'out': 'joint_{}'.format(i+1),
                'type': conn_type,
                'weight': weight,
            } for i in range(0, 9)
        ]

        # right
        stretch_conn_rostral += [
            {
                'in': 'osc_body_{}_R'.format(i),
                'out': 'joint_{}'.format(i+1),
                'type': conn_type,
                'weight': weight*-1,
            } for i in range(0, 9)
        ]
        network_joint2osc += stretch_conn_rostral
        return stretch_conn_rostral

    def add_contact_conn(network_contact2osc, conn_type, weight=10):
        
        # left osc
        network_contact2osc += [
            {
                'in':'osc_body_{}_L'.format(i),
                'out': LimblessExperimentNetwork.agnathax_links[i],
                'type': conn_type, #'REACTION2FREQ',
                'weight': weight,
            } for i in range(0,10)
        ]

        # right osc
        network_contact2osc += [
            {
                'in':'osc_body_{}_R'.format(i),
                'out': LimblessExperimentNetwork.agnathax_links[i],
                'type': conn_type, #'REACTION2FREQ',
                'weight': weight*-1,
            } for i in range(0,10)
        ]
        return 
        

if __name__ == '__main__':
    raise ValueError("Not a file that is supposed to be run")