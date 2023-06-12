import numpy as np


class RobotInitialOscillator:
    """Reset the oscillator initial conditions

    Usually played with phase of the oscillator
    """

    def __init__(self, cond):
        """initialize object with int cond variable

        Args:
            cond (int, optional): variable that will determine the values
                                of oscillator phase.
                cond = 0        : ideal phases
                cond != 0       : random phases
        """
        self.cond = cond

    def setup(self, animat_options):
        """setup function, to be called when need to set the phases in animat_options

        Args:
            animat_options (_type_): _description_
        """
        if self.cond == 0:
            RobotInitialOscillator.ideal_oscillator_phase(animat_options)
        elif self.cond == -1:
            RobotInitialOscillator.random_oscillator_phase_set(animat_options)
        else:
            # random
            RobotInitialOscillator.random_oscillator_phase(animat_options)

    def ideal_oscillator_phase(animat_options):
        """Set ideal (2pi) phase for oscillators

        Args:
            animat_options (_type_): animat_option object
        """
        initial_phase_l = np.linspace(2 * np.pi, 0, 10)
        initial_phase_r = initial_phase_l - np.pi  # np.linspace(np.pi, -np.pi, 10)
        RobotInitialOscillator.set_oscillator_phase(
            animat_options=animat_options,
            initial_phase_l=initial_phase_l,
            initial_phase_r=initial_phase_r,
        )
        return

    def random_oscillator_phase(animat_options):
        """Set random phases for oscillators

        Args:
            animat_options (_type_): animat_option object
        """
        # initial_phase_l = np.random.random_sample(10)*(2*np.pi)
        # initial_phase_r = np.random.random_sample(10)*(2*np.pi) - np.pi
        initial_phase_l = np.random.random_sample(10) * (np.pi)
        initial_phase_r = (
            initial_phase_l - np.pi
        )  # np.random.random_sample(10) * (np.pi / 2)
        RobotInitialOscillator.set_oscillator_phase(
            animat_options=animat_options,
            initial_phase_l=initial_phase_l,
            initial_phase_r=initial_phase_r,
        )
        # print(initial_phase_l)
        # print(initial_phase_r)
        return

    def random_oscillator_phase_set(animat_options):
        """Set preset random conditions

        Args:
            animat_options (_type_): _description_
        """
        # wiggle
        initial_phase_l = np.array(
            [
                5.17145387,
                6.21031771,
                3.20626371,
                4.68446372,
                1.44003797,
                4.18264995,
                0.20604126,
                5.51063516,
                3.77788864,
                3.42433152,
            ]
        )
        initial_phase_r = initial_phase_l - np.pi  #  np.array(
        #     [
        #         -1.13958819,
        #         -0.41038592,
        #         -0.63139568,
        #         -1.20212702,
        #         1.84092402,
        #         -1.85471113,
        #         1.21257644,
        #         2.54544774,
        #         0.71828385,
        #         -2.63267731,
        #     ]
        # )
        RobotInitialOscillator.set_oscillator_phase(
            animat_options=animat_options,
            initial_phase_l=initial_phase_l,
            initial_phase_r=initial_phase_r,
        )
        return

    def set_oscillator_phase(animat_options, initial_phase_l, initial_phase_r):
        """set the phases (l and r) in animat_options object

        Args:
            animat_options (_type_): animat_option object
            initial_phase_l (array): phase for the left
            initial_phase_r (array): phase for the right
        """

        for i, elem in enumerate(animat_options.control.network.oscillators):
            if i % 2 == 0:
                # even - L
                index = int(i / 2)
                elem.initial_phase = initial_phase_l[index]
            else:
                # odd - R
                index = int((i - 1) / 2)
                elem.initial_phase = initial_phase_r[index]
        return


if __name__ == "__main__":
    raise ValueError("Not a file that is supposed to be run")
