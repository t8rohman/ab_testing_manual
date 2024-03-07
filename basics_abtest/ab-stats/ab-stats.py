import math
import scipy.stats as stats


class SamplePowerAnalysis:
    '''
    Calculate the sample size needed for hypothesis tests based on effect size, power, alpha, and standard deviation.

    Attributes
    -----
    mu1 : float
        Mean before. For dichotomous variables, pass the null hypothesis proportion (p1).
    mu2 : float
        Mean after.
    mud : float
        Difference of means (mu2 - mu1). For dichotomous variables, pass the lift percentage (p2-p1).
    std_dev : float
        Standard deviation.
    alpha : float
        Level of statistical significance.
    power : float
        Level of statistical power.
    '''
    
    def __init__(self, mu1=None, mu2=None, mud=None, std_dev=None, alpha=0.05, power=0.8):
        self.mu1 = mu1
        self.mu2 = mu2
        self.std_dev = std_dev
        self.z_alpha = stats.norm.ppf(1 - alpha/2)
        self.z_power = stats.norm.ppf(power)

        if mud is None:
            self.mud = mu2 - mu1
        else:
            self.mud = mud


    @staticmethod
    def one_sample_n(z_alpha, z_power, effect_size):
        '''
        Calculate the sample size for a one-sample test.

        Parameters
        -----
        z_alpha : float
            Z-score corresponding to the desired significance level.
        z_power : float
            Z-score corresponding to the desired power.
        effect_size : float
            The desired effect size.

        Returns
        -----
        n : float
            The calculated sample size.
        '''
        n = ((z_alpha + z_power) / effect_size) ** 2
        return n

    @staticmethod
    def two_sample_n(z_alpha, z_power, effect_size):
        '''
        Calculate the sample size for a two-sample test.

        Parameters
        -----
        z_alpha : float
            Z-score corresponding to the desired significance level.
        z_power : float
            Z-score corresponding to the desired power.
        effect_size : float
            The desired effect size.

        Returns
        -----
        n : float
            The calculated sample size.
        '''
        n = 2 * ((z_alpha + z_power) / effect_size) ** 2
        return n


    # CONTINUOUS OUTCOME / VARIABLE

    def onesample_con(self):
        '''
        Calculate the sample size for a one-sample test with a continuous variable.
        '''
        mud = self.mud
        std_dev = self.std_dev
        z_alpha = self.z_alpha
        z_power = self.z_power

        effect_size = abs(mud) / std_dev
        n = self.one_sample_n(z_alpha, z_power, effect_size)

        return n

    def twosample_con(self):
        '''
        Calculate the sample size for a two-sample test with a continuous variable.
        '''
        mud = self.mud
        std_dev = self.std_dev
        z_alpha = self.z_alpha
        z_power = self.z_power

        effect_size = mud / std_dev
        n = self.two_sample_n(z_alpha, z_power, effect_size)

        return n

    def matchsample_con(self):
        '''
        Calculate the sample size for a matched-sample test with a continuous variable.
        '''
        mud = self.mud
        std_dev = self.std_dev
        z_alpha = self.z_alpha
        z_power = self.z_power

        effect_size = mud / std_dev
        n = self.one_sample_n(z_alpha, z_power, effect_size)

        return n


    # DICHOTOMOUS OUTCOME / VARIABLE

    def onesample_dic(self):
        '''
        Calculate the sample size for a one-sample test with a dichotomous variable.
        '''
        p1 = self.mu1
        pd = self.mud
        z_alpha = self.z_alpha
        z_power = self.z_power

        effect_size = abs(pd) / math.sqrt(p1 * (1 - p1))
        n = self.one_sample_n(z_alpha, z_power, effect_size)

        return n

    def twosample_dic(self):
        '''
        Calculate the sample size for a two-sample test with a dichotomous variable.
        '''
        p1 = self.mu1
        pd = self.mud
        z_alpha = self.z_alpha
        z_power = self.z_power

        effect_size = abs(pd) / math.sqrt(p1 * (1 - p1))
        n = self.two_sample_n(z_alpha, z_power, effect_size)

        return n