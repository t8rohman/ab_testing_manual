import statsmodels.api as sm
import matplotlib.pyplot as plt


class ItsStats():
    """
    A class for analyzing interrupted time series (ITS) data using statistical models.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    its_model : statsmodels model
        The fitted ITS model using statsmodels.
    cf_model : statsmodels model
        The fitted counterfactual model using statsmodels.
    t_col : str
        The name of the column in `df` containing the time variable.
    y_col : str
        The name of the column in `df` containing the outcome variable.
    start : int
        The index representing the start of the intervention period.
    end : int
        The index representing the end of the intervention period.
    constant : bool, optional
        Whether to include a constant term in the models. Default is True.

    Methods:
    --------
    summary_frame():
        Returns a summary DataFrame containing the ITS and counterfactual model predictions.
    
    plot_its():
        Plots the actual, ITS model predictions, and counterfactual model predictions over time.
    """


    def __init__(self, df, its_model, cf_model, t_col, y_col, start, end, constant=True):
        """
        Initialize the ItsStats class with the provided data and models.
        """
        self.df = df
        self.its_model = its_model
        self.cf_model = cf_model
        self.t_col = t_col
        self.y_col = y_col
        self.start = start
        self.end = end
        self.constant = constant


    @staticmethod
    def summary_frame(self):
        """
        Return a summary DataFrame containing the ITS and counterfactual model predictions.
        """
        df = self.df
        its_model = self.its_model
        cf_model = self.cf_model
        y_col = self.y_col
        start = self.start
        end = self.end
        constant = self.constant

        y_pred = its_model.get_prediction(0, end - 1)
        y_its = y_pred.summary_frame(alpha=0.05)

        if constant == True:
            exog_w_intercept_cf = sm.add_constant(df[y_col][start:])
        else:
            exog_w_intercept_cf = df[y_col][start:]

        y_cf = cf_model.get_forecast(steps=end-start, exog=exog_w_intercept_cf).summary_frame(alpha=0.05)

        return y_its, y_cf


    def plot_its(self):
        """
        Plot the actual, ITS model predictions, and counterfactual model predictions over time.
        """
        df = self.df
        t_col = self.t_col
        y_col = self.y_col
        start = self.start
        
        y_its, y_cf = self.summary_frame()

        fig, ax = plt.subplots(figsize=(20,8))

        ax.scatter(df[t_col], df[y_col], facecolors='none', edgecolors='steelblue', label="actual", linewidths=2)

        # plot model mean bounce prediction
        ax.plot(df[t_col][:start], y_its['mean'][:start], 'b-', label="model prediction")
        ax.plot(df[t_col][start:], y_its['mean'][start:], 'b-')

        # plot the counterfactual
        ax.plot(df[t_col][start:], y_cf['mean'], 'k.', label="counterfactual")
        ax.fill_between(df[t_col][start:], y_cf['mean_ci_lower'], y_cf['mean_ci_upper'], color='k', alpha=0.1, label="counterfactual 95% CI")

        # plot line marking intervention moment
        ax.axvline(x=12.5, color='r', label='intervention')

        ax.legend(loc='best')
        plt.xlabel("T")
        plt.ylabel("Y")

        plt.show()