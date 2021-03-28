import pandas
import numpy
from scipy.optimize import minimize
from src.profile import profile
from plotly.subplots import make_subplots
import plotly.graph_objects as go

diffuse_state = 0
diffuse_variance = 1e4


class LocalLevelModel:
    data_train = None
    sigma_eps2 = None
    sigma_nu2 = None

    def __init__(self):
        pass

    def observation_equation(self, x):
        return x

    def inv_observation_equation(self, y):
        return y

    def set_data_train(self, y):
        if isinstance(y, pandas.Series):
            self.data_train = self.inv_observation_equation(y)
        else:
            raise ValueError("Data must be a pandas.Series object.")

    @staticmethod
    def get_diffuse_initial_state():
        return diffuse_state, diffuse_variance

    @staticmethod
    def error_variance(state_variance, sigma_eps2):
        return state_variance + sigma_eps2

    @staticmethod
    def kalman_gain(error_variance, state_variance):
        return state_variance / error_variance

    def filter_step(self, state, state_variance, obs, sigma_eps2):
        error = obs - state
        if numpy.isnan(error):
            error_variance = numpy.nan
            estimator = state
            estimator_variance = state_variance
        else:
            error_variance = self.error_variance(state_variance, sigma_eps2)
            kalman_gain = self.kalman_gain(error_variance, state_variance)
            estimator = state + kalman_gain * error
            estimator_variance = state_variance * (1. - kalman_gain)
        return {'state': estimator,
                'state_variance': estimator_variance,
                'error': error,
                'error_variance': error_variance}

    @staticmethod
    def prediction_step(estimator, estimator_variance, sigma_nu2):
        prediction = estimator
        prediction_variance = estimator_variance + sigma_nu2
        return {'state': prediction, 'state_variance': prediction_variance}

    def run_recursion(self, state0, state_variance0, sigma_eps2, sigma_nu2, data=None):
        if data is None:
            data = self.data_train
        results = self.kalman_recursion(data=data, sigma_eps2=sigma_eps2, sigma_nu2=sigma_nu2,
                                        state0=state0, state_variance0=state_variance0)

        estimator_results = self.consolidate_results(data, results['estimator'],
                                                     results['estimator_variance'], 'estimator')
        prediction_results = self.consolidate_results(data, results['prediction'][:-1],
                                                      results['prediction_variance'][:-1], 'prediction')
        error_results = self.consolidate_results(data, results['error'],
                                                 results['error_variance'], 'error')

        return {'estimator': estimator_results, 'prediction': prediction_results, 'error': error_results}

    def smooth_states(self, results, sigma_eps2, data=None):
        if data is None:
            data = self.data_train
        error_variance = results['error']['error_variance']
        error_gain = sigma_eps2 / error_variance
        error = results['error']['error']
        wsi = numpy.zeros(error.shape)  # weighted sum of innovarions/error
        wsivi = numpy.zeros(error.shape)  # weighted sum of the inverse variaces of innovaions

        prediction = results['prediction']['prediction']
        prediction_variance = results['prediction']['prediction_variance']
        smoothed_state = numpy.zeros(error.shape)
        smoothed_state_variance = numpy.zeros(error.shape)
        n = len(error)
        for t_ in range(n):
            t = n - t_ - 1
            if t == (n - 1):
                wsi[t] = 0.
                wsivi[t] = 0.
            else:
                wsi[t] = error[t+1] / error_variance[t+1] + error_gain[t+1] * wsi[t+1]
                smoothed_state[t+1] = prediction[t+1] + prediction_variance[t+1] * wsi[t]
                wsivi[t] = 1. / error_variance[t+1] + error_gain[t+1] ** 2 * wsivi[t+1]
                smoothed_state_variance[t+1] = prediction_variance[t+1]-prediction_variance[t+1] ** 2 * wsivi[t]
        wsi0 = error[0] / error_variance[0] + error_gain[0] * wsi[0]
        smoothed_state[0] = prediction[0] + prediction_variance[0] * wsi0
        wsivi0 = 1./ error_variance[0] + error_gain[0] ** 2 * wsivi[0]
        smoothed_state_variance[0] = prediction_variance[0] - prediction_variance[0] ** 2 * wsivi0

        results['smoothed'] = self.consolidate_results(data,
                                                       smoothed_state,
                                                       smoothed_state_variance, 'smoothed')
        results['weighted_error'] = self.consolidate_results(data,
                                                             wsi,
                                                             wsivi, 'weighted_error')
        return results


    @profile
    def kalman_recursion(self, data, sigma_eps2, sigma_nu2, state0, state_variance0):
        n_size = len(data)
        results = dict()
        results['estimator'] = numpy.empty([n_size])
        results['estimator_variance'] = numpy.empty([n_size])
        results['prediction'] = numpy.empty([n_size + 1])
        results['prediction'][0] = state0
        results['prediction_variance'] = numpy.empty([n_size + 1])
        results['prediction_variance'][0] = state_variance0
        results['error'] = numpy.empty([n_size])
        results['error_variance'] = numpy.empty([n_size])
        for i in range(len(data)):
            filter_result = self.filter_step(state=results['prediction'][i],
                                             state_variance=results['prediction_variance'][i],
                                             obs=data.iloc[i],
                                             sigma_eps2=sigma_eps2)
            results['estimator'][i] = filter_result['state']
            results['estimator_variance'][i] = filter_result['state_variance']
            results['error'][i] = filter_result['error']
            results['error_variance'][i] = filter_result['error_variance']

            prediction_result = self.prediction_step(estimator=filter_result['state'],
                                                     estimator_variance=filter_result['state_variance'],
                                                     sigma_nu2=sigma_nu2)
            results['prediction'][i + 1] = prediction_result['state']
            results['prediction_variance'][i + 1] = prediction_result['state_variance']

        return results

    @staticmethod
    def consolidate_results(data, value, variance, name):
        n = len(value)
        results = pandas.DataFrame(data=numpy.hstack([value.reshape([n, 1]), variance.reshape([n, 1])]),
                                   index=data.index,
                                   columns=[name, '%s_variance' % name])
        return results

    def fit(self):
        sigma_eps20 = 1
        sigma_nu20 = 1

        res = minimize(fun=self.min_loc_likelihood, x0=numpy.array([sigma_eps20, sigma_nu20]), method='BFGS')

        self.sigma_eps2 = max(res.x[0], 0)
        self.sigma_nu2 = max(res.x[1], 0)

        return res

    @profile
    def log_likelihood(self, x):
        sigma_eps2 = x[0]
        sigma_nu2 = x[1]
        state0, state_variance0 = self.get_diffuse_initial_state()
        data = self.data_train.copy()

        results = self.kalman_recursion(data=data,
                                        sigma_eps2=sigma_eps2,
                                        sigma_nu2=sigma_nu2,
                                        state0=state0,
                                        state_variance0=state_variance0)
        mask = ~numpy.isnan(results['error'])
        error_variance = numpy.maximum(results['error_variance'][mask][1:], 1e-12)
        log_error_term1 = numpy.log(error_variance)
        log_error_term2 = numpy.divide(numpy.power(results['error'][mask][1:], 2),
                                       error_variance)
        logl = -0.5 * numpy.sum(log_error_term1 + log_error_term2)
        return logl

    @profile
    def min_loc_likelihood(self, x):
        return -self.log_likelihood(x)

    def plot_model(self, results, y, x=None, smoothed=False):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("(i)", "(ii)", "(iii)", "(iv)"))
        try:
            t = y.index
        except AttributeError:
            t = numpy.arange(len(y))

        if smoothed:
            confidence_interval = results['smoothed']['smoothed_variance'].apply(lambda v: 1.96 * numpy.sqrt(v))

            fig.add_trace(
                go.Scatter(x=t, y=results['smoothed']['smoothed'] - confidence_interval, name='smoothed lower',
                           mode='lines', line=dict(color='blue', width=0.), opacity=0., showlegend=False),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['smoothed']['smoothed'] + confidence_interval, name='smoothed upper',
                           mode='lines', line=dict(color='blue', width=0.), opacity=0., fill='tonexty', showlegend=False),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['smoothed']['smoothed'], name='smoothed', mode='lines',
                           line=dict(color='blue')),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['smoothed']['smoothed_variance'], name='smoothed_variance', mode='lines'),
                row=1, col=2
            )

        else:
            confidence_interval = results['estimator']['estimator_variance'].apply(lambda v: 1.96 * numpy.sqrt(v))

            fig.add_trace(
                go.Scatter(x=t, y=results['estimator']['estimator'] - confidence_interval, name='estimator lower',
                           mode='lines', line=dict(color='blue', width=0.), opacity=0., showlegend=False),
                row=1, col=1
            )


            fig.add_trace(
                go.Scatter(x=t, y=results['estimator']['estimator'] + confidence_interval, name='estimator upper',
                           mode='lines', line=dict(color='blue', width=0.), opacity=0., fill='tonexty', showlegend=False),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['estimator']['estimator'], name='estimator', mode='lines',
                           line=dict(color='blue')),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['estimator']['estimator_variance'], name='estimator_variance', mode='lines'),
                row=1, col=2
            )

        if x is not None:
            fig.add_trace(
                go.Scatter(x=t, y=x, name='state', mode='lines'),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(x=t, y=self.inv_observation_equation(y), name='observation', mode='markers',
                       marker=dict(color='red'), opacity=0.7),
            row=1, col=1
        )

        if smoothed:
            fig.add_trace(
                go.Scatter(x=t, y=results['weighted_error']['weighted_error'], name='weighted_error', mode='lines'),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['weighted_error']['weighted_error_variance'], name='weighted_error_variance', mode='lines'),
                row=2, col=2
            )

            fig.update_yaxes(title_text="Smoothing cumulant", row=2, col=1)
            fig.update_yaxes(title_text="Snoothing variance cumulant", row=2, col=2)

        else:
            fig.add_trace(
                go.Scatter(x=t, y=results['error']['error'], name='error', mode='lines'),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['error']['error_variance'], name='error_variance', mode='lines'),
                row=2, col=2
            )

            fig.update_yaxes(title_text="Error", row=2, col=1)
            fig.update_yaxes(title_text="Error Variance", row=2, col=2)

        fig.update_yaxes(title_text="Effecive Reproduction Number", row=1, col=1)
        fig.update_yaxes(title_text="Effecive Reproduction Number Variance", row=1, col=2)
        fig.update_layout(height=1000, width=1000, showlegend=False)
        fig.show()

    def plot_state(self, results, y, x=None, smoothed=False):
        fig = go.Figure()
        try:
            t = y.index
        except AttributeError:
            t = numpy.arange(len(y))

        if smoothed:
            confidence_interval = results['smoothed']['smoothed_variance'].apply(lambda v: 1.96 * numpy.sqrt(v))

            fig.add_trace(
                go.Scatter(x=t, y=results['smoothed']['smoothed'] - confidence_interval, name='smoothed lower',
                           mode='lines', line=dict(color='blue', width=0.), opacity=0., showlegend=False)
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['smoothed']['smoothed'] + confidence_interval, name='smoothed upper',
                           mode='lines', line=dict(color='blue', width=0.), opacity=0., fill='tonexty', showlegend=False)
            )

            fig.add_trace(
                go.Scatter(x=t, y=results['smoothed']['smoothed'], name='smoothed', mode='lines',
                           line=dict(color='blue')))

        else:
            confidence_interval = results['estimator']['estimator_variance'].apply(lambda v: 1.96 * numpy.sqrt(v))

            fig.add_trace(
                go.Scatter(x=t, y=results['estimator']['estimator'] - confidence_interval, name='estimator lower',
                           mode='lines', line=dict(color='blue', width=0.), opacity=0., showlegend=False))

            fig.add_trace(
                go.Scatter(x=t, y=results['estimator']['estimator'] + confidence_interval, name='estimator upper',
                           mode='lines', line=dict(color='blue', width=0.), opacity=0., fill='tonexty', showlegend=False))

            fig.add_trace(
                go.Scatter(x=t, y=results['estimator']['estimator'], name='estimator', mode='lines',
                           line=dict(color='blue'))
            )

        if x is not None:
            fig.add_trace(
                go.Scatter(x=t, y=x, name='state', mode='lines')
            )

        fig.add_trace(
            go.Scatter(x=t, y=self.inv_observation_equation(y), name='observation', mode='markers',
                       marker=dict(color='red'), opacity=0.7)
        )

        fig.add_trace(
            go.Scatter(x=t, y=numpy.ones(len(y)), name='observation', mode='lines',
                       line=dict(
                           color="black",
                           width=2,
                           dash="dashdot",
                       ))
        )

        fig.update_yaxes(title_text="Effecive Reproduction Number")
        fig.update_layout(showlegend=False)
        fig.show()
