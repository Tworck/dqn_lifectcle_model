# %%
# Import numpy modules
from hashlib import new
import numpy as np
from numpy.random import RandomState
from numpy import ndarray, newaxis
from gym import spaces

# Import logging module
import logging
import logging.config

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import multiprocessing
import time  # needed to wait for window to open

# Setup logger
logger = logging.getLogger(__name__)


class MertonEnvironment:
    def __init__(
        self,
        wealth_0: float,
        rf: float,
        mu,
        sigma: float,
        kappa: float,
        stock_price: float = 1.0,
        bond_price: float = 1.0,
        n_paths: int = 1,
        T: int = 1,
        n_discr: int = 1,
        n_action_discr: int = 1,
        seed: int = None,
        render=False,
    ):
        # If a seed is specified, the algorithm should take this seed
        # for reproducibility purposes. Else do not use a specific seed
        if seed != None:
            self.rng = np.random.RandomState(seed)
            logger.info(f"Stochastic processes initialized with seed {seed}")
        else:
            # This generates a random Random seed each time instead of
            # a fixed seed like in the if case
            self.rng = np.random.RandomState()
            logger.warning("No random seed was used.")

        # The dimensions are necessary to provide the replay buffer with
        # dimensions. We call it observation and action space because
        # this is how it is done in the gym environment from open ai
        # We want to keep the nomenclature so that we are able to call
        # other environments accordingly. See documentation of gym
        # environment
        # number of discretizations for possible actions that the agend
        # can choose
        self.n_action_discr = n_action_discr
        self.observation_space = self.ObservationSpace()
        self.action_space = self.ActionSpace(
            self.rng, n_paths, self.n_action_discr)

        # Number of trajectories to be simulated. In the easiest case
        # this includes the number of stock trajectories
        self.n_paths = n_paths
        # Number of discretization steps to calculate between starting
        # period (currently T=0) and final period (i.e. T = 100)
        self.n_discr = n_discr
        # Starting wealth of investment
        #! hard coded the wealth, this is bäh
        self.wealth_0 = wealth_0
        self.wealth = self.wealth_0 * np.ones((self.n_paths, 1))
        # Risk free retrun
        self.rf = rf
        # Drift term for calculating stock returns
        self.mu = mu
        # Volatility of the stocks
        self.sigma = sigma
        # Risk aversion parameter for mean variance utility approach
        self.kappa = kappa
        # Stock and Bond prices
        # todo this is the simplest form with only 2 base prices
        self.stock_price = stock_price
        self.bond_price = bond_price
        # Time horizon of investment. Can mean 10 years, can mean 100
        # days. depends on interpretation and rest of the model
        self.T = T
        self.dt = T / n_discr

        # Starting period
        self.t = 0
        # Starting episode. One episode is completed when the agend
        # played once through the lifecycle
        # Number of states corresponds to number of episodes
        self.episode = 0

        # Calculate returns of bonds and stocks
        # self.s, self.b = self.stocks_bonds_growth()

        # Reward range where the 0th entry is the lowest possible reward
        # and the 1st entry is the biggest possible rewards
        self.reward_range = (0, np.inf)

        if render:
            self.renderer = self.Renderer(self)
            self.renderer.daemon = True
            self.renderer.start()
            # Wait 2s for window to open
            time.sleep(2)

    def step(self, actions: ndarray):

        # weight = actions.squeeze()
        weight = actions
        # print(weight)

        if self.t < self.n_discr:

            s_portfolio_return = np.multiply(weight, self.s[:, self.t] - 1)
            b_portfolio_return = np.multiply(
                (1 - weight), self.b[:, self.t] - 1)

            self.portfolio_growth = s_portfolio_return + b_portfolio_return + 1

            # new_wealth is an absolute value
            self.new_wealth = self.wealth * self.portfolio_growth
            #! Clipping might not be necessary
            self.new_wealth = self.new_wealth.clip(1e-6)

            rewards = self.reward()
            # states array needs to have shape (n_paths, states)
            # i.e. (5 paths, 2 states) where states is s and portfolio_growth
            states = np.array(
                [self.s[:, self.t], self.portfolio_growth[:]], dtype="float32"
            ).T

            dones = np.repeat(False, self.n_paths)
            logger.info(
                "Next state reached. Successfully stepped through environment.")
            self.t += 1

        # The else statement functions as the reset of the environment
        else:
            logging.info(f"Environment has reached last step. ({self.t})")
            states = self.reset()
            rewards = self.reward()
            dones = np.repeat(True, self.n_paths)

        # Save as attributes for rendering later on
        self.rewards = rewards.copy()
        self.states = states.copy()
        # self.actions = actions.copy()

        return (states, rewards, dones, {})

    def reset(self):
        # Set time step to 0
        self.t = 0
        logger.info(
            f"""Initial time is t: {self.t}, Number of humans lives or Monte-Carlo-Simulation paths is set to {self.n_paths}."""
        )
        # Calculate returns of bonds and stocks
        self.s, self.b = self.stocks_bonds_growth()
        # Portfolio growth in the first period is 1 (starting wealth)
        self.portfolio_growth = np.ones(self.n_paths)
        # Initialize a wealth array. This currently only includes a
        # fixed wealth wealth_0
        self.new_wealth = self.wealth_0 * np.ones(self.n_paths)
        self.wealth = self.new_wealth

        # The state should correspond to the stock growth rate in the respective
        # time step and the wealth growth rate
        states = np.array(
            [self.s[:, self.t], self.portfolio_growth[:]], dtype="float32"
        ).T
        # Save as attributes for rendering later on
        self.states = states.copy()

        self.t += 1

        return states

    def reward(self):
        # d_wealth = self.new_wealth - self.wealth
        #! is the order correct now?
        rewards = np.log(1+self.new_wealth/self.wealth)
        # rewards = d_wealth - (self.kappa / 2) * (d_wealth ** 2)
        self.wealth = self.new_wealth
        # print(rewards)
        return rewards.squeeze()

    def stocks_bonds_growth(self):
        # Create a matrix of random variables for calculation of the
        # stock prices via Black Scholes model
        # shape: (trajectories, discretisations)
        z = self.rng.standard_normal((self.n_paths, self.n_discr))

        # Calculate the returns. Working with returns makes it easier
        # to vectorize the calculations
        # s: stock growths
        # b: bonds growths
        s = np.exp(
            (self.mu - 0.5 * self.sigma ** 2) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
        )
        # print(f"{s.shape=}")
        # print(f"{s=}")

        b = np.ones((self.n_paths, self.n_discr)) * np.exp(self.rf * self.dt)
        compound_s = np.cumprod(s)
        compound_b = np.cumprod(b)
        # print(f"{compound_s=}\n {compound_b=}")
        # print(f"Shape of Bond and Stock return array:\n {s.shape=}, {b.shape=}")
        return s, b

    def merton_ratio(self):
        merton_ratio = (self.mu - self.rf) / self.sigma ** 2
        return merton_ratio

    def mu_from_ratio(self, equity_ratio: ndarray):
        mu = (self.sigma**2 * equity_ratio) + self.rf
        return mu

    def render(self):
        if hasattr(self, "actions"):
            self.renderer.queue.put(
                (self.t, self.actions, self.states, self.rewards))
        else:
            # In the 0th period we do not have actions or rewards,
            # thats why we pass None
            self.renderer.queue.put((self.t, None, self.states, None))

    class ActionSpace:
        """Represents the space in which actions can be chosen for this
        environment.

        It holds the same attributes as you would generally find in a OpenAI
        gym environment. The action spaces of gym for continuous problems
        are generally of type Box. This is not the case here.

        Args:
            rng (numpy.random.RandomState): Random number generator that
                can be seeded and is used to generate random number for
                stochastic variables.
        """

        def __init__(self, rng, n_paths, n_action_discr):
            self.rng = rng
            self.n_paths = n_paths
            self.n_action_discr = n_action_discr
            self.shape = (n_action_discr,)

        def sample(self):  # -> ndarray:
            """Sample random actions from a uniformly distributed
            distribution.

            Pull a weight from the distribution. It allows the agent to
            go into a short position. This value is decided as the mean
            of distribution.

            Args:
                n_paths (int, optional): Number of monte carlo paths for
                    which to initialize or calculate the initial state
                    of the environment. Defaults to 1.

            Returns:
                ndarray: Array of random actions for exploration period.
                    Generated with Dirichlet distribution using identical
                    concentration values alpha=1 for all dimensions.
                    Only used within the warm-up phase.

            """
            # Pull a weight from the distribution. It can allow the agent to
            # go into a short position. This value is decided as the mean
            # of distribution.
            actions = self.rng.randint(self.n_action_discr)
            # Return a relative action
            return actions

        # -> ndarray:
        def get_noise(self, n_paths: int = 1, noise_factor: float = 0.01):
            """Generate noise for the actions

            Generate noise for the action (weight). The multiplicator
            'level' adjusts the strength of applied noise.

            Args:
                n_paths (int, optional): Number of monte carlo paths for
                    which to initialize or calculate the initial state
                    of the environment. Careful: This n_paths is not the 
                    same as the self.n_paths of the evironment above.
                    Defaults to 1.
                noise_factor (float, optional): Factor to  scale the noise
                    that is added to the action. This is a hyperparameter
                    of the Agent. Defaults to 0.01.

            Returns:
                ndarray: returns noise values for the actions.
            """
            # Generate noise for the action (weight). The multiplicator
            # 'level' adjusts the strength of applied noise.
            noise = self.rng.normal(
                loc=0, scale=noise_factor, size=(n_paths, 1))

            return noise

    class ObservationSpace:
        """
        Represents the space in which states are observed for this environment.

        It holds the same attributes as you would generally find in a OpenAI
        gym environment. The observation spaces of gym for continuous problems
        are generally of type Box. This is not the case here.
        """

        def __init__(self):
            self.shape = (2,)
            self.high = np.array([np.inf, np.inf], dtype=np.float32)
            self.low = np.array([-np.inf, -np.inf], dtype=np.float32)

    class Renderer(multiprocessing.Process):
        def __init__(self, environment, update_intervall=20):
            super(multiprocessing.Process, self).__init__()

            self.queue = multiprocessing.Queue()

            self.update_intervall = update_intervall  # ms
            self.end_time = environment.n_discr
            self.env = environment

            # --- Preallocate arrays that hold the average data
            self.avg_actions = np.zeros(
                (self.end_time, environment.action_space.shape[0])
            )
            self.avg_states = np.zeros(
                (self.end_time, environment.observation_space.shape[0])
            )
            self.avg_reward = np.zeros(self.end_time)

        def run(self):
            # ---
            app = QtGui.QApplication([])
            self.win = pg.GraphicsLayoutWidget(show=True)
            self.win.setWindowTitle("Learning and Performance Overview")

            self.setup()

            # --- Generate Timer that updates plot every intervall
            # Needs to be attribute, otherwise will be garbage collected
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_data)
            self.timer.start(self.update_intervall)
            app.exec()

        def setup(self):
            # --- Create plots
            self.state_plot = self.win.addPlot()
            self.win.nextRow()
            self.action_plot = self.win.addPlot()
            self.win.nextRow()
            self.reward_plot = self.win.addPlot()

            # --- Show Grid
            self.state_plot.showGrid(x=True, y=True)
            self.action_plot.showGrid(x=True, y=True)
            self.reward_plot.showGrid(x=True, y=True)

            # --- Show Legend
            self.state_plot.addLegend()
            self.action_plot.addLegend()
            self.reward_plot.addLegend()

            # --- Set x and y axis labels
            self.state_plot.setLabel("bottom", "period")
            self.action_plot.setLabel("bottom", "period")
            self.reward_plot.setLabel("bottom", "period")

            #! Change labels
            self.state_plot.setLabel("left", "stock return")
            self.action_plot.setLabel("left", "weight factor")
            self.reward_plot.setLabel("left", "utility [a.u.]")

            # --- Set range of x axis
            self.state_plot.setXRange(0, self.end_time, padding=0)
            self.action_plot.setXRange(0, self.end_time, padding=0)
            self.reward_plot.setXRange(0, self.end_time, padding=0)

            # --- Create line styles
            stocks_pen = pg.mkPen(color="#2ca02c", width=1,
                                  style=QtCore.Qt.SolidLine)
            wealth_pen = pg.mkPen(color="#A9A9A9", width=1,
                                  style=QtCore.Qt.SolidLine)
            weight_pen = pg.mkPen(color="#d62728", width=1,
                                  style=QtCore.Qt.SolidLine)

            reward_pen = pg.mkPen(
                color="#9467bd", width=1.5, style=QtCore.Qt.SolidLine)

            # --- Create line references
            self.stocks = self.state_plot.plot(
                pen=stocks_pen, name="stocks growth")
            self.wealth = self.state_plot.plot(
                pen=wealth_pen, name="portfolio growth")
            self.weight = self.action_plot.plot(pen=weight_pen, name="weight")

            self.reward = self.reward_plot.plot(pen=reward_pen, name="reward")

        def update_data(self):
            while not self.queue.empty():
                self.t, actions, states, rewards = self.queue.get()

                if self.t == 1:  # needed to change this to one for some reason
                    self.avg_states[self.t] = np.mean(states, axis=0)
                elif self.t == self.end_time:
                    self.avg_actions[self.t - 1] = np.mean(actions, axis=0)
                    self.avg_reward[self.t - 1] = np.mean(rewards)
                else:
                    # --- Calculate averages
                    self.avg_states[self.t] = np.mean(states, axis=0)
                    self.avg_actions[self.t - 1] = np.mean(actions, axis=0)
                    self.avg_reward[self.t - 1] = np.mean(rewards)

            if hasattr(self, "t"):
                self.update_plot()

        def update_plot(self):
            # --- Update plots
            # this is the stock array
            self.stocks.setData(self.avg_states[: self.t, 0])
            self.wealth.setData(self.avg_states[: self.t, 1])
            # this is the "weight"
            self.weight.setData(self.avg_actions[: self.t, 0])

            self.reward.setData(self.avg_reward[: self.t])


# %%

# if __name__ == "__main__":

#     np.random.seed(0)

#     rf = 0.02
#     mu = 0.1
#     sigma = 0.2
#     kappa = 0.008
#     stock_price = 1.0
#     bond_price = 1.0
#     wealth_0 = 100

#     T = 1
#     n_paths = 1
#     n_discr = 20
#     seed = 0

#     mu = np.ones(n_discr) * 0.1

#     # Ugly way to do this. maybe make something with vstack/ hstack
#     # for generalizing
#     #! only works properly for 20 ndiscr
#     # equity_ratios = np.ones(n_discr)
#     # equity_ratios[0:5] = 2
#     # equity_ratios[5:10] = 1
#     # equity_ratios[10:15] = 0.5
#     # equity_ratios[15:20] = 3

#     # equity_ratios = np.tile(equity_ratios,(n_paths,1))

#     wealth = wealth_0 * np.ones(n_paths)

#     env = MertonEnvironment(
#         wealth_0,
#         rf,
#         mu,
#         sigma,
#         kappa,
#         stock_price=stock_price,
#         bond_price=bond_price,
#         n_paths=n_paths,
#         T=T,
#         n_discr=n_discr,
#         seed=seed,
#         render=True,
#     )

#     states_list = []
#     reward_list = []

#     # env.mu = env.mu_from_ratio(equity_ratios)
#     # print(env.mu)
#     # merton_ratios = env.merton_ratio()
#     # print(merton_ratios)
#     # print(equity_ratios)

#     for _ in range(1):
#         for i in range(env.n_discr):
#             if i == 0:
#                 states = env.reset()
#             else:
#                 states, rewards, dones, info = env.step(actions)
#                 reward_list.append(rewards)

#             env.render()

#             states_list.append(states)

#             # actions = np.random.uniform(low=0, high=1, size=(n_paths,))
#             actions = np.ones((n_paths,)) * 1.2
#             # actions = equity_ratios

#     # print(np.array(states_list)[:,0])
#     # pg = np.cumproduct(np.array(states_list)[:, 0, 1])
#     # w = wealth_0 * pg
#     # print(f"{w=}")
#     # print(np.array(reward_list))
#     # print(states.shape)
#     # print(env.wealth)
#     env.renderer.join()


# %%
