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
        mu: float,
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

        #! Important: In discrete models we need an array that contains
        #! the possible actions
        self.equity_ratios = np.linspace(self.action_space.low,
                                         self.action_space.high, self.n_action_discr)

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
        self.dt = T/n_discr

        # Starting period
        self.t = 0
        # Starting episode. One episode is completed when the agend
        # played once through the lifecycle
        # Number of states corresponds to number of episodes
        self.episode = 0

        # Initialize the bonds, stocks and difference matrices
        self.s = np.zeros((self.n_paths, self.n_discr + 1))
        self.b = np.zeros((self.n_paths, self.n_discr + 1))
        self.d_b = np.zeros((self.n_paths, self.n_discr))
        self.d_s = np.zeros((self.n_paths, self.n_discr))

        self.b[:, 0] = self.bond_price * np.ones(self.n_paths)
        self.s[:, 0] = self.stock_price * np.ones(self.n_paths)

        # Calculate bond and stock prices
        # self.stocks_bonds_prices()

        self.final_wealth = np.zeros(n_paths)

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

        #! Not a good solution. If the environment is stepped through
        #! with Merton agent, the actions are floats rather than indexes 
        #! (which are int type).
        #todo this will not work with multiple paths I think
        if type(actions) == int:
            weight = self.equity_ratios[actions]
        else:
            weight = actions

        if self.t < self.n_discr:
            n_stocks = weight * self.wealth / self.s[:, self.t]
            n_bonds = (1-weight) * self.wealth / self.b[:, self.t]
            # print(f"{self.s[:, self.t]= }")
            # print(f"{self.b[:, self.t]= }")
            # print(f"{n_stocks= }")
            # print(f"{n_bonds= }")

            d_x = n_stocks * self.d_s[:, self.t] + \
                n_bonds * self.d_b[:, self.t]
            # print(f"{d_x= }")
            # print(f"{d_x.shape= }")

            rewards = d_x - (self.kappa/2) * (d_x**2)

            self.wealth = self.wealth + d_x

            # print(f"{d_x= }")
            # print(f"{d_x.shape= }")
            # print(f"{self.wealth= }")
            # print(f"{self.wealth.shape= }")
            self.t += 1  # ! this had to be moved here

            self.states = np.array([
                self.s[:, self.t],
                self.wealth[:]
            ],
                dtype="float32"
            ).T

            dones = np.repeat(False, self.n_paths)

            logger.info(
                "Next state reached. Successfully stepped through environment.")

        else:
            logging.info(f"Environment has reached last step. ({self.t})")
            rewards = np.zeros(self.n_paths)
            d_x = np.zeros(self.n_paths)
            self.final_wealth = self.wealth
            dones = np.repeat(True, self.n_paths)

            _ = self.reset()

        # Save as attributes for rendering later on
        # self.rewards = rewards.copy()
        # self.states = states.copy()
        # self.actions = actions.copy()

        return (self.states, rewards, dones, {"d_x": d_x},)

    def reset(self):
        # Set time step to 0
        self.t = 0
        logger.info(
            f"""Initial time is t: {self.t}, Number of humans lives or Monte-Carlo-Simulation paths is set to {self.n_paths}.""")
        # Calculate returns of bonds and stocks
        self.stocks_bonds_prices()

        self.wealth = self.wealth_0 * np.ones(self.n_paths)
        # print("reset", f"{self.wealth=}")

        self.states = np.array([
            self.s[:, self.t],
            self.wealth[:]
        ],
            dtype="float32"
        ).T

        # self.states = states.copy()
        # print("reset", f"{self.states=}")
        return self.states

    def stocks_bonds_prices(self):
        # Create a matrix of random variables for calculation of the
        # stock prices via Black Scholes model
        # shape: (trajectories, discretisations)
        z = self.rng.standard_normal((self.n_paths, self.n_discr))
        # print(f"{z=}")

        # Calculate the absolute value.
        # s: stock prices
        # d_s: difference in stock prices
        # b: bond prices
        # d_b: difference in bond prices

        for t in range(1, self.n_discr + 1):
            self.s[:, t] = self.s[:, t-1] * np.exp((self.mu-0.5*self.sigma**2)*self.dt +
                                                   self.sigma*np.sqrt(self.dt)*z[:, t-1])

            self.b[:, t] = self.b[:, t-1] * \
                np.ones(self.n_paths)*np.exp(self.rf*self.dt)

        # print(f"{z.shape= }")
        # print(f"{self.s.shape= }")
        # print(f"{self.b.shape= }")
        # print(f"{self.s= }")
        # print(f"{self.b= }")

        for t in range(1, self.n_discr):
            self.d_s[:, t] = self.s[:, t+1] - self.s[:, t]
            self.d_b[:, t] = self.b[:, t+1] - self.b[:, t]

        # print(f"{self.d_s.shape= }")
        # print(f"{self.d_b.shape= }")

        # print(f"Shape of Bond and Stock return array:\n {s.shape=}, {b.shape=}")

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
            self.low = 0
            self.high = 5
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


if __name__ == "__main__":
    rf = 0.02
    mu = 0.1
    sigma = 0.2
    kappa = 0.008
    stock_price = 1.0
    bond_price = 1.0
    wealth_0 = 100

    T = 1
    n_paths = 1
    n_discr = 20
    seed = 0

    wealth = wealth_0 * np.ones(n_paths)

    env = MertonEnvironmentAbs(
        wealth_0,
        rf,
        mu,
        sigma,
        kappa,
        stock_price=stock_price,
        bond_price=bond_price,
        n_paths=n_paths,
        T=T,
        n_discr=n_discr,
        seed=seed,
        render=False
    )

    states_list = []
    for _ in range(1):
        for i in range(env.T):
            if i == 0:
                states = env.reset()
            else:
                states, rewards, dones, info = env.step(actions)

            # env.render()

            states_list.append(states)

            # actions = np.random.uniform(low=0, high=1, size=(n_paths,))
            actions = (np.ones(n_paths) * 2)

    print(np.array(states_list)[:, 0, 1])
    # env.renderer.join()
