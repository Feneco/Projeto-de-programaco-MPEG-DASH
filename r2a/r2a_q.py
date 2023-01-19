"""
@title: Q Learning based MPEG dash quality selection Client. 


@author: Wagner C.C. Batalha (wagnerweb2010@gmail.com) 2023

@description: Based on the article with the same name as the title. By using a
Q learning algorithm, we may be able to choose the best bitrate of a video
in order to maximize the Quality of Service. 
Note: In some parts of the code you may pass through some documentation lines or 
comments that will be something like "Equation(1)". These refer to the equations
of the article that are numbered. 
"""

from player.parser import *
from r2a.ir2a import IR2A
import numpy as np
import json



class QConfig():
    def __init__(self):
        with open("r2a/q_config.json") as f:
            q_config_parameters = json.load(f)

            self.learningRate         = q_config_parameters["Learning Rate"          ]
            self.discountRate         = q_config_parameters["Discount Rate"          ]
            self.maxOscillationLength = q_config_parameters["Max Oscillation Length" ]
            self.learningPhaseLength  = q_config_parameters["Learning Phase Length"  ]

            r_weights = q_config_parameters["Reward Weights"]
            self.weightQuality       = r_weights["Quality"       ]
            self.weightOscillation   = r_weights["Oscillation"   ]
            self.weightBufferFilling = r_weights["Buffer Filling"]
            self.weightBufferChange  = r_weights["Buffer Change" ]



class EnvironmentState:
    """
    Stores the Environment State variables. These are used
    On the Q algorithm to calculate rewards.
    """
    def __init__(self):
        self.maxQualityLevel = 0
        self.maxBandwidth    = 0
        self.maxBufferLength = 0

        self.bufferFilling       = 0
        self.bufferFillingChange = 0
        self.qualityLevel        = 0
        self.bandwidth           = 0
        self.oscillationLength   = 0
        self.oscillationDepth    = 0



class Rewards:
    """
    Class that calculates rewards. These are functions that uses
    the video, buffer and connection status to calculate rewards to chosen actions.
    """
    def __init__(self, qConfig:QConfig):
        self.qConfig  = qConfig
        self.envState = None

        self._lastQualityLevel  = 0
        self._lastBufferFilling = 0
        self._lastAction        = 0


    def rewardQuality(self) -> float:
        """
        Equation (2)
        """
        r = ((self.envState.qualityLevel    - 1)
            /(self.envState.maxQualityLevel - 1)) * 2 - 1
        return r


    def rewardOscillation(self) -> float:
        """
        Equation (3)
        """
        OLi   = self.envState.oscillationLength
        OLmax = self.qConfig.maxOscillationLength
        ODi   = self.envState.oscillationDepth
        if (OLi == 0):
            return 0
        else:
            n = -1 / pow(OLi, (2 / ODi))
            m = (OLi - 1) / ((OLmax-1) * pow(OLmax, (2 / ODi)))
            r = n + m
            return r


    def rewardBufferFilling(self) -> float:
        """
        Equation (4)
        """
        if (self.envState.bufferFilling <= ( 0.1 * self.envState.maxBufferLength )):
            r = -1.0
        else: # bufferHealth > 0.1 * maxBufferSize
            r = ((2 * self.envState.bufferFilling)
                /((1 - 0.1) * self.envState.maxBufferLength))
            r -= ((1 + 0.1)
                  /(1 - 0.1))
        return r


    def rewardBufferChange(self) -> float:
        """
        Equation (5)
        """
        if self.envState.bufferFilling <= self._lastBufferFilling:
            r = ((self.envState.bufferFilling - self._lastBufferFilling)
                /(self._lastBufferFilling))
        else: # self.conInfo.bufferFilling > self.lastBufferHealth
            r = ((self.envState.bufferFilling - self._lastBufferFilling)
                /(self.envState.bufferFilling - (self._lastBufferFilling // 2)))
        return r


    def getReward(self, envState:EnvironmentState) -> float:
        """
        Get reward given the parameters. self.rewardWeight Dictionary values can be changed to
        tune this method.
        """
        self.envState = envState
        r =  self.qConfig.weightQuality       * self.rewardQuality      ()
        r += self.qConfig.weightOscillation   * self.rewardOscillation  ()
        r += self.qConfig.weightBufferFilling * self.rewardBufferFilling()
        r += self.qConfig.weightBufferChange  * self.rewardBufferChange ()
        self._lastBufferFilling = self.envState.bufferFilling
        self._lastQualityLevel  = self.envState.qualityLevel



class Q:
    def __init__(self, nStates:int, nActions:int, rewardFunction:Rewards, qConfig:QConfig):
        self.nStates = nStates
        self.nActions = nActions
        self.q = np.zeros([nStates, nActions])
        self.qConfig = qConfig
        self.rewardFunction = rewardFunction

        self.iteration = 0
        self.lastActionSelection = 0
        self.lastState = 0


    def calculate_last_reward(self, environmentState:EnvironmentState):
        """
        Equation (1)
        """
        # The environmentState variable will be updated only after self.select_action() is called,
        # so it makes sense to update the Q table of the last state/action combination.

        if self.iteration == 0:
            # The first time this method runs, it won't have 
            # valid data to calculate stuff, so do nothing
            return
        
        r = self.rewardFunction.getReward(environmentState)
        qOldVal = self.q[self.lastState, self.lastActionSelection]
        qNewVal = qOldVal + self.qConfig.learningRate \
            * ( r + self.qConfig.discountRate * np.max(self.q[self.lastState,:]) - qOldVal )
        self.q[self.lastState, self.lastActionSelection] = qNewVal


    def _get_action(self, environmentState:EnvironmentState) -> int:
        # The actual Learn/Use_knowledge decision is made here.
        # For now it's just a simple algorithm that i made that will still apply some 
        # randomness in the choice after some iterations
        if self.iteration == 0:
            # First choice is going to be 0
            return 0
        elif self.iteration < self.qConfig.learningPhaseLength:
            # In the beginning the agent must learn, it will
            # pick truly at random
            return np.random.choice(self.nActions, 1)
        else:
            # After that, actions with greater rewards will
            # be picked far more
            s = environmentState.qualityLevel
            p = np.square(q[s,:])
            p = np.divide(p, np.sum(p))
            return np.random.choice(self.nActions, p=p)


    def select_action(self, environmentState:EnvironmentState) -> int:
        self.calculate_last_reward()
        chosenAction = self._get_action(environmentState)
        self.lastState = environmentState.qualityLevel
        self.lastActionSelection = chosenAction
        self.iteration += 1
        return chosenAction



#########################################################################################################
#########################################################################################################
class R2A_Q(IR2A):
    def __init__(self, id):
        IR2A.__init__(self, id)
        # List with the bitrates available. Only the Quantity
        # and Index of bitrates are used instead of the actual values
        self.bitrates = []
        # Load the Q specific configuration
        self.qConfig = QConfig()
        # Initiate the EnvState variable
        self.environmentState = EnvironmentState()
        # Creates a Reward Object
        self.rewardHandler = Rewards(self.qConfig)
        # The q agent is only initialized in self.handle_xml_response
        self.q = None


    def handle_xml_request(self, msg):
        self.send_down(msg)


    def handle_xml_response(self, msg):
        parsed_mpd = parse_mpd(msg.get_payload())
        self.bitrates = parsed_mpd.get_qi()

        # The lines in this block  are setting the following
        # variables of self.environmentState:
        # * maxQualityLevel
        # * maxBandwidth
        # * maxBufferLength
        # TODO: Find a way to get maxBandwidth
        self.environmentState.maxQualityLevel = len(self.bitrates)
        self.environmentState.maxBandwidth = 10000 # 10 kbps, for now
        with open('dash_client.json') as f:
            # This file is already open in another class, we could optimize if we could
            # reach it from here.
            config_parameters = json.load(f)
            self.environmentState.maxBufferLength = int(config_parameters["max_buffer_size"])

        # Initializing Q agent
        self.q = Q(N, N, self.rewardHandler, self.qConfig)

        self.send_up(msg)


    def handle_segment_size_request(self, msg):
        # Update the state with new values
        # then we get the quality level from the agent
        # after that we send it down
        self.updateEnvState()
        nextQualityLevel = self.q.select_action(self.environmentState)
        msg.add_quality_id(self.bitrates[nextQualityLevel])
        self.environmentState.qualityLevel = nextQualityLevel # update self.environmentState
        self.send_down(msg)


    def handle_segment_size_response(self, msg):
        self.send_up(msg)


    def initialize(self):
        pass


    def finalization(self):
        pass

    def updateEnvState(self):
        # TODO: update self.environmentState before proceeding with functions below
        # This function should update the following values:
        # * bufferFilling
        # * bufferFillingChange
        # * qualityLevel < This one is already updated in self.handle_segment_size_request() but we could update it here too
        # * bandwidth
        # * oscillationLength
        # * oscillationDepth
        pass