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
from render import RenderHeatMap


class QConfig():
    def __init__(self):
        with open("r2a/q_config.json") as f:
            q_config_parameters = json.load(f)

            self.learningRate         = q_config_parameters["Learning Rate"          ]
            self.discountRate         = q_config_parameters["Discount Rate"          ]
            self.maxOscillationLength = q_config_parameters["Max Oscillation Length" ]

            self.inverseSensitivity   = q_config_parameters["Inverse sensitivity"    ]

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
        return r



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

        self.explorationProbability = np.ones([nStates])
        self.actionInfluence = 1/nActions
        self.qLearning = 0


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
        self.qLearning = ( r + self.qConfig.discountRate * np.max(self.q[environmentState.qualityLevel,:]) - qOldVal ) #alterei aqui, tava self.q[self.lastState, :]
        qNewVal = qOldVal + self.qConfig.learningRate * self.qLearning
        self.q[self.lastState, self.lastActionSelection] = qNewVal


    def _get_action(self, environmentState:EnvironmentState) -> int:
        # The actual Learn/Use_knowledge decision is made here.
        # For now it's just a simple algorithm that i made that will still apply some 
        # randomness in the choice after some iterations
        s = environmentState.qualityLevel
        E = np.random.uniform(0, 1)

        if E < self.explorationProbability[s]:
            # softmax(nActions)
            actions = (self.q[s, :])
            e_x = np.exp(actions - np.max(actions))
            softmax = e_x / e_x.sum()
            return np.random.choice(self.nActions, 1, p=softmax)[0]

        # argmax(Q(s,b))
        print("no exploration: ", np.argmax(self.q[s, :]))
        return int(np.argmax(self.q[s, :]))


    def select_action(self, environmentState:EnvironmentState) -> int:
        self.calculate_last_reward(environmentState)
        chosenAction = self._get_action(environmentState)
        self.update_exploration_probability(environmentState)
        self.lastState = environmentState.qualityLevel
        self.lastActionSelection = chosenAction
        self.iteration += 1
        return chosenAction

    def update_exploration_probability(self, environmentState:EnvironmentState):
            #maior mudança foi aqui, tinha feito errado, (antes) basicamente calculando a probabilidade como se fosse uma só pra todos, e não uma pra cada estado.
        xd = np.exp(-(abs(self.qConfig.learningRate * self.qLearning) / self.qConfig.inverseSensitivity))
        self.explorationProbability[environmentState.qualityLevel] = self.actionInfluence * ((1 - xd) / (1 + xd)) + (1 - self.actionInfluence) * self.explorationProbability[environmentState.qualityLevel]


#########################################################################################################
#########################################################################################################
class R2A_Q(IR2A):
    def __init__(self, id):
        IR2A.__init__(self, id)
        self.r = RenderHeatMap()
        # List with the bitrates available. Only the amount
        # and Index of bitrates are used instead of the actual kbps values
        self.bitrates = []
        # Load the Q specific configuration
        self.qConfig = QConfig()
        # Initiate the EnvState variable
        self.environmentState = EnvironmentState()
        # Creates a Reward Object
        self.rewardHandler = Rewards(self.qConfig)
        # The q agent is only initialized in self.handle_xml_response
        self.q = None
        # History of qualities levels chosen. Used to calculate oscillation variables
        self.qualityHistory = []


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
        self.environmentState.maxBandwidth = 10000 # 10 kbps, for now. It's not being used, yet.
        with open('dash_client.json') as f:
            # This file is already open in another class, we could optimize if we could
            # reach it from here.
            config_parameters = json.load(f)
            self.environmentState.maxBufferLength = int(config_parameters["max_buffer_size"])
        
        # Initializing Q agent
        self.q = Q(self.environmentState.maxQualityLevel, 4,
                   self.rewardHandler, self.qConfig)

        self.send_up(msg)


    def handle_segment_size_request(self, msg):
        # Update the state with new values
        # then we get the quality level from the agent
        # after that we send it down
        self.updateEnvState()
        choosenaction = self.q.select_action(self.environmentState)
        nextQualityLevel = self.environmentState.qualityLevel
        
        if   choosenaction == 0:
            nextQualityLevel -= 4
        elif choosenaction == 1:
            nextQualityLevel -= 2
        elif choosenaction == 2:
            nextQualityLevel += 0
        elif choosenaction == 3:
            nextQualityLevel += 1
        nextQualityLevel = max(0, min(nextQualityLevel, self.environmentState.maxQualityLevel-1))

        msg.add_quality_id(self.bitrates[nextQualityLevel])
        self.environmentState.qualityLevel = nextQualityLevel # update self.environmentState
        self.send_down(msg)
        
        self.qualityHistory.append(nextQualityLevel)
        if len(self.qualityHistory) > self.qConfig.maxOscillationLength:
            self.qualityHistory.pop(0)
        self.r.renderframe(self.q.q)


    def handle_segment_size_response(self, msg):
        self.send_up(msg)


    def initialize(self):
        pass


    def finalization(self):
        pass

    def updateEnvState(self):
        # TODO: This function should update the following values:
        # * bufferFilling
        # * bufferFillingChange
        # * qualityLevel < This one is already updated in self.handle_segment_size_request() but we could update it here too
        # * bandwidth < probably can be calculated using self.handle_segment_size_response() and calculating the time it takes to send messages. As it's not being used, it is not a priority.
        # * oscillationLength
        # * oscillationDepth

        # Buffer Variables
        self.getBuffer()
        # Oscillation Calculation
        self.getOscillation(self.qualityHistory)


    def getBuffer(self):
        buffer = self.whiteboard.get_playback_buffer_size()
        thisBufferFill = 0
        if len(buffer) > 0:
            thisBufferFill = buffer[-1][1] # get the last segment buffer length
        lastBufferFill = self.environmentState.bufferFilling
        self.environmentState.bufferFilling = thisBufferFill
        self.environmentState.bufferFillingChange = thisBufferFill - lastBufferFill


    def getOscillation(self, l):
        oscillationStartIndex = 0
        for i in range(len(l)-1):
            if l[i] > l[i+1]:
                oscillationStartIndex = i
                break
        if len(l) > 0:
            oscillationStartValue = l[oscillationStartIndex]
            if l[-1] >= oscillationStartValue:
                self.environmentState.oscillationLength = 0
                self.environmentState.oscillationDepth  = 0
            else:
                self.environmentState.oscillationLength = self.qConfig.maxOscillationLength - oscillationStartIndex
                self.environmentState.oscillationDepth  = oscillationStartValue - l[-1]
