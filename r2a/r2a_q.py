"""
@author: Wagner C.C. Batalha (wagnerweb2010@gmail.com) 2023

@description: Q Learning based MPEG dash quality selection Client. 

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



class ConnectionInfo:
    """
    Stores the conInfo variables.
    """
    def __init__(self):
        self.maxQualityLevel = 0
        self.maxBandwidth = 0
        self.maxBufferLength = 0

        self.bufferFilling       = 0
        self.bufferFillingChange = 0
        self.qualityLevel        = 0
        self.bandwidth           = 0
        self.oscillationLength   = 0
        self.oscillationDepth    = 0



class Rewards:
    def __init__(self, config:QConfig):
        self.config = config
        self.conInfo = None

        self._lastQualityLevel = 0
        self._lastBufferFilling = 0
        self._lastAction = 0


    def rewardQuality(self) -> float:
        """
        Equation (2)
        """
        r = ((self.conInfo.qualityLevel    - 1)
            /(self.conInfo.maxQualityLevel - 1)) * 2 - 1
        return r
    

    def rewardOscillation(self) -> float:
        """
        Equation (3)
        """
        OLi   = self.conInfo.oscillationLength
        OLmax = self.config.maxOscillationLength
        ODi   = self.conInfo.oscillationDepth
        if (OLi == 0):
            return 0
        else:
            n = -1 / pow(OLi, (2 / ODi))
            m = (OLi - 1) / ((OLmax-1)*pow(OLmax, (2 / ODi)))
            r = n + m
            return r


    def rewardBufferFilling(self) -> float:
        """
        Equation (4)
        """
        if (self.conInfo.bufferFilling <= ( 0.1 * self.conInfo.maxBufferLength )):
            r = -1.0
        else: # bufferHealth > 0.1 * maxBufferSize
            r = ((2 * self.conInfo.bufferFilling)
                /((1 - 0.1) * self.conInfo.maxBufferLength))
            r -= ((1 + 0.1) 
                  /(1 - 0.1))
        return r
    
    
    def rewardBufferChange(self) -> float:
        """
        Equation (5)
        """
        if self.conInfo.bufferFilling <= self._lastBufferFilling:
            r = ((self.conInfo.bufferFilling - self._lastBufferFilling)
                /(self._lastBufferFilling))
        else: # self.conInfo.bufferFilling > self.lastBufferHealth
            r = ((self.conInfo.bufferFilling - self._lastBufferFilling)
                /(self.conInfo.bufferFilling - (self._lastBufferFilling // 2)))
        return r


    def getReward(self, connectionInfo:ConnectionInfo) -> float:
        """
        Get reward given the parameters. self.rewardWeight Dictionary values can be changed to
        tune this method.
        """
        self.conInfo = connectionInfo
        r =  self.config.weightQuality       * self.rewardQuality      ()
        r += self.config.weightOscillation   * self.rewardOscillation  ()
        r += self.config.weightBufferFilling * self.rewardBufferFilling()
        r += self.config.weightBufferChange  * self.rewardBufferChange ()
        self._lastBufferFilling = self.conInfo.bufferFilling
        self._lastQualityLevel  = self.conInfo.qualityLevel
        
    
    # def updateQTable(self):
    #     self.conInfo.updateValues()
    #     qValMax = np.max(self.qTable.q[self.conInfo.qualityLevel,:])
    #     qVal    = self.qTable.q[self.conInfo.qualityLevel, self._lastAction]

    #     # Equation(1)
    #     self.qTable.q[self.conInfo.qualityLevel, self._lastAction] = \
    #         ( qVal + self.config.learningRate * ( self.config.discountRate * qValMax - qVal ) )



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


    def calculate_last_reward(self, environmentState:ConnectionInfo):
        """
        Equation (1)
        """
        r = self.rewardFunction.getReward(environmentState)
        qOldVal = self.q[self.lastState, self.lastActionSelection]
        qValMax = np.max(self.q[self.lastState,:])
        qNewVal = qOldVal + self.qConfig.learningRate * ( r + self.qConfig.discountRate * qValMax - qOldVal )
        self.q[self.lastState, self.lastActionSelection] = qNewVal


    def get_action(self, environmentState:ConnectionInfo) -> int:
        if self.iteration == 0:
            # First choice is going to be 0
            return 0
        elif self.iteration < self.qConfig.learningPhaseLength:
            # In the beginning the agent must learn, it will
            # pick randomly
            return np.random.choice(self.nActions, 1)
        else:
            # After that, actions with greater rewards will
            # be picked far more
            s = environmentState.qualityLevel
            p = np.square(q[s,:])
            p = np.divide(p, np.sum(p))
            return np.random.choice(self.nActions, p=p)


    def select_action(self, environmentState:ConnectionInfo) -> int:
        self.calculate_last_reward()
        action = self.get_action(environmentState)
        self.lastState = environmentState.qualityLevel
        self.lastActionSelection = action
        self.iteration += 1
        return action
       


#########################################################################################################
#########################################################################################################
class R2A_Q(IR2A):
    def __init__(self, id):
        IR2A.__init__(self, id)
        self.bitrates = []
        self.qConfig = QConfig()
        self.environmentState = ConnectionInfo()
        self.rewardHandler = Rewards(self.qConfig)
        self.q = None # initialized in self.handle_xml_response


    def handle_xml_request(self, msg):
        self.send_down(msg)


    def handle_xml_response(self, msg):
        parsed_mpd = parse_mpd(msg.get_payload())
        self.bitrates = parsed_mpd.get_qi()
        # setting variables of self.conInfo:
        #   maxQualityLevel
        #   maxBandwidth
        #   maxBufferLength
        N = len(self.List)
        self.environmentState.maxQualityLevel = N
        # TODO: Find a way to get maxBandwidth
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
        # TODO: update self.environmentState before proceeding with functions below
        nextQualityLevel = self.q.select_action(self.environmentState)
        msg.add_quality_id(self.bitrates[nextQualityLevel])
        self.environmentState.qualityLevel = nextQualityLevel
        self.send_down(msg)


    def handle_segment_size_response(self, msg):
        self.send_up(msg)


    def initialize(self):
        pass


    def finalization(self):
        pass
