"""
@author: Wagner C.C. Batalha (wagnerweb2010@gmail.com) 2023

@description: Q Learning based MPEG dash quality selection Client. 

"""
from r2a.ir2a import IR2A

class R2AQ(IR2A):
    def __init__(self, qualityLevels, maxBufferSize):
        self.qualityLevels = qualityLevels
        self.maxBufferSize = maxBufferSize

        # Chosen before running algorithm
        self.rewardWeight   = { "quality":0.1,       "oscillation":0.2, 
                                "bufferFilling":0.2, "bufferChange":0.2 }
        self.lastBufferHealth = 0


    def rewardQuality(self, qualityLevel:int) -> float:
        """
        Equation (2)
        """
        r = ((qualityLevel - 1)
            /(self.qualityLevels - 1)) * 2 - 1
        return r
    

    def rewardOscillation(self) -> float:
        """
        Equation (3)
        """
        # TODO: write Roscillation function
        pass


    def rewardBufferFilling(self, bufferHealth:float) -> float:
        """
        Equation (4)
        """
        if (bufferHealth <= ( 0.1 * self.maxBufferSize )):
            r = -1.0
        else: # bufferHealth > 0.1 * maxBufferSize
            r = ((2 * bufferHealth )
                /((1 - 0.1) * self.maxBufferSize))
            r = r - ((1 + 0.1) 
                    /(1 - 0.1))
        return r
    
    
    def rewardBufferChange(self, bufferHealth:float) -> float:
        """
        Equation (5)
        """
        if bufferHealth <= self.lastBufferHealth:
            r = ((bufferHealth - self.lastBufferHealth)
                /(self.lastBufferHealth))
        else: # bufferHealth > self.lastBufferHealth
            r = ((bufferHealth - self.lastBufferHealth)
                /(bufferHealth - (self.lastBufferHealth / 2)))
        return r


    def getReward(self, qualityLevel:int, bufferHealth:float) -> float:
        """
        Get reward given the parameters. self.rewardWeight Dictionary values can be changed to
        tune this method.

        Parameters
        ----------
        qualityLevel: int 
            Integer describing the level from N levels of quality available(self.qualityLevels).
            Doesn't correspond to actual bitrates, rather larger values correspond with better 
            video qualities.

        bufferHealth: float
            Value in seconds of remaining video that is already stored on client waiting for be 
            played.
        """
        r  = self.rewardWeight["quality"]       * self.rewardQuality(qualityLevel)
        r += self.rewardWeight["oscillation"]   * self.rewardOscillation()
        r += self.rewardWeight["bufferFilling"] * self.rewardBufferFilling(bufferHealth)
        r += self.rewardWeight["bufferChange"]  * self.rewardBufferChange(bufferHealth)
        self.lastBufferHealth = bufferHealth
