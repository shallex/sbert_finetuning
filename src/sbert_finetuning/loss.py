from enum import Enum, auto


class LossType(Enum):
    BatchAllTripletLoss = auto()
    BatchHardTripletLoss = auto()
    BatchHardSoftMarginTripletLoss = auto()
    BatchSemiHardTripletLoss = auto()

    def __str__(self):
        return self.name
