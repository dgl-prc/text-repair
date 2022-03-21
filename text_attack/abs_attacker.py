from abc import ABCMeta,abstractmethod

class TextAttacker(metaclass=ABCMeta):

    @abstractmethod
    def paraphrase_text(self,input_text,*args):
        pass

    @abstractmethod
    def attack(self,input_text,kwargs):
        pass
