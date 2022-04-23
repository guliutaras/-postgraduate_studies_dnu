import math

class ActivationFunction:
    # __singleton = None
    # def __new__(cls):
    #     cls.__singleton = cls
    def __init__(self, types='Sigmoid'):
        # if self.__singleton is None:
        #     self.__singleton = self
        # cls = self.__singleton
        self.func = self.sine
        self.dfunc = self.dsine

        if types == 'Sigmoid':
            self.func = self.sigmoid
            self.dfunc = self.dsigmoid
        if types == 'Sine':
            self.func = self.sine
            self.dfunc = self.dsine
        if types == 'Gausse':
            self.func = self.gause
            self.dfunc = self.dgause

    def run(self, x):
        return self.func(x)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def dsigmoid(self, y):
        return y * (1 - y)
    
    def sine(self, x):
        return math.sin(x)
    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def dsine(self, y):
        return math.cos(y)

    def gause(self, x):
        return math.exp(-x**2)
    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def dgause(self, y):
        return -2*y*math.exp(-y**2)


if __name__ == '__main__':
    myfunc = ActivationFunction('Sine')
