class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        """
        doutは損失関数の微分からこの重みまで存在する関数の微分の積
        """
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy