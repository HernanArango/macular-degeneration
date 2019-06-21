import RRO
import numpy as np
class RRO_thresholding:

    def __init__(self):
        pass


    #eight scenarios
    def scenarios(self,img,i,j):
        U = [0,0,0,0,0,0,0,0]
        RRO_pixel = RRO.RRO()

        #scenario 1
        X = [
                img[i-2][j-2],img[i-1][j-2],img[i][j-2],img[i+1][j-2],img[i+2][j-2],
                              img[i-1][j-1],img[i][j-1],img[i+1][j-1],img[i+2][j-1],
                                                        img[i+1][j],  img[i+2][j],
                                                                      img[i+2][j+1]
            ]

        Y = [
            img[i-2][j-1],
            img[i-2][j],img[i-1][j],
            img[i-2][j+1],img[i-1][j+1],img[i][j+1],img[i+1][j+1],
            img[i-2][j+2],img[i-1][j+2],img[i][j+2],img[i+1][j+2],img[i+2][j+2]

        ]

        U1 = RRO_pixel.calculate(X,Y)
        self.is_edge(U1)

        #scenario 2
        X = [
                img[i-2][j],img[i-1][j],
                img[i-2][j+1],img[i-1][j+1],img[i][j+1],img[i+1][j+1],img[i+2][j+1],
                img[i-2][j+2],img[i-1][j+2],img[i][j+2],img[i+1][j+2],img[i+2][j+2]
            ]

        Y = [
                img[i-2][j-2],img[i-1][j-2],img[i][j-2],img[i+1][j-2],img[i+2][j-2],
                img[i-2][j-1],img[i-1][j-1],img[i][j-1],img[i+1][j-1],img[i+2][j-1],
                                                            img[i+1][j],img[i+2][j]
            ]

        U2 = RRO_pixel.calculate(X,Y)
        self.is_edge(U2)

        #scenario 3
        X = [
                img[i-2][j-2],img[i-1][j-2],img[i][j-2],
                img[i-2][j-1],img[i-1][j-1],img[i][j-1],
                img[i-2][j],img[i-1][j],
                img[i-2][j+1],img[i-1][j+1],
                img[i-2][j+2],img[i-1][j+2],

            ]

        Y = [
                            img[i+1][j-2],img[i+2][j-2],
                            img[i+1][j-1],img[i+2][j-1],
                            img[i+1][j],  img[i+2][j],
                img[i][j+1],img[i+1][j+1],img[i+2][j+1],
                img[i][j+2],img[i+1][j+2],img[i+2][j+1]


            ]

        U3 = RRO_pixel.calculate(X,Y)
        self.is_edge(U3)

        #scenario 4
        X = [
                img[i-2][j-2],img[i-1][j-2],
                img[i-2][j-1],img[i-1][j-1],
                img[i-2][j],img[i-1][j],
                img[i-2][j+1],img[i-1][j+1],img[i][j+1],
                img[i-2][j+2],img[i-1][j+2],img[i][j+2]

            ]

        Y = [
                img[i][j-2],img[i+1][j-2],img[i+2][j-2],
                img[i][j-1],img[i+1][j-1],img[i+2][j-1],
                            img[i+1][j],  img[i+2][j],
                            img[i+1][j+1],  img[i+2][j+1],
                            img[i+1][j+2],  img[i+2][j+2]



            ]

        U4 = RRO_pixel.calculate(X,Y)
        self.is_edge(U4)

        #scenario 5
        X = [
                img[i-2][j-2],img[i-1][j-2],img[i][j-2],img[i+1][j-2],img[i+2][j-2],
                img[i-2][j-1], img[i-1][j-1],img[i][j-1],img[i+1][j-1],img[i+2][j-1],
                img[i-2][j],                                             img[i+2][j]
            ]

        Y = [
                              img[i-1][j],              img[i+1][j],
                img[i-2][j+1],img[i-1][j+1],img[i][j+1],img[i+1][j+1],img[i+2][j+1],
                img[i-2][j+2],img[i-1][j+2],img[i][j+2],img[i+1][j+2],img[i+2][j+2]



            ]

        U5 = RRO_pixel.calculate(X,Y)
        self.is_edge(U5)

        #scenario 6
        X = [
                img[i-2][j-2],img[i-1][j-2],
                img[i-2][j-1],img[i-1][j-1],img[i][j-1],
                img[i-2][j],img[i-1][j],
                img[i-2][j+1],img[i-1][j+2],img[i][j+2],
                img[i-2][j+1],img[i-1][j+2]
            ]

        Y = [
                              img[i-1][j],              img[i+1][j],
                img[i-2][j+1],img[i-1][j+1],img[i][j+1],img[i+1][j+1],img[i+2][j+1],
                img[i-2][j+2],img[i-1][j+2],img[i][j+2],img[i+1][j+2],img[i+2][j+2]



            ]

        U6 = RRO_pixel.calculate(X,Y)
        self.is_edge(U6)

        #scenario 7
        X = [
                img[i-2][j-2],img[i-1][j-2],img[i][j-2],img[i+1][j-2],img[i+2][j-2],
                img[i-2][j-1],img[i-1][j-1],img[i][j-1],img[i+1][j-1],img[i+2][j-1],
                img[i-1][j],img[i-2][j]
            ]

        Y = [

                                                            img[i+1][j],img[i+2][j],
                img[i-2][j+1],img[i-1][j+1],img[i][j+1],img[i+1][j+1],img[i+2][j+1],
                img[i-2][j+2],img[i-1][j+2],img[i][j+2],img[i+1][j+2],img[i+2][j+2]
            ]

        U7 = RRO_pixel.calculate(X,Y)
        self.is_edge(U7)

        #scenario 8
        X = [
                img[i-2][j-2],img[i-1][j-2],img[i][j-2],img[i+1][j-2],img[i+2][j-2],
                img[i-2][j-1],img[i-1][j-1],img[i][j-1],img[i+1][j-1],
                img[i-2][j],img[i-1][j],
                img[i-2][j+1]
            ]

        Y = [                                                         img[i+2][j-1],
                                                            img[i+1][j],img[i+2][j],
                              img[i-1][j+1],img[i][j+1],img[i+1][j+1],img[i+2][j+1],
                img[i-2][j+2],img[i-1][j+2],img[i][j+2],img[i+1][j+2],img[i+2][j+2]

            ]

        U8 = RRO_pixel.calculate(X,Y)
        self.is_edge(U8)

        return False

    def is_edge(self,Umax):
        if Umax >= 0.15:
            return True

    def calculate(self,img):

        rows, cols = img.shape
        
        #start in 3 for evite problem with the limits
        for i in range(3,rows-3):
            print("fila",i)
            for j in range(3,cols-3):
                if img[i][j] != 0:
                    if self.scenarios(img,i,j):
                        img[i][j] = 1
                    else:
                        img[i][j] = 0


        return img
