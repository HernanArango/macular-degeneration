import math
import numpy as np

class RRO:

    def calculate(self,X,Y):
        #X = sorted(X)
        #Y = sorted(Y)
        m = len(X)
        n = len(Y)
        #U(Y,Xi)
        #print(m,n)
        uyxi = np.array([])
        uxyi = np.array([])
        vx = 0
        vy = 0
        tmp = []
        for i in range(0,m):
            tmp.append(self.lower_values(X[i],Y))

        uyxi = np.append(uyxi,tmp)
        #U(X,Yi)


        tmp = []
        for i in range(0,n):
            tmp.append(self.lower_values(Y[i],X))

        uxyi = np.append(uxyi,tmp)
        #U(Y,X)
        uyx = self.mean(uyxi)
        #print("uyx",uyx)
        #U(X,Y)
        uxy = self.mean(uxyi)

        #print(uyx,uxy)
        uyx_numpy = np.full((1, m), uyx, dtype=float)
        uxy_numpy = np.full((1, n), uxy, dtype=float)
        #print("m ",m, "uyxi ", uyxi.size)
        #print(uyxi.size,uyx_numpy.size)

        vx = np.sum (np.power(uyxi - uyx_numpy,2))
        vy = np.sum (np.power(uxyi - uxy_numpy,2))
        #index
        """
        for i in range(0,m):
            vx += pow(uyxi[i] - uyx , 2)

        #index
        for i in range(0,n):
            vy += pow(uxyi[i] - uxy , 2)
        """

        U = ((m * uyx) - (n * uxy))/(2 * math.sqrt(vx + vy + uyx * uxy))

        #print("U -> ",U)

        return U

    #U(Y,Xi)
    def lower_values(self,X,Y):
        #print("-------")
        lower_values = 0
        half = 0
        whole = 0
        for Yi in Y:

            if  X >= Yi:
                #print(X,">=", Yi)
                #lower_values += Xi
                if Yi == X:
                    half += 1;

                else:
                    whole += 1;
            else:
                break

        whole += half/2;

        rank = whole

        if(half > 0 and (half % 2) == 1):
            rank += 0.5;
        #print(rank)
        return rank

    #U(Y,X)
    def mean(self,a):
        val = 0
        for x in a:
            val += x

        return val/len(a)

#X = [5.025,6.7,6.725,6.75,7.05,7.25,8.375]
#Y = [4.875,5.125,5.225,5.425,5.55,5.75,5.925,6.125]
#RRO(X,Y)
