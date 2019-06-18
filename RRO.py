import math

class RRO:
    uyxi = []
    uxyi = []
    vx = 0
    vy = 0
    def calculate(self,X,Y):
        #X = sorted(X)
        #Y = sorted(Y)
        m = len(X)
        n = len(Y)
        #U(Y,Xi)

        for i in range(0,m):
            self.uyxi.append(self.lower_values(X[i],Y))

        #U(X,Yi)

        for i in range(0,n):
            self.uxyi.append(self.lower_values(Y[i],X))

        #U(Y,X)
        uyx = self.mean(self.uyxi)
        #print("uyx",uyx)
        #U(X,Y)
        uxy = self.mean(self.uxyi)

        #print(uyx,uxy)


        #index

        for i in range(0,m):
            self.vx += pow(self.uyxi[i] - uyx , 2)

        #index
        for i in range(0,n):
            self.vy += pow(self.uxyi[i] - uxy , 2)

        U = ((m * uyx) - (n * uxy))/(2 * math.sqrt(self.vx + self.vy + uyx * uxy))

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
