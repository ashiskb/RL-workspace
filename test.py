import numpy as np

def a_func():
        left, down, right, up = (0,1,2,3)
        T = np.zeros([16,4],dtype=int)
        #non-goal and non-hole transitions
        T[0,left], T[0,right], T[0,up], T[0, down] = (0,1,0,4)
        T[1,left], T[1,right], T[1,up], T[1, down] = (0,2,1,5)
        T[2,left], T[2,right], T[2,up], T[2, down] = (1,3,2,6)
        T[3,left], T[3,right], T[3,up], T[3, down] = (2,3,3,7)
        T[4,left], T[4,right], T[4,up], T[4, down] = (4,5,0,8)
        T[5,left], T[5,right], T[5,up], T[5, down] = (4,6,1,9)
        T[7,left], T[7,right], T[7,up], T[7, down] = (6,7,3,11)
        T[8,left], T[8,right], T[8,up], T[8, down] = (8,9,4,12)
        T[11,left], T[11,right], T[11,up], T[11, down] = (10,11,7,15)
        T[12,left], T[12,right], T[12,up], T[12, down] = (12,13,8,12)
        T[13,left], T[13,right], T[13,up], T[13, down] = (12,14,9,13)
        T[14,left], T[14,right], T[14,up], T[14, down] = (13,15,10,14)

        # terminal Goal state
        T[10,left], T[10,right], T[10,up], T[10,down] = (10,10,10,10)

        # terminal Hole state
        T[6,left], T[6,right], T[6,up], T[6, down] = (6,6,6,6)
        T[9,left], T[9,right], T[9,up], T[9, down] = (9,9,9,9)
        T[15,left], T[15,right], T[15,up], T[15, down] = (15,15,15,15)

        #print(T)
        return T

if __name__ == '__main__':
    T = a_func()
    state = 0
    for r in np.arange(T.shape[0]):
        valid_actions = np.where(T[r,:]!=r)
        print(valid_actions[0])
        print('state = {}, available moves: {}'.format(r,valid_actions))
        break