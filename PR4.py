#Implement Gradient Descent Algorithm to find the local minima of a function. 
#For example, find the local minima of the function y=(x+3)^2 starting from the point x=2.

def logic(cur_x, rate, precision, previous_step_size, maxiCnt, iCnt, data):
    while previous_step_size > precision and iCnt < maxiCnt:
        prev_x = cur_x
        cur_x = cur_x - rate * data(prev_x)
        previous_step_size = abs(cur_x - prev_x)
        iCnt = iCnt+1
        print("Iteration", iCnt,"\nX value is ",cur_x)
    
    print("Local Minimum occurs at", cur_x)

def main():
    cur_x = 2
    rate = 0.01
    precision = 0.001
    previous_step_size = 1
    maxiCnt = 1000
    iCnt = 0
    data = lambda x: 2*(x+3)

    logic(cur_x, rate, precision, previous_step_size, maxiCnt, iCnt, data)

if __name__ =="__main__":
    main()