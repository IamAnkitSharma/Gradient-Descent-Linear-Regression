import numpy as np

def gradientDescent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    learning_rate = 0.001
    n = len(x)
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        # print(y_predicted)
        md = (-2 / n) * sum(x * (y - (y_predicted)))
        bd = (-2 / n) * sum(y - (y_predicted))
        # print(md,bd)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        cost = 1 / n * sum([val ** 2 for val in (y - y_predicted)])
        #print("m => {} b => {} iteration => {} cost => {} ".format(m_curr,b_curr,i,cost))

    return m_curr, b_curr


x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

m, b = gradientDescent(x, y)
print(m * x + b)
#Output => array([2.01264586, 4.00780129, 6.00295672, 7.99811216, 9.99326759])
