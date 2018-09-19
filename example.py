import numpy as np
from model import LstmNetwork

EPOCH = 100
LEARNING_RATE = 0.5

x_list = [np.array([[1], [2]]), np.array([[0.5], [3]])]
y_list = [0.5, 1.25]

lstm = LstmNetwork(x_dimension = 2, memory_dimension = 2)

# lstm.fit(x_list, y_list, learning_rate = LEARNING_RATE)

# module = lstm.lstm_unit_list[1]

# print("final gate a: ", module.gate_a)
# print("final gate i: ", module.gate_i)
# print("final gate f: ", module.gate_f)
# print("final gate o: ", module.gate_o)
# print("final state: ", module.state)
# print("final output: ", module.o)

# print("new Wa: %s new Ua: %s new ba: %s" % (module.parameter.Wa, module.parameter.Ua, module.parameter.ba))
# print("new Wi: %s new Ui: %s new bi: %s" % (module.parameter.Wi, module.parameter.Ui, module.parameter.bi))
# print("new Wf: %s new Uf: %s new bf: %s" % (module.parameter.Wf, module.parameter.Uf, module.parameter.bf))
# print("new Wo: %s new Uo: %s new bo: %s" % (module.parameter.Wo, module.parameter.Uo, module.parameter.bo))

for epoch in range(EPOCH):
    lstm.fit(x_list, y_list, learning_rate = LEARNING_RATE)
    loss = 0

    for index in range(len(lstm.lstm_unit_list)):
        loss += (lstm.lstm_unit_list[index].o[0] - y_list[index]) ** 2

    print("output 1: %s, output 2: %s, loss: %s" % (lstm.lstm_unit_list[0].o[0], lstm.lstm_unit_list[1].o[0], loss))
