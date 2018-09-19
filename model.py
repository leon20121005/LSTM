import numpy as np

def get_random_array(a, b, *dimensions):
    return np.random.rand(*dimensions) * (b - a) + a

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LstmParameter:
    def __init__(self, x_dimension, memory_dimension):
        self.x_dimension = x_dimension
        self.memory_dimension = memory_dimension

        self.Wa = get_random_array(-1, 1, memory_dimension, x_dimension)        # weight of input activation
        self.Wi = get_random_array(-1, 1, memory_dimension, x_dimension)        # weight of input gate
        self.Wf = get_random_array(-1, 1, memory_dimension, x_dimension)        # weight of forget gate
        self.Wo = get_random_array(-1, 1, memory_dimension, x_dimension)        # weight of output gate
        self.Ua = get_random_array(-1, 1, memory_dimension, memory_dimension)   # recurrent connection weight of input activation
        self.Ui = get_random_array(-1, 1, memory_dimension, memory_dimension)   # recurrent connection weight of input gate
        self.Uf = get_random_array(-1, 1, memory_dimension, memory_dimension)   # recurrent connection weight of forget gate
        self.Uo = get_random_array(-1, 1, memory_dimension, memory_dimension)   # recurrent connection weight of output gate
        self.ba = get_random_array(-1, 1, memory_dimension, 1)                  # bias of input activation
        self.bi = get_random_array(-1, 1, memory_dimension, 1)                  # bias of input input gate
        self.bf = get_random_array(-1, 1, memory_dimension, 1)                  # bias of input forget gate
        self.bo = get_random_array(-1, 1, memory_dimension, 1)                  # bias of input output gate

        self.initialize_delta_parameter()

    def apply_difference(self, learning_rate):
        self.Wa -= learning_rate * self.delta_Wa
        self.Wi -= learning_rate * self.delta_Wi
        self.Wf -= learning_rate * self.delta_Wf
        self.Wo -= learning_rate * self.delta_Wo
        self.Ua -= learning_rate * self.delta_Ua
        self.Ui -= learning_rate * self.delta_Ui
        self.Uf -= learning_rate * self.delta_Uf
        self.Uo -= learning_rate * self.delta_Uo
        self.ba -= learning_rate * self.delta_ba
        self.bi -= learning_rate * self.delta_bi
        self.bf -= learning_rate * self.delta_bf
        self.bo -= learning_rate * self.delta_bo

        self.initialize_delta_parameter()

    def initialize_delta_parameter(self):
        self.delta_Wa = np.zeros_like(self.Wa)
        self.delta_Wi = np.zeros_like(self.Wi)
        self.delta_Wf = np.zeros_like(self.Wf)
        self.delta_Wo = np.zeros_like(self.Wo)
        self.delta_Ua = np.zeros_like(self.Ua)
        self.delta_Ui = np.zeros_like(self.Ui)
        self.delta_Uf = np.zeros_like(self.Uf)
        self.delta_Uo = np.zeros_like(self.Uo)
        self.delta_ba = np.zeros_like(self.ba)
        self.delta_bi = np.zeros_like(self.bi)
        self.delta_bf = np.zeros_like(self.bf)
        self.delta_bo = np.zeros_like(self.bo)

class LstmUnit:
    def __init__(self, lstm_parameter):
        self.parameter = lstm_parameter

    def forward(self, x, previous_out, previous_state):
        # compute gates
        self.gate_a = np.tanh(np.dot(self.parameter.Wa, x) + np.dot(self.parameter.Ua, previous_out) + self.parameter.ba)
        self.gate_i = sigmoid(np.dot(self.parameter.Wi, x) + np.dot(self.parameter.Ui, previous_out) + self.parameter.bi)
        self.gate_f = sigmoid(np.dot(self.parameter.Wf, x) + np.dot(self.parameter.Uf, previous_out) + self.parameter.bf)
        self.gate_o = sigmoid(np.dot(self.parameter.Wo, x) + np.dot(self.parameter.Uo, previous_out) + self.parameter.bo)

        # compute memory cell and output
        self.state = self.gate_a * self.gate_i + self.gate_f * previous_state
        self.o = self.gate_o * np.tanh(self.state)

    def backward(self, y, next_lstm_unit, previous_lstm_unit):
        DELTA_out = next_lstm_unit.previous_DELTA_out if next_lstm_unit is not None else 0
        next_delta_state = next_lstm_unit.delta_state if next_lstm_unit is not None else 0
        next_gate_f = next_lstm_unit.gate_f if next_lstm_unit is not None else 0
        previous_state = previous_lstm_unit.state if previous_lstm_unit is not None else 0

        # print("backward() - self.o")
        # print(self.o)
        # print("backward() - y")
        # print(y)
        # print("backward() - DELTA_out")
        # print(DELTA_out)

        self.delta_out = self.o - y + DELTA_out
        self.delta_state = self.delta_out * self.gate_o * (1 - np.tanh(self.state) ** 2) + next_delta_state * next_gate_f

        self.delta_gate_a = self.delta_state * self.gate_i * (1 - self.gate_a ** 2)
        self.delta_gate_i = self.delta_state * self.gate_a * self.gate_i * (1 - self.gate_i)
        self.delta_gate_f = self.delta_state * previous_state * self.gate_f * (1 - self.gate_f)
        self.delta_gate_o = self.delta_out * np.tanh(self.state) * self.gate_o * (1 - self.gate_o)

        # print("backward() - self.delta_gate_a")
        # print(self.delta_gate_a)

        U_transpose = np.transpose(np.vstack((self.parameter.Ua, self.parameter.Ui, self.parameter.Uf, self.parameter.Uo)))
        delta_gates = np.vstack((self.delta_gate_a, self.delta_gate_i, self.delta_gate_f, self.delta_gate_o))

        # print("backward() - U_transpose")
        # print(U_transpose)
        # print("backward() - delta_gates")
        # print(delta_gates)

        self.previous_DELTA_out = np.dot(U_transpose, delta_gates)

class LstmNetwork:
    def __init__(self, x_dimension, memory_dimension):
        self.parameter = LstmParameter(x_dimension, memory_dimension)

    def fit(self, x_list, y_list, learning_rate):
        self.lstm_unit_list = [LstmUnit(self.parameter) for x in x_list]

        # forward
        for index in range(len(self.lstm_unit_list)):
            if index is 0:
                self.lstm_unit_list[index].forward(x_list[index], np.zeros_like(self.parameter.ba), np.zeros_like(self.parameter.ba))
            else:
                previous_out = self.lstm_unit_list[index - 1].o
                previous_state = self.lstm_unit_list[index - 1].state
                self.lstm_unit_list[index].forward(x_list[index], previous_out, previous_state)

        # backward
        for index in reversed(range(len(self.lstm_unit_list))):
            if index is len(self.lstm_unit_list) - 1:
                self.lstm_unit_list[index].backward(y_list[index], None, self.lstm_unit_list[index - 1])
            elif index is 0:
                self.lstm_unit_list[index].backward(y_list[index], self.lstm_unit_list[index + 1], None)
            else:
                self.lstm_unit_list[index].backward(y_list[index], self.lstm_unit_list[index + 1], self.lstm_unit_list[index - 1])

        self.compute_delta_parameter(x_list)
        self.parameter.apply_difference(learning_rate)

    def compute_delta_parameter(self, x_list):
        for index in range(len(self.lstm_unit_list)):

            # print(np.outer(self.lstm_unit_list[index].delta_gate_a, x_list[index]))

            self.parameter.delta_Wa += np.outer(self.lstm_unit_list[index].delta_gate_a, x_list[index])
            self.parameter.delta_Wi += np.outer(self.lstm_unit_list[index].delta_gate_i, x_list[index])
            self.parameter.delta_Wf += np.outer(self.lstm_unit_list[index].delta_gate_f, x_list[index])
            self.parameter.delta_Wo += np.outer(self.lstm_unit_list[index].delta_gate_o, x_list[index])
            self.parameter.delta_ba += self.lstm_unit_list[index].delta_gate_a
            self.parameter.delta_bi += self.lstm_unit_list[index].delta_gate_i
            self.parameter.delta_bf += self.lstm_unit_list[index].delta_gate_f
            self.parameter.delta_bo += self.lstm_unit_list[index].delta_gate_o

        for index in range(len(self.lstm_unit_list) - 1):
            self.parameter.delta_Ua += self.lstm_unit_list[index + 1].delta_gate_a * self.lstm_unit_list[index].o
            self.parameter.delta_Ui += self.lstm_unit_list[index + 1].delta_gate_i * self.lstm_unit_list[index].o
            self.parameter.delta_Uf += self.lstm_unit_list[index + 1].delta_gate_f * self.lstm_unit_list[index].o
            self.parameter.delta_Uo += self.lstm_unit_list[index + 1].delta_gate_o * self.lstm_unit_list[index].o
