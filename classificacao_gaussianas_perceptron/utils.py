import numpy as np
from copy import copy

weight_neuron1 = []
weight_neuron2 = []

def randomFullFillMatrix(matrixSize, lower_boundary, upper_boundary):
    """Função usada para criar e preencher aleatoriamente uma Matriz

    Args:
        matrixSize (int): Tamanho da matriz a ser criada
        lower_boundary (float): Valor mínimo utilizado no preenchimento dos dados
        upper_boundary (float): Valor máximo utilizado no preenchimento dos dados

    Returns:
        list of lists: Matriz (matrixSize x MatrixSize) preenchida aleatoriamente com valores entre
                lower_boundary e upper_boundary
    """
    column = []
    matrix = []

    for _ in range(matrixSize[0]):
        for _ in range(matrixSize[1]):
            column.append(np.random.uniform(lower_boundary, upper_boundary))

        #Realizo o append de uma lista dentro da outra ao invés de somente os valores corridos
        #para facilitar a visualização e manipulação
        
        matrix.append(column)
        column = []

    return matrix

def BinaryEncoder(y_vector):
    """Função que codifica em binário o vator de alvos

    Args:
        y_vector (list): Vetor de alvos

    Returns:
        np.array: Vetor de alvos codificado em binário
    """

    encoded_vector = []

    for i in range(len(y_vector)):
        if y_vector[i] == 1:
            encoded_vector.append([0,0])
        elif y_vector[i] == 2:
            encoded_vector.append([1,0])
        elif y_vector[i] == 3:
            encoded_vector.append([1,1])

    return np.array(encoded_vector)

def to_numpy_vector(data, target):
    """Função que transforma os Pandas Dataframes em numpy arrays

    Args:
        data (pd.Dataframe): Dataframe com os dados do nosso problema
        target (pd.Dataframe): Dataframe com os alvos do nosso problema

    Returns:
        data, target: Numpy arrays
    """
    
    data = data.to_numpy()
    target = target.to_numpy() 

    return data, target

def heaviside(value):
    """Função degrau usada na ativação do Perceptron

    Args:
        value (float): Valor calculado pelo Perceptron

    Returns:
        boolean: Resultado da função degrau
    """

    if value >= 0.0:
        return 1 
    else:
        return 0

def predict(row, weights):
    """Função que calcula a predição do Perceptron 

    Args:
        row (_type_): _description_
        weights (_type_): _description_

    Returns:
        boolean: Retorna o resultado da função degrau, calculada para a predição do Perceptron
    """

    activation = 0

    for i in range(len(row)):
        activation += weights[i] * row[i]

    return heaviside(activation)

# Estimando os pesos do Perceptron usando stochastic gradient descent
def train_weights(W, train, y_train_encoded, l_rate, n_epoch):

    O = np.zeros((train.shape[0], 2))

    global weight_neuron1
    global weight_neuron2

    #Salvando os pesos iniciais
    weight_neuron1.append(copy(W[:,0]))
    weight_neuron2.append(copy(W[:,1]))

    for epoch in range(n_epoch):
        
        num_errors = 0
        index = 0

        for row in train:

            #Neurônio 1
            prediction_1 = predict(row, W[:,0])
            delta1 = prediction_1 - y_train_encoded[index][0]

            O[index][0] = delta1

            #Neurônio 2
            prediction_2 = predict(row, W[:,1])
            delta2 = prediction_2 - y_train_encoded[index][1]

            O[index][1] = delta2

            #Contagem de erros na época
            if (delta1 != 0) or (delta2 != 0):
                num_errors = num_errors + 1

            index += 1
        
        #Atualizando pesos sinápticos
        for i in range(2):
            for j in range(3):
                for k in range(train.shape[0]):
                    W[j][i] = W[j][i] - l_rate * train[k][j] * O[k][i]

        #Salvando pesos sinápticos de cada neurônio
        weight_neuron1.append(copy(W[:,0]))
        weight_neuron2.append(copy(W[:,1]))

        print('>epoch=%d, lrate=%.3f, num_of_samples_wrong=%.3f' % (epoch, l_rate, num_errors))
    
    weight_neuron1 = np.asarray(weight_neuron1, dtype=np.float32)
    weight_neuron2 = np.asarray(weight_neuron2, dtype=np.float32)
    return W