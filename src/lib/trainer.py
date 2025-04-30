from src.lib.backend import backend, HasBackend

from src.lib.optimizers import SGD, Adam

class Trainer(HasBackend):
    pass

class trainer(Trainer):
    def __init__(self, model, inputs, targets, train_test_split = 0.8, random_state = 42, shuffle = True, custom_train_test_set = None, float_type = None):
        self.model = model
        
        inputs = self.asarray(inputs)
        targets = self.asarray(targets)
        
        self.inputs = inputs
        self.targets = targets
        
        float_type = float_type if float_type is not None else self.xp.float32
        
        nb_of_inputs = inputs.shape[0]
        nb_of_targets = targets.shape[0]
        
        self.nb_of_inputs = nb_of_inputs
        self.nb_of_targets = nb_of_targets
        
        self.shuffle = shuffle
        self.train_test_split = train_test_split
        self.random_state = random_state
        
        assert(nb_of_inputs == nb_of_targets)
        
        if shuffle:
            indices = self.xp.arange(len(inputs))
            indices = self.xp.random.permutation(indices)
            inputs = inputs[indices]
            targets = targets[indices]
        
        self.train_size = int(train_test_split*self.nb_of_inputs)
        self.test_size = self.nb_of_inputs - self.train_size
        
        if custom_train_test_set is None:
            self.train_X = inputs[:self.train_size]
            self.test_X = inputs[self.train_size:]
            self.train_Y = targets[:self.train_size]
            self.test_Y = targets[self.train_size:]
        else:
            self.train_X, self.train_Y, self.test_X, self.test_Y = custom_train_test_set
            self.train_X = self.asarray(self.train_X)
            self.train_Y = self.asarray(self.train_Y)
            self.test_X = self.asarray(self.test_X)
            self.test_Y = self.asarray(self.test_Y)
            
        self.train_X = self.train_X.astype(float_type)
        self.train_Y = self.train_Y.astype(float_type)
        self.test_X = self.test_X.astype(float_type)
        self.test_Y = self.test_Y.astype(float_type)

        
    def train(self, nb_epochs = 100, batch_size = 64, optimizer=SGD(learning_rate=0.01), test_accuracy = True):
        print(0)
        self.model.intialise_targets(self.test_Y)
        self.model.forward(self.test_X, training = False)
        
        print(self.model.loss)
        print(self.model.accuracy)
        for epoch in range(1, nb_epochs+1):
            print(epoch)
            X_batches, Y_batches = self.batch_train_data(batch_size=batch_size) 
            
            for x, y in zip(X_batches, Y_batches):
                self.model.intialise_targets(y)
                self.model.forward(x)
                self.model.backward()
                self.model.update_params(optimizer = optimizer)
            
            if test_accuracy:
                ###test####
                self.model.intialise_targets(self.test_Y)
                self.model.forward(self.test_X, training=False)
                
                print(self.model.loss) 
                print(self.model.accuracy)     
                
        print("final")      
        self.model.intialise_targets(self.test_Y)
        self.model.forward(self.test_X, training=False)
        
        print(self.model.loss) 
        print(self.model.accuracy) 
        
        return self.model.loss, self.model.accuracy
            
    def batch_train_data(self, batch_size = 64):
        batch_train_data_X = self.train_X
        batch_train_data_y = self.train_Y
        
        indices = self.xp.arange(len(batch_train_data_X))
        indices = self.xp.random.permutation(indices)
        batch_train_data_X = batch_train_data_X[indices]
        batch_train_data_y = batch_train_data_y[indices]

        n_splits = int(self.xp.ceil(len(batch_train_data_X) / batch_size))
        X_batches = self.xp.array_split(batch_train_data_X, n_splits)
        y_batches = self.xp.array_split(batch_train_data_y, n_splits)
        
        return X_batches, y_batches

        
        
        
        
        
        
        
        
    