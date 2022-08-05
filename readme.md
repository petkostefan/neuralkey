# Neural key exchange protocol

Traditional key exchange protocols are based on number theory, the most widely used among them being the Diffie-Hellman protocol. On the other hand, a new key exchange method based on artificial neural networks is emerging. This method uses the synchronization feature of the two neural networks to create a shared secret.

In the neural key exchange protocol, each party creates a neural network with one hidden layer, called a Tree parity machine. They learn from each other's outputs until fully synchronized, after which their weights can be used as encryption keys.

The most important thing about this protocol is that it ensures that the key cannot be deduced, even though an attacker has all the parameters of the neural network and eavesdrops on their communication.

![Tree parity machine](https://upload.wikimedia.org/wikipedia/commons/4/42/Tree_Parity_Machine.jpg)

## Implementation
The implementation consists of 3 python files: **learning_rules**, in which the learning rules are defined, **tree_parity_machine** in which the class that defines the neural network and its behaviour is located and **main** in which we create 3 machines for synchronizing trees and simulate the pairing of Alice's and Bob's machine while Eve eavesdrops their communication and tries to synchronize her machine.

To run the simulation, run the **main** file. In the main file, the hyperparameters of the network can be changed: N (number of inputs - default: 10), K (number of hidden neurons - default: 100), L (synaptic depth - default: 10) and learning rule (one of 3 learning rules: Hebbian, Anti-Hebbian and Random Walk - default: Hebbian) After syncing, sucess message is shown along with time taken and a graph representing the syncing process. 