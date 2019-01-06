# MountainCar_DQN_RND
### Playing Mountain-Car without reward engineering, by combining DQN and Random Network Distillation (RND).

This project contains a simple PyTorch implementation of DQN [1] for playing Mountain-Car. Mountain-Car is a classic control game in which a car must swing back and forth in order to reach the flag on top of the mountain. Unlike other classic problems like CartPole, the reward in Mountain-Car is sparse as positive feedback is only given upon reaching the flag, which is unlikely with random actions.
Usually this is circumvented buy engineering the reward signal in order to get a smoother learning process, by using the velocity or piostion increments as additional rewards, but for general problems, this may not be easy to do, as it requires some knowledge about how to solve the problem.

This project opts for a methodical way of exploration, that is not domain specific, by using Exploration By Random Network Distillation [2], a method developed by OpenAI researchers for hard exploration games like Montezuma's Revenge. This method trains a neural network to try and predict the outputs of a different random neural network, and the prediction error is added to the true reward signal. By updating the network, what we gain is a measure of "familiarity" with game states, encouraging our RL algorithm to explore those states that we are not familiar with and eventually discovering positive feedbacks.

This plot shows accumulated returns against episode:

![Alt text](https://user-images.githubusercontent.com/46422351/50738799-a29c2780-11e0-11e9-82f4-e1ac46ee1a3e.png)

And this one shows the augmented returns, after adding the exploration bonus given by RND:

![Alt text](https://user-images.githubusercontent.com/46422351/50738963-fe1ae500-11e1-11e9-9cf1-084f067ef79f.png)


1. Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
2. Burda, Yuri, et al. "Exploration by random network distillation." arXiv preprint arXiv:1810.12894 (2018).

