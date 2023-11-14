# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm

## PROBLEM STATEMENT
The FrozenLake environment within OpenAI Gym presents a challenging gridworld problem for reinforcement learning agents. In this scenario, an agent is placed within a 2D grid environment. The objective is to guide the agent from its initial position to a designated goal state while avoiding treacherous hazards. Notably, the agent cannot exit the boundaries of this enclosed gridworld as it is surrounded by a fence.

## MONTE CARLO CONTROL ALGORITHM
Step 1:
Initialize Q-values, state-value function, and the policy.

Step 2:
Interact with the environment to collect episodes using the current policy.

Step 3:
For each time step within episodes, calculate returns (cumulative rewards) and update Q-values.

Step 4:
Update the policy based on the improved Q-values.

Step 5:
Repeat steps 2-4 for a specified number of episodes or until convergence.

Step 6:
Return the optimal Q-values, state-value function, and policy.



## MONTE CARLO CONTROL FUNCTION
```python
import numpy as np
from tqdm import tqdm

def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n

    disc = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    def decay_schedule(init_value, min_value, decay_ratio, n):
        return np.maximum(min_value, init_value * (decay_ratio ** np.arange(n)))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    def select_action(state, Q, epsilon):
        return np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(nA)

    for e in tqdm(range(n_episodes), leave=False):
        traj = gen_traj(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=np.bool)

        for t, (state, action, reward, _, _) in enumerate(traj):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            n_steps = len(traj[t:])
            G = np.sum(disc[:n_steps] * traj[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        Q_track[e] = Q

    V = np.max(Q, axis=1)
    pi = {s: np.argmax(Q[s]) for s in range(nS)}

    return Q, V, pi
```

## OUTPUT:
![image](https://github.com/Fawziya20/monte-carlo-control/assets/75235022/11656fc9-83d3-4f76-ad9e-80689c387f50)

![image](https://github.com/Fawziya20/monte-carlo-control/assets/75235022/48a2a704-f3ff-4b0b-a0de-110754e9c7f1)




## RESULT:
Monte Carlo Control successfully learned an optimal policy for the specified environment.
