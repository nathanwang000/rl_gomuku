# KL-Regularized RL: Policy and Value Are the Same Thing

## Motivation

In standard RL, once you have the value function you can derive the policy by taking an argmax. But the reverse is not true — going from policy back to value is ill-posed because the argmax discards the relative magnitudes of non-optimal actions.

Yet both are useful:
- **Policy head**: enables behavioral cloning, accelerates learning
- **Value head**: crucial for variance reduction in policy gradient
- **Both**: indispensable for MCTS

It feels redundant to have two separate heads with no formal tie between them. The fix is KL-regularized RL, which makes the relationship **invertible**: policy log-ratios and advantages become the same object, just scaled by β.

---

## Step 1: Define the Objective

Standard RL maximizes expected discounted reward. Add a KL penalty against a reference policy at every step:

$$J(\pi) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t \Bigl(r(s_t, a_t) - \beta \log\frac{\pi(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\Bigr)\right]$$

Nothing has been assumed yet — this is just a definition of what we want to maximize. The β > 0 hyperparameter controls how much we penalize deviation from the reference policy.

---

## Step 2: Solve the Bandit Case (One Step, No Future)

Before tackling the full MDP, solve the one-step version: maximize over π(·|s) for a fixed state s, treating r(a) as given.

Write the Lagrangian (with multiplier λ enforcing Σ π(a) = 1):

$$\mathcal{L} = \sum_a \pi(a)\, r(a) - \beta \sum_a \pi(a) \log\frac{\pi(a)}{\pi_{\text{ref}}(a)} - \lambda\!\left(\sum_a \pi(a) - 1\right)$$

Take ∂L/∂π(a) = 0:

$$r(a) - \beta\log\frac{\pi^*(a)}{\pi_{\text{ref}}(a)} - \beta - \lambda = 0$$

Rearrange:

$$\log\frac{\pi^*(a)}{\pi_{\text{ref}}(a)} = \frac{r(a)}{\beta} - 1 - \frac{\lambda}{\beta}$$

$$\pi^*(a) = \pi_{\text{ref}}(a)\exp\!\left(\frac{r(a)}{\beta}\right) \cdot \underbrace{\exp\!\left(-1 - \frac{\lambda}{\beta}\right)}_{\text{same constant for all } a}$$

The constant is determined by normalization — it must equal 1/Z where:

$$Z = \sum_{a'} \pi_{\text{ref}}(a')\exp\!\left(\frac{r(a')}{\beta}\right)$$

So the optimal policy is:

$$\boxed{\pi^*(a) = \frac{\pi_{\text{ref}}(a)\exp(r(a)/\beta)}{Z}}$$

Now rearrange to express r(a) in terms of π*:

$$r(a) = \beta\log\frac{\pi^*(a)}{\pi_{\text{ref}}(a)} + \beta\log Z$$

Define $V^* \triangleq \beta \log Z$. Then:

$$r(a) = V^* + \beta\log\frac{\pi^*(a)}{\pi_{\text{ref}}(a)}$$

**This is the DPO identity.** Reward equals the policy log-ratio times β, up to a constant V* that is the same for all actions at a given state (and thus cancels in pairwise preference comparisons). The relationship between r and π* is now invertible.

---

## Step 3: Lift to the Sequential MDP

Define the KL-regularized value and Q functions:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t \ge 0} \gamma^t \Bigl(r(s_t,a_t) - \beta\log\tfrac{\pi(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\Bigr)\,\Big|\, s_0 = s\right]$$

$$Q^\pi(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s' \sim P(\cdot|s,a)}[V^\pi(s')]$$

**Why does Q not include the KL cost?** Because Q is defined conditional on having already taken action a — there is no more decision at this step, so there is no KL cost to pay here. The future KL costs are already buried inside V^π(s'), which is inside Q.

**Deriving the relationship between V and Q:**

Start from the definition of V^π(s) and peel off the t=0 term:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\underbrace{r(s_0,a_0) - \beta\log\frac{\pi(a_0|s_0)}{\pi_{\text{ref}}(a_0|s_0)}}_{t=0} + \sum_{t=1}^\infty \gamma^t(\ldots)\,\Big|\,s_0=s\right]$$

At t=0 the agent picks a_0 ~ π(·|s), transitions to s_1 ~ P(·|s,a_0). Factor the expectation:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}\!\left[r(s,a) - \beta\log\frac{\pi(a|s)}{\pi_{\text{ref}}(a|s)} + \mathbb{E}_{s' \sim P(\cdot|s,a)}\!\left[\sum_{t=1}^\infty \gamma^t(\ldots)\right]\right]$$

The inner expectation over s' is exactly γV^π(s') — the discounted future starting from the next state. So:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}\!\left[r(s,a) - \beta\log\frac{\pi(a|s)}{\pi_{\text{ref}}(a|s)} + \gamma\,\mathbb{E}_{s'}[V^\pi(s')]\right]$$

Recognize Q^π(s,a) = r(s,a) + γ E_{s'}[V^π(s')] and substitute:

$$\boxed{V^\pi(s) = \mathbb{E}_{a \sim \pi}\!\left[Q^\pi(s,a) - \beta\log\frac{\pi(a|s)}{\pi_{\text{ref}}(a|s)}\right]}$$

The KL cost sits inside the expectation over a because it is paid at the moment of choosing a, alongside Q^π.

---

## Step 4: Solve for the Optimal Policy and V*

Maximize V^π(s) over π(·|s) for each state s independently. This is exactly the same Lagrangian problem as Step 2, with Q*(s,a) playing the role of r(a):

$$\pi^*(a|s) = \frac{\pi_{\text{ref}}(a|s)\exp(Q^*(s,a)/\beta)}{Z(s)}, \quad Z(s) = \sum_{a'}\pi_{\text{ref}}(a'|s)\exp\!\left(\frac{Q^*(s,a')}{\beta}\right)$$

Now **substitute π* back** to find V*(s) explicitly. The log-ratio under π* evaluates to:

$$\log\frac{\pi^*(a|s)}{\pi_{\text{ref}}(a|s)} = \frac{Q^*(s,a)}{\beta} - \log Z(s)$$

Plug into the expression for V*:

$$V^*(s) = \sum_a \pi^*(a|s)\left[Q^*(s,a) - \beta\left(\frac{Q^*(s,a)}{\beta} - \log Z(s)\right)\right]$$

$$= \sum_a \pi^*(a|s)\left[\cancel{Q^*(s,a)} - \cancel{Q^*(s,a)} + \beta\log Z(s)\right]$$

$$= \beta\log Z(s) \cdot \underbrace{\sum_a \pi^*(a|s)}_{=1}$$

$$\boxed{V^*(s) = \beta\log\sum_{a'}\pi_{\text{ref}}(a'|s)\exp\!\left(\frac{Q^*(s,a')}{\beta}\right)}$$

The Q terms cancel exactly — the KL penalty "used up" all the reward information, leaving only β log Z(s). V* is purely the log-partition function, a soft-max (log-sum-exp) instead of the hard max of standard RL.

---

## Step 5: The Modified Bellman Equation (Not a New Postulate)

The standard Q Bellman equation always holds: Q*(s,a) = r(s,a) + γ E_{s'}[V*(s')]. Just substitute the expression for V* derived above:

$$Q^*(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}\!\left[\beta\log\sum_{a'}\pi_{\text{ref}}(a'|s')\exp\!\left(\frac{Q^*(s',a')}{\beta}\right)\right]$$

This is **not** a new assumption — it's the standard Bellman equation with V* = β log Z plugged in.

---

## Step 6: Invertibility — Policy and Advantage Are the Same Object

From the optimal policy formula:

$$\log\frac{\pi^*(a|s)}{\pi_{\text{ref}}(a|s)} = \frac{Q^*(s,a)}{\beta} - \underbrace{\log Z(s)}_{V^*(s)/\beta} = \frac{Q^*(s,a) - V^*(s)}{\beta} = \frac{A^*(s,a)}{\beta}$$

$$\boxed{A^*(s,a) = \beta\log\frac{\pi^*(a|s)}{\pi_{\text{ref}}(a|s)}}$$

In standard RL the argmax destroys the relative magnitudes of non-optimal actions — going from policy to value is ill-posed. Here the softmax preserves them. **Policy log-ratios and advantages are the same object, just scaled by β.**

---

## Step 7: Collapsing to a Single Head (Uniform Reference)

With uniform π_ref(a|s) = 1/|A|, the optimal policy is pure softmax over Q*:

$$\pi^*(a|s) = \frac{\exp(Q^*(s,a)/\beta)}{\sum_{a'}\exp(Q^*(s,a')/\beta)}$$

If the network outputs logits l(a|s) and applies softmax, then comparing to the above:

$$\frac{Q^*(s,a)}{\beta} = l(a|s) + C(s)$$

i.e. Q*(s,a) = β · l(a|s) up to a state-dependent constant that cancels in the softmax. Substituting into V*:

$$V^*(s) = \beta\log\sum_{a'}\frac{1}{|A|}\exp\!\left(\frac{Q^*(s,a')}{\beta}\right) = \beta\cdot\text{logsumexp}(l(\cdot|s)) + \text{const}$$

**A single policy head gives you everything:**

| Quantity | How to read it from the policy head |
|---|---|
| Q*(s,a) | β × logit of action a |
| V*(s) | β × logsumexp over all logits |
| A*(s,a) | β × log π*(a\|s)  (log-prob, not logit) |

No separate value head is needed. The value is the logsumexp of the policy logits — differentiable and free. This is exactly what **Soft Q-Learning** exploits.

---

## Summary: Why This Matters for Architecture

| Approach | Policy head | Value head | Formally tied? |
|---|---|---|---|
| Standard AC (PPO etc.) | Separate | Separate | No (only shared backbone) |
| SAC / SQL | Derived from Q | Derived from Q | **Yes — analytically** |
| Advantage-as-policy (REBEL) | = advantage head | log-partition | **Yes — same head** |
| IQL | BC + exp(A) reweighting | Separate V | Partially |
| AlphaZero | MCTS visit counts | MCTS backup | Via tree search |

The cleanest answer: under KL-regularized RL with uniform reference, a single softmax policy head implicitly parameterizes Q, V, and A simultaneously. Adding a separate value head is not wrong, but it introduces a redundancy that the math says you don't need.
