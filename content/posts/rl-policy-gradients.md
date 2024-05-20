+++ 
draft = false
date = 2024-05-08T14:48:04-05:00
title = "RL using Policy Gradients"
description = "A post where I explain Policy Gradients in the context of RL"
slug = ""
authors = ["Wilbert Pumacay"]
tags = ["RL", "ML"]
categories = ["RL"]
externalLink = ""
series = []
katex=true
+++

This is the first post on a series dedicated to Policy Gradient based algorithms,
in which we will cover the theory of how Policy Gradients work. We'll implement
the vanilla version of the `REINFORCE` algorithm, and use it to solve some simple
tasks from `Gymnasium`.

## 0. Preliminaries

Before getting into the math of Policy Gradients, let's define some terms that will
help us make the math simpler. Let's define a `trajectory` \\( \tau \\) as the collection
of all states and actions that we get from the interaction with the environment.
For example, for the finite horizon case we will define a trajectory as follows:

\\[
\tau = \left( s_{0},a_{0},s_{1},a_{1},\dots, s_{T-2}, a_{T-2}, s_{T-1} \right)
\\]

These trajectories come from a certain distribution induced by our policy \\(\pi\\),
which we denote as follows:

\\[
\tau \sim \rho_{\pi}(\tau)
\\]

## 1. Policy Gradients

Recall that the objective we're trying to solve in the RL setup is to maximize the
expected sum of rewards during our interactions with the environment. For now we'll
work on the Finite-Horizon setting, in which we can write the objective as follows:

\\[
J = \mathbb{E} \left[ \sum_{t=0}^{T-1} r(s_{t}, a_{t}) \right]
\\]

Unlike Value-Based methods, in which we tried to learn an optimal State-Value function
or Action-Value function, Policy Gradients is a Policy-Based method, where we try
to learn an optimal policy \\( \pi \\) directly. In the tabular case, where we have
a discrete finite set of states \\( \mathrm{S} \\) and a discrete finite set of actions
\\( \mathrm{A} \\), we can represent the policy \\( \pi \\) as a lookup table.
However, to deal with the case of infinite sets, or for continuous state and action
spaces, we have to resort to use a function approximator (like a neural network),
to represent our policy. We'll state this by writing \\( \pi_{\theta} \\) as our
policy that is being parametrized by a neural network with parameters \\(\theta\\).
The objective can then be phrased as finding the best set of parameters \\(\theta^{\ast}\\)
that maximize the expected sum of rewards, as follows:

\\[
\theta^{\ast} = \arg\max_{\theta} \mathbb{E}
    \left[ \sum_{t=0}^{T-1} r(s_{t}, a_{t}) \right]
\\]

Using the notation of trajectories, we can re-write the sum of rewards over an
episode in a more compact, using \\( r(\tau) = \sum_{t=0}^{T-1} r(s_{t}, a_{t}) \\)
we can write the objective as follows:

\\[
\theta^{\ast} = \arg\max_{\theta} E_{\tau \sim p_{\theta}(\tau)}
    \left[ r(\tau) \right]
\\]

Unrolling the expectation, we have that the objective we want to optimize is the
following:

\\[
\tag{1} J(\theta) = \int p_{\theta}(\tau) r(\tau) d\tau
\\]

### 1.1. Direct Policy Differentiation

The Policy Gradients method tries to directly optimize the objective from eq. (1),
by directly differentiating such objective and using gradient ascent with the
resulting gradient.

\\[
\nabla_{\theta} J(\theta) = \nabla_{\theta} \int p_{\theta}(\tau) r(\tau) d\tau =
    \int \nabla_{\theta} p_{\theta}(\tau) r(\tau) d\tau
\\]

Note however that we don't have access to the environment dynamics, so we can't
directly compute the gradient over the trajectory distribution. Instead, what we
have are samples from the environment. We can rearrange the previous expression
by using the following trick:

\\[
\nabla_{\theta} p_{\theta}(\tau) =
    p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) \tag{2}
\\]

Replacing the expression in eq. (2), we get the following expression for the gradient
of the objective (`Policy Gradient`):

\\[
\hat{g} = \nabla_{\theta} J(\theta) =
    \int p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) d\tau
\\]

This last expression is in the form of an expectation, so we van just rearrange it
back to the expectation form, resulting the following expression for the
`Policy Gradient`.

\\[
\hat{g} = E_{\tau \sim p_{\theta}(\tau)}
    \left[ \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) \right] \tag{3}
\\]

Because we now have an expectation, we can compute this estimate of the gradient
using samples, namely trajectory samples \\(\tau^{(k)} \\) as follows:

\\[
\hat{g} \approx \frac{1}{N} \sum_{i=0}^{N-1}
    \nabla_{\theta} \log p_{\theta}(\tau^{(i)}) r(\tau^{(i)})
\\]

Note that the only quantity we still have to compute is the gradient over the
trajectory distribution \\(p_{\theta}(\tau)\\), which we can expand as follows:

\\[
p(\tau) = p(s_{0},a_{0},\dots,s_{T-2},a_{T-2},s_{T-1})=
    p(s_{0}) \prod_{t=0}^{T-2} \pi_{\theta}(a_{t}|s_{t}) p(s_{t+1}|s_{t}, a_{t})
\\]

Replacing this expression in our estimate for the gradient we obtain the following:

\\[
\hat{g} \approx \frac{1}{N} \sum_{i=0}^{N-1}
    \nabla_{\theta} \log \left( p(s_{0}) \prod_{t=0}^{T-2}
        \pi_{\theta}(a_{t}|s_{t}) p(s_{t+1}|s_{t}, a_{t}) \right) r(\tau)
\\]

Note that the expression inside the \\( \log \\) can be split into terms, as the
logarithm splits when using products as argument. This results in the following
expression:

\\[
\hat{g} \approx \frac{1}{N} \sum_{i=0}^{N-1} \sum_{t=0}^{T-1}
    \nabla_{\theta} \left[ \log p(s_{0}) + \sum_{t=0}^{T-2}
        \pi_{\theta}(a_{t}|s_{t}) p(s_{t+1}|s_{t}, a_{t})
            \right] r(\tau)
\\]
