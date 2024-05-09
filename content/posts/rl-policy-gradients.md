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

The probability \\( \rho_{\pi}(\tau) \\) can be computed as follows:

## 1. Policy Gradients

Recall that the objective we're trying to solve in the RL setup is to maximize the
expected sum of rewards during our interactions with the environments. For now we'll
work on the Finite-Horizon setting, in which we can write the objective as follows:

\\[
J = \mathbb{E} \left[ \sum_{t=0}^{T-1} r(s_{t}, a_{t}) \right]
\\]

Unlike Value-Based methods, in which we tried to learn an optimal State-Value function
or Action-Value function, Policy Gradients is a Policy-Based method, where we try
to learn an optimal policy directly. We'll therefore enphasize this by having our
policy being parametrized by some parameters \\( \theta \\) of a neural network.
The objective can then be phrased as finding the best set of parameters that maximize
the expected sum of rewards, as follows:

\\[
\theta^{*} = \arg\min_{\theta} \mathbb{E}
    \left[ \sum_{t=0}^{T-1} r(s_{t}, a_{t}) \right]
\\]
