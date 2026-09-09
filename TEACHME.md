<!-- This document describes the aim and motivation of this project. -->

# It Takes Two (it2): Emergent Locomotion Skills from Multi-Agent Games

## Motivation

RL-based locomotion often depends on hand-crafted task rewards, command distributions, and curricula. Instead of designing a curriculum over single-agent locomotion commands, this project starts from a multi-agent game: a chaser-evader setting that can generate diverse locomotion behavior through interaction.

The game policy itself is not the final product. A policy trained only to play the game is useful as a behavior generator, but it is not directly reusable for downstream tasks that require explicit commands, goals, reasoning, or planning. The central goal of `it2` is to turn the trajectories induced by the game into reusable locomotion skills and perceptual goal models.

## Core Idea

The project follows a four-stage story:

1. Train a primary policy to play the multi-agent game.
2. Relabel the resulting trajectories into goal or command targets.
3. Distill game behavior into a goal-conditioned or command-conditioned policy interface.
4. Use the learned goal proposal model and goal-conditioned policy as building blocks for planning.

In notation, the goal proposal model can be written as

$$
p^\pi(g \mid s_t),
$$

where `s_t` is the agent's current perception and `g` is a goal label derived from a future trajectory segment `(s_t, s_{t+1}, ..., s_{t+H})` under the primary policy `pi`.

Examples of goal labels include:

- $g = (x, \dot{x})_{t+1}$: next-step position and velocity, useful for extracting a locomotion command interface.
- $g = \mathbf{x}_{t+H}$: a future position or root-state target to reach.
- $g = (\mathbf{x}_{t}, \mathbf{x}_{t+1}, ..., \mathbf{x}_{t+H})$: a short future trajectory.

## Current Implementation

The repository currently implements the game, the primary PPO policy, trajectory relabeling, policy distillation, and future prediction. A full planner is not implemented yet.

`src/it2/commands.py` defines `Game`, a paired chaser-evader command. It assigns chaser and evader roles, computes relative target commands, tracks catch events, records root trajectories, and optionally exposes a secondary replay command.

`cfg/task/A2Game.yaml` and `cfg/task/G1Game.yaml` instantiate this game for the Unitree A2 quadruped and G1 humanoid. Both tasks combine game rewards with regular locomotion rewards and terrain randomization. They also expose exteroceptive observations: a raycast camera for A2 and a height scan for G1.

`src/it2_learning/ppo_it2.py` defines `PPOPolicy`, whose configured stages are:

- `policy`: train the primary game-playing PPO policy from `primary_command`.
- `policy_distill`: fit the relabeled secondary command pathway to match the primary policy pathway.
- `future`: train a future predictor from relabeled trajectory targets.

## Primary Policy

The primary policy is the game-playing policy trained with PPO. Its command is `primary_command`, which encodes the relative target state and role information from the chaser-evader game.

The game creates an automatic curriculum because the chaser and evader adapt to one another. As training improves, the policy should encounter increasingly varied locomotion behavior: pursuit, escape, turning, obstacle negotiation, acceleration, stopping, and recovery.

These rollout trajectories are the raw material for the later stages. In code, `root_state_w` is part of the rollout data expected by the learning module, so future states can be relabeled into command or goal targets after collection.

## Relabeled Goals and Commands

The project uses programmatic relabeling rather than manual annotation. In `src/it2_learning/future_relabel.py`, the implemented target families are:

- `LocomotionCommand`: a next-step body-frame command consisting of relative xy position, xy linear velocity, and yaw rate.
- `FutureState`: an H-step future root-state target relative to the current body frame.
- `FutureTrajectory`: a short sequence of future relative root states, optionally decimated.

This makes `g` a specified goal parameterization computed from trajectories. The broader research question is which parameterizations produce reusable skills and useful planning abstractions.

## Skill Extraction by Distillation

The `policy_distill` stage extracts a locomotion-style interface from the game policy. The primary command path acts as the teacher. The secondary command path receives relabeled `LocomotionCommand` targets and is trained to match the primary path's fused feature and actor mean.

This is a form of self-distillation: the game policy supplies both the behavior and the supervision, while the student pathway learns to reproduce that behavior through a more reusable command representation.

The intended outcome is a goal-conditioned or command-conditioned policy that can be driven by explicit locomotion goals rather than by the original chaser-evader game command.

Here, secondary tasks mean downstream objectives that are not identical to winning the chaser-evader game, but can reuse the skills discovered by the game policy. Examples include following a desired velocity, reaching a waypoint, choosing footholds, traversing difficult terrain, or executing a proposed future trajectory.

## Goal Proposal and Future Prediction

The `future` stage is the implementation closest to the goal proposal model $p^\pi(g \mid s_t)$. Given the current perception, it predicts possible future trajectory targets generated by the primary policy's behavior distribution.

Two predictor families are implemented:

- `VAEFuturePredictor`: a conditional VAE-style model for multi-modal future targets.
- `CFMFuturePredictor`: a conditional flow-matching model for trajectory generation.

Both predictors are trained on `_future_target` labels produced from rollout trajectories. In this sense, the model learns what futures are plausible under the dataset induced by the primary policy, rather than learning an externally specified task planner.

## Exteroception and Information Gain

The project is explicitly perceptive: the agent can condition on proprioception alone or on proprioception plus exteroceptive signals. This distinction matters because terrain perception can change the set of plausible and useful future goals.

`EncoderTwo` in `src/it2_learning/encoders.py` creates separate context features for future prediction. One future query is restricted to proprioceptive information, while another can attend to both proprioception and exteroception. This lets the project compare:

- $p^\pi(g \mid o_{\text{pro}})$: futures predicted from proprioception alone.
- $p^\pi(g \mid o_{\text{pro}}, o_{\text{ext}})$: futures predicted with exteroceptive information.

For the VAE predictor, the logged quantity `future/latent_ig` is based on `KL(p(z|o_ext)||p(z|o_pro))`. This provides a concrete signal for how much the exteroceptive observation changes the latent distribution over possible futures.

## Planning Status

The repo does not yet contain a full deliberative planner, such as MPC, tree search, or graph search over proposed goals. The current evaluation path can sample proprioceptive and exteroceptive future predictions and expose them as visualization overlays.

The planning direction is therefore:

1. Use the future predictor as a goal proposal distribution.
2. Score or select proposed goals according to a downstream objective.
3. Execute selected goals through the distilled goal-conditioned policy.
4. Replan as perception changes.

This keeps the current implementation honest: `it2` already learns game behavior, relabeled command interfaces, and perceptual future models; explicit reasoning and planning are the next layer to build on top.

Questions:

1. How should future contact positions be relabeled as goals, given that contact choices largely determine terrain traversability?
2. Is pure distillation enough for the secondary policy, or do we need an RL fine-tuning stage with rewards for secondary tasks?

Agent's Answer:

For future contact positions, the most useful goal label is probably not just raw world-frame foot positions. Contact goals should be represented relative to the current body/root frame, tied to a short horizon, and masked by episode validity. This matches the existing `NextContactPos` relabeler in `future_relabel.py`, which stacks future `last_contact_pos` samples over the next `H` steps and expresses them in the root frame at time `t`.

A good contact-goal target could be:

$$
g_t^{contact} = \{R_t^\top(p^{foot}_{t+k} - x_t)\}_{k=0}^{H-1},
$$

optionally with one row per foot and a validity/contact mask. This keeps the label invariant to global pose and makes it easier for a policy or predictor to learn terrain-dependent foothold choices. For planning, it may be better to predict a distribution over contact sequences or contact regions rather than a single deterministic footstep sequence, because many foothold choices can be valid on the same terrain.

I would separate contact-goal learning into two levels:

1. A low-level relabeling target: future contact positions or contact patches in the body frame.
2. A higher-level traversability objective: whether the proposed contacts lead to stable motion, progress, and low recovery cost. (comment: this point will be another module's responsibility)

The current distillation strategy is a good first step, but it is unlikely to be sufficient by itself. The primary game policy can act as a teacher for states and relabeled commands that lie close to its own rollout distribution. Once the secondary policy is driven by externally selected goals, planner-selected futures, or contact goals, it will encounter distribution shift. In that regime, behavior cloning or feature/action distillation can become brittle.

Since there is no separate expert teacher for arbitrary secondary tasks, DAgger is not directly available unless we treat the primary policy as an oracle only for states where its original game command is still meaningful. A more realistic path is:

1. Use distillation to initialize the secondary command-conditioned policy.
2. Add RL fine-tuning on secondary tasks with rewards that encode task success, stability, contact quality, and regularization.
3. Keep the distilled policy close to the game-derived behavior with an auxiliary imitation or KL loss, at least early in fine-tuning.
4. Expand the command/goal distribution gradually, so the policy does not immediately leave the support of the game trajectories.

In other words, distillation should provide the skill prior, while RL should adapt that prior to explicit downstream objectives. This makes the project story stronger: the game discovers diverse locomotion; relabeling and distillation expose the skills through a reusable interface; secondary-task RL turns that interface into robust task execution.

Comment: here comes the question: we do not really want to design rewards for each secondary task, which kind of defeats the original
motivation of using a multi-agent game. Can we come up with a general goal-reaching or following objective or reward function that can
work with different secondary tasks?

Agent's Answer:

Yes, but the objective should be general over a small set of goal spaces, not completely task-free. The key is to make the task specify a goal representation `g`, while the reward is always computed by a reusable distance or tracking metric in that goal space.

For example, define a goal extractor

$$
\phi(s_{t:t+H}) \rightarrow g,
$$

where `g` can be a next-step locomotion command, a future root state, a contact sequence, or a short trajectory. Then train the secondary policy with a generic goal-reaching objective:

$$
r_t = -d(\phi(s_{t:t+H}), g_t) + \lambda_p \Delta progress - \lambda_u ||a_t||^2 - \lambda_c cost(s_t).
$$

The distance `d` changes with the goal type, but the objective form does not. For root-state goals, `d` can be body-frame position, yaw, and velocity error. For trajectory goals, it can be an average tracking error over the horizon. For contact goals, it can compare predicted or realized foot contacts to target contact patches with a validity mask. The regularization terms can remain shared: survival, stability, smoothness, energy, action rate, and safety constraints.

This preserves the motivation of the game because we are not hand-designing a new semantic reward for every secondary task. The game still discovers the behavior distribution and the relabeled goals. The secondary RL objective only teaches the policy to reliably reach or follow goals from that discovered goal space.

A useful way to phrase the research goal is:

> Learn a universal goal-conditioned locomotion policy over relabeled goals from the game, using a shared goal-reaching objective instead of task-specific reward engineering.

The important design choice is therefore the goal interface. If `g` is expressive enough, many secondary tasks reduce to choosing goals: velocity following chooses velocity-like goals, navigation chooses future root positions, foothold planning chooses contact goals, and terrain traversal chooses feasible future trajectories. Planning or goal proposal decides `g`; the policy only needs to solve the generic follow/reach problem.