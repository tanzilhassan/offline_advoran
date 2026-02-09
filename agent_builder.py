import tensorflow as tf
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
import config

def create_ppo_agent(train_env):
    """Constructs PPO Agent with Actor/Value Networks."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.ppo_learning_rate)

    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()

    # Actor Network
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=config.ppo_actor_fc_layers,
        activation_fn=tf.keras.activations.tanh 
    )

    # Value Network
    value_net = value_network.ValueNetwork(
        observation_spec,
        fc_layer_params=config.ppo_value_fc_layers,
        activation_fn=tf.keras.activations.tanh
    )

    # PPO Agent
    agent = ppo_clip_agent.PPOClipAgent(
        train_env.time_step_spec(),
        action_spec,
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=config.ppo_entropy_regularization,
        importance_ratio_clipping=config.ppo_importance_ratio_clipping,
        normalize_observations=config.ppo_normalize_observations,
        normalize_rewards=config.ppo_normalize_rewards,
        use_gae=config.ppo_use_gae,
        num_epochs=config.ppo_num_epochs,
        debug_summaries=False,
        summarize_grads_and_vars=False
    )

    agent.initialize()
    return agent