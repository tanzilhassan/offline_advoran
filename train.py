import os
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common
from tf_agents.policies import policy_saver

import config
import ran_env_wrapper
import agent_builder

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus and config.use_gpu_in_env:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def compute_avg_return(environment, policy, num_episodes=5, verbose=False):
    total_return = 0.0
    for i in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        
        if verbose:
            print(f"\n--- Evaluation Episode {i+1} ---")
            
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            
            # This triggers the print statement defined in ran_env.py
            if verbose:
                try:
                    environment.render()
                except Exception:
                    pass
                    
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def main():
    print("--- Starting PPO Training Pipeline ---")
    
    # 1. Init Envs
    train_env = ran_env_wrapper.get_training_env()
    eval_env = ran_env_wrapper.get_eval_env()

    # 2. Init Agent
    agent = agent_builder.create_ppo_agent(train_env)

    # 3. Replay Buffer (Sized for one iteration batch)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=config.num_steps_per_episode * 5
    )

    # 4. Collection Driver
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=config.collect_episodes_per_iteration
    )

    # 5. Graph Optimization
    agent.train = common.function(agent.train)
    collect_driver.run = common.function(collect_driver.run)
    agent.train_step_counter.assign(0)

    # 6. Checkpointing & Logging
    train_summary_writer = tf.summary.create_file_writer(config.log_dir)
    train_checkpointer = common.Checkpointer(
        ckpt_dir=config.checkpoint_dir,
        max_to_keep=5,
        agent=agent,
        policy=agent.policy,
        global_step=agent.train_step_counter
    )
    train_checkpointer.initialize_or_restore()

    print(f"Training for {config.total_training_iterations} iterations...")

    # --- Training Loop ---
    for i in range(config.total_training_iterations):
        step = agent.train_step_counter.numpy()

        # A. Collect
        collect_driver.run()

        # B. Train
        trajectories = replay_buffer.gather_all()
        train_loss = agent.train(experience=trajectories)
        replay_buffer.clear() # Critical for PPO

        # C. Log
        if step % config.log_interval == 0:
            print(f'step = {step}: loss = {train_loss.loss}')
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.loss, step=step)

        # D. Eval & Save
        if step % config.eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_episodes=5)
            print(f'step = {step}: Average Return = {avg_return}')
            with train_summary_writer.as_default():
                tf.summary.scalar("average_return", avg_return, step=step)
            train_checkpointer.save(global_step=step)

    # --- Save Final Policy ---
    print("Saving Final Policy...")
    if not os.path.exists(config.policy_dir):
        os.makedirs(config.policy_dir)
    saver = policy_saver.PolicySaver(agent.policy, batch_size=None)
    saver.save(config.policy_dir)
    print("Done.")

if __name__ == "__main__":
    main()