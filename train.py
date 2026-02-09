import os
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.system import system_multiprocessing as multiprocessing

import config
import ran_env_wrapper
import agent_builder

tf.config.optimizer.set_jit(True) 

gpus = tf.config.list_physical_devices('GPU')
if gpus and config.use_gpu_in_env:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def compute_avg_return(environment, policy, num_episodes=3):
    """
    Evaluates the policy and prints the PRB/Sched configurations 
    via the environment's render method.
    """
    total_return = 0.0
    for i in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        
        print(f"\n--- Evaluation Episode {i+1} ---")
        
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            

            try:
                environment.pyenv.render()
            except Exception:
                pass
                    
        total_return += episode_return
        
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def main(_):
    print(f"--- STARTING TRAINING RUN: {config.run_id} ---")
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.policy_dir, exist_ok=True)


    train_env = ran_env_wrapper.get_training_env()
    eval_env = ran_env_wrapper.get_eval_env()

    agent = agent_builder.create_ppo_agent(train_env)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=config.collect_episodes_per_iteration * config.num_steps_per_episode * 2
    )

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=config.collect_episodes_per_iteration
    )

    # Graph Optimization & Counter Reset
    agent.train = common.function(agent.train)
    collect_driver.run = common.function(collect_driver.run)
    agent.train_step_counter.assign(0)

    # Setup Checkpointer
    train_checkpointer = common.Checkpointer(
        ckpt_dir=config.checkpoint_dir,
        max_to_keep=5,
        agent=agent,
        policy=agent.policy,
        global_step=agent.train_step_counter
    )
    
    # Initialize from scratch or restore if a checkpoint exists in the run folder
    restored_path = train_checkpointer.initialize_or_restore()
    if restored_path:
        print(f"Restored training from: {restored_path}")
    else:
        print("Starting fresh training (random initialization).")

    # Setup TensorBoard Summary Writer
    train_summary_writer = tf.summary.create_file_writer(config.log_dir)

    # --- Training Loop ---
    start_step = agent.train_step_counter.numpy()
    print(f"Training for {config.total_training_iterations} iterations starting from step {start_step}...")

    for _ in range(start_step, config.total_training_iterations):
    
        collect_driver.run()

        trajectories = replay_buffer.gather_all()
        train_loss = agent.train(experience=trajectories)
        
        replay_buffer.clear() 

        step = agent.train_step_counter.numpy()

        if step % config.log_interval == 0:
            print(f'Step {step}: loss = {train_loss.loss:.4f}')
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.loss, step=step)

        if step % config.eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy)
            print(f'Step {step}: Average Return = {avg_return:.4f}')
            
            with train_summary_writer.as_default():
                tf.summary.scalar("average_return", avg_return, step=step)
            
            train_checkpointer.save(global_step=step)

    # --- Final Policy Export ---
    print("\n--- Training Complete. Saving Final Policy ---")
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(config.policy_dir)
    print(f"Policy successfully saved to: {config.policy_dir}")

if __name__ == "__main__":
    multiprocessing.handle_main(main)
