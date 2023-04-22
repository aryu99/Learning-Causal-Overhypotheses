# Learning Causal Overhypotheses
Project for UT Austin ECE 381V Causal Reinforcement Learning

## Usage

```(bash)
$ python dibs_scipt.py [-h] [--num_samples NUM_SAMPLES] [--num_particles NUM_PARTICLES] [--n_vars N_VARS] [--particle_dim PARTICLE_DIM] [--num_codebook_per_node NUM_CODEBOOK_PER_NODE] --method {VQDiBS,JointDiBS} [--plot_dir PLOT_DIR] --env_config ENV_CONFIG [--exp_config EXP_CONFIG]

arguments:
  -h, --help            show this help message and exit
  --num_samples NUM_SAMPLES
                        Number of observation samples to collect
  --num_particles NUM_PARTICLES
                        Number of particles to use for $Z$
  --n_vars N_VARS       Number of nodes in SCM
  --particle_dim PARTICLE_DIM
                        Dimension of embedding. If `PARTICLE_DIM = N_VARS`, then embedding will be full-rank.
  --num_codebook_per_node NUM_CODEBOOK_PER_NODE
                        Number of codebooks per node
  --method {VQDiBS,JointDiBS}
                        Method to use for embedding
  --plot_dir PLOT_DIR   Directory to save plots
  --env_config ENV_CONFIG
                        Environment config file
  --exp_config EXP_CONFIG
                        Experiment config file
```

