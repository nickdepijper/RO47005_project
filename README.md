# RO47005_project
Quadcopter project

dataclass:
  path_description:
    -start_pos
    -end_pos
    -n_timesteps

Class descriptions:

obstacle:
  attributes:
    public:
    private:
      -position_list
      -radius
      -path
      -current_path_index
      -move_forward
      -path_description[dataclass]
  functions:
    public:
      +get_position_list()
      +get_radius()
      +update_position()
    private:
      +generate_geometric_description()

drone:
  attributes:
    public:
      -
    private:
      -position
      -orientation
      -dynamic model
    
mpc:
  attributes:
    private:
      -obstacle_list
      -drone_list (filled with one drone)
      -
  functions:
      +generate_constraints()
      +solver()
      
world_description:
  attributes:
    private:

  functions:
    public:
      +generate_dynamic_obstacles
      +generate_static_obstacles
    private:
      +constructor_generate_all
      +generate_dynamic_obstacles
      +generate_static_obstacles




      
    
