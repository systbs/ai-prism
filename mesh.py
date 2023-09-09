import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_rectangular_prism(length, width, height, offset):
    x = np.linspace(offset[0], offset[0] + length, num=2)
    y = np.linspace(offset[1], offset[1] + width, num=2)
    z = np.linspace(offset[2], offset[2] + height, num=2)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    vertices = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    return vertices

def create(dirs=(2,2,2), limits=[(-1,1),(-1,1),(-1,1)]):
  n,m,p = dirs

  x1, x2 = limits[0]
  y1, y2 = limits[1]
  z1, z2 = limits[2]

  # Dimensions of the rectangular prism
  unit_length = (x2 - x1) / n
  unit_width = (y2 - y1) / m
  unit_height = (z2 - z1) / p

  # Creating a mesh composed of n by m by p by l rectangular prisms
  all_prism = np.empty((0, 8), dtype=np.int64)
  all_vertices = np.empty((0, 3))
  for i in range(n):
      for j in range(m):
          for k in range(p):
            offset = [x1 + i * unit_length, y1 + j * unit_width, z1 + k * unit_height]
            prism_vertices = create_rectangular_prism(unit_length, unit_width, unit_height, offset)
            new_order = np.array([0,4,6,2,1,5,7,3])
            sorted_array = prism_vertices[new_order]

            # Calculate the original length of the list
            original_length = len(all_vertices)

            all_vertices = np.concatenate((all_vertices, sorted_array), axis=0)

            # Create indices for the newly added rows
            new_indices = list(range(original_length, len(all_vertices)))

            all_prism = np.concatenate((all_prism, [new_indices]), axis=0)

  unique_vertices, unique_indices = np.unique(all_vertices, axis=0, return_index=True)

  # Find the indices of removed rows
  removed_indices = np.setdiff1d(np.arange(len(all_vertices)), unique_indices)

  # Create a dictionary for mapping indices
  index_mapping = {tuple(vertex): idx for idx, vertex in zip(unique_indices, unique_vertices)}

  def replace_elements(the_map, x):
      return the_map.get(x, x)

  # Create a dictionary to hold the mapping of removed_indices to corresponding unique_indices
  replacement_map_1 = {index: index_mapping[tuple(all_vertices[index])] for index in removed_indices}
  vectorized_replace_1 = np.vectorize(lambda x: replace_elements(replacement_map_1, x))

  replacement_map_2 = {value: index for index, value in enumerate(unique_indices)}
  vectorized_replace_2 = np.vectorize(lambda x: replace_elements(replacement_map_2, x))

  # Update all_prism using unique_indices
  updated_prism = np.zeros_like(all_prism)
  for i, prism_row in enumerate(all_prism):
      prism_row = vectorized_replace_1(prism_row)
      prism_row = vectorized_replace_2(prism_row)
      updated_prism[i] = prism_row

  # Remove duplicate rows from all_vertices and their corresponding rows from all_prism
  all_vertices = unique_vertices
  all_prism = updated_prism

  return all_prism, all_vertices

def plot(prism, vertices, title="", selected_idx=[]):
  # Create a 3D scatter plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Extract coordinates from all_vertices
  x_coords = vertices[:, 0]
  y_coords = vertices[:, 1]
  z_coords = vertices[:, 2]

  selected_idx = np.unique(np.array(selected_idx, dtype=np.int64), axis=0)
  normal_idx = np.setdiff1d(np.arange(len(vertices)), selected_idx)

  # Scatter plot for all_vertices
  ax.scatter(x_coords[normal_idx], y_coords[normal_idx], z_coords[normal_idx], marker='o')
  ax.scatter(x_coords[selected_idx], y_coords[selected_idx], z_coords[selected_idx], marker='*')

  # Plot the edges of the cubes
  for prism_indices in prism:
      vertices_t = vertices[prism_indices]
      edge_order = np.array([1,2,2,3,3,4,4,1,5,6,6,7,7,8,8,5,1,5,2,6,3,7,4,8])
      edge_order = edge_order - 1
      for i in range(0, len(edge_order), 2):
          start_vertex = edge_order[i]
          end_vertex = edge_order[i + 1]
          ax.plot([vertices_t[start_vertex, 0], vertices_t[end_vertex, 0]],
                  [vertices_t[start_vertex, 1], vertices_t[end_vertex, 1]],
                  [vertices_t[start_vertex, 2], vertices_t[end_vertex, 2]], color='gray')

  # Set labels for the axes
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title(title)

  # Show the plot
  plt.show()