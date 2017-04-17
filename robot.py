import numpy as np
import turtle
import sys
import math
import copy
import operator

class Robot(object):
	def __init__(self, maze_dim):
		"""
		Use the initialization function to set up attributes that your robot
		will use to learn and navigate the maze. Some initial attributes are
		provided based on common information, including the size of the maze
		the robot is placed in.
		"""

		self.maze_dim = maze_dim

		self.racing = False
		self.race_time_step = 0
		
		self.initial_location = (0,0)
		self.location = self.initial_location
		self.heading = 'up'
		self.latest_sensor_reading = None

		self.optimal_path = None
		self.optimal_steps = None
		self.goal_visited = False

		self.take_additional_steps = False
		self.additional_step_instructions = []
		self.nodes_visited = [self.initial_location]
		self.take_second_step = False
		self.second_step_instructions = None

		self.mapper = self.intiliase_maze_graph(maze_dim)
		
	def intiliase_maze_graph(self, maze_dim):
		self.maze_dim = maze_dim

		self.goal_location = None

		self.walls = self.initial_walls(maze_dim)
		self.cell_possibilities = self.initial_cell_possibilities(maze_dim)

		dim = maze_dim
		self.goal_wall_coords = [(dim + 1, dim / 2 + 1), (dim + 2, dim / 2),
								 (dim + 2, dim / 2 - 1), (dim + 1, dim / 2 - 1),
								 (dim - 1, dim / 2 - 1), (dim - 2, dim / 2 - 1),
								 (dim - 2, dim / 2), (dim - 1, dim / 2 + 1)]

	def maze_first_explore(self):
		'''
		Exploration of the maze by deciding the rotation to do.
		'''
		if self.goal_location == self.location and not self.goal_visited:
			self.goal_visited = True

		step = self.calculate_next_step()
		rotation, movement = step

		print "sensors info={} \ncurrent_robot_location = {} \nheading={} \n(rotation,movement)={}\n".format(self.latest_sensor_reading,
			self.location, self.heading, step)
		self.print_maze_consol(self.location, self.heading)

		self.location = self.calculate_node(self.location, self.heading, step)
		self.heading = self.calculate_heading(self.heading, rotation)

		self.nodes_visited.append(self.location)

		return rotation, movement

	def calculate_next_step(self):
		'''
		In order to improve the efficiency of the Robot's traversal of the maze
		and to avoid the Robot getting stuck 'ping-ponging' between two nodes
		indefinitely, the Navigator will attempt to follow the first two steps
		of each calculated path through the maze. The only cases when this isn't
		done is when the path has only one step or when the second step becomes
		invalid as a result of the sensor reaidngs following the first step.

		When attempting to calculate a new path, the target node is either the
		closest node with the greatest uncertainty or the goal node if its
		location is known, but has not yet been visited.
		'''
		loc = self.location
		heading = self.heading

		if self.take_additional_steps:
			next_step = self.additional_step_instructions.pop()
			next_node = self.calculate_node(loc, heading, next_step)
			if next_node != None and self.move_is_valid(loc, next_node):
				self.take_additional_steps = len(self.additional_step_instructions) > 0
				return next_step
			else:
				self.take_additional_steps = False

		second_step_node = None
		second_step = self.second_step_instructions

		if self.take_second_step and second_step != None:
			second_step_node = self.calculate_node(loc, heading, second_step)

		if second_step_node != None and self.move_is_valid(loc, second_step_node):
			self.take_second_step = False
			return second_step

		# Navigate to the location of the maze with least knowledge.
		target = self.closest_least_certain_node()
		# If the goal has been found, but not yet visited, go there instead.
		if not self.goal_visited and self.goal_found():
			target = self.goal_location
		maze_graph = self.build_graph_from_maze()
		path = self.Dijkstra_best_path(maze_graph, loc, target)
		steps = self.convert_path_to_steps(path, heading)

		repeat_length = self.is_loop()
		if repeat_length > 0:
			self.take_additional_steps = True
			self.additional_step_instructions = steps[1:repeat_length + 1]

		if len(steps) > 1:
			self.take_second_step = True
			self.second_step_instructions = steps[1]
		else:
			self.second_step_instructions = None
			
		rotation = steps[0]
		return rotation

	def is_loop(self):
		'''
			Check to know whether the robot is doing loop between a set of nodes
		'''
		loop_lengths = range(2, 6)
		robot_is_stuck = False
		repeat_length = 0
		for length in loop_lengths:
			first_path  = self.nodes_visited[-length:]
			second_path = self.nodes_visited[-length * 2 + 1:-length + 1]
			second_path.reverse()
			third_path  = self.nodes_visited[-length * 3 + 2:-length * 2 + 2]
			if first_path == second_path and second_path == third_path:
				repeat_length = length
				break

		return repeat_length


	def closest_least_certain_node(self):
		'''
		Find the node with the greatest uncertainty (greatest number of
		possible shapes) that is closest to the current location.
		'''
		uncertainties = self.cell_possibilities
		max_uncertainty = max([max(column) for column in uncertainties])
		peak_locations = []
		for i in range(self.maze_dim):
			for j in range(self.maze_dim):
				if uncertainties[i][j] == max_uncertainty:
					peak_locations.append((i, j))
		closest_peak = peak_locations[0]
		if len(peak_locations) > 1:
			loc = self.location
			for k in range(len(peak_locations)):
				dist_a = self.distance_between_nodes(loc, closest_peak)
				dist_b = self.distance_between_nodes(loc, peak_locations[k])
				if dist_a > dist_b:
					closest_peak = peak_locations[k]
		return closest_peak

	def build_graph_from_maze(self, fastest_route = False, treat_unknown_as_walls = False):
		'''
			Creates an undirected graph from the maze.
		'''
		graph = {}
		open_list = set([self.initial_location])

		while len(open_list) > 0:
			location = open_list.pop()
			if location in graph.keys():
				next
			else:
				graph[location] = []
			
			x, y = location
			for direction in ['up', 'right', 'down', 'left']:
				for i in range(1,4):
					ty, tx = y, x
					if direction == 'up':
						tx  = x + i
					elif direction == 'right':
						ty  = y + i
					elif direction == 'down':
						tx  = x - i
					elif direction == 'left':
						ty  = y - i

					target = (tx, ty)

					if self.move_is_valid(location, target, treat_unknown_as_walls):
						graph[location].append(target)
						if target not in graph.keys():
							open_list.add(target)
						if not fastest_route and self.cell_possibilities[tx][ty] > 1:
							break
					else:
						break

		return graph

	def move_is_valid(self, location, target, treat_unknown_as_walls = False):
		'''
		Will moving from location to target, given the current knowledge of the
		maze, result in hitting a wall?
		- If treat_unknown_as_walls, an attempt to move from location to target
			through a wall / openning of unknown state is considered invalid.
		'''
		valid_move = True
		x, y = location
		tx, ty = target

		wall_values = [1]
		if treat_unknown_as_walls:
			wall_values.append(-1)

		if y == ty:
			if tx < 0 or tx >= self.maze_dim:
				valid_move = False
			elif x < tx:
				for i in range(tx - x):
					if self.walls[2 * (x + i + 1)][y] in wall_values:
						valid_move = False
						break
			else:
				for i in range(x - tx):
					if self.walls[2 * (x - i)][y] in wall_values:
						valid_move = False
						break
		else:
			if ty < 0 or ty >= self.maze_dim:
				valid_move = False
			elif y < ty:
				for i in range(ty - y):
					if self.walls[2 * x + 1][y + i + 1] in wall_values:
						valid_move = False
						break
			else:
				for i in range(y - ty):
					if self.walls[2 * x + 1][y - i] in wall_values:
						valid_move = False
						break

		return valid_move
	
	def Dijkstra_best_path(self, graph, start, target, print_path_costs = False):
		'''
		Dijkstra's algorithm to find the fastest path from start to target
		through the given undirected graph.
		'''
		optimal_path = []
		#print start
		# Make sure the target is in the graph
		if target in graph.keys():
			# Assign to every node a tentative distance value: set it to zero for
			# our initial node and to infinity for all other nodes.

			largest_possible_cost = self.maze_dim ** 2

			path_costs = {}

			# Used for sorting by path cost.
			cost_for_node = lambda n: path_costs[n]

			for node in graph.keys():
				path_costs[node] = largest_possible_cost
			path_costs[start] = 0

			# Set the initial node as current. Mark all other nodes unvisited.
			# Create a set of all the unvisited nodes called the unvisited set.
			current_node = start
			unvisited_list = copy.copy(graph.keys())

			while len(unvisited_list) > 0:
				# For the current node, consider all of its neighbours and
				# calculate their tentative distances. Compare the newly
				# calculated tentative distance to the current assigned value
				# and assign the smaller one otherwise, keep the current value.

				distance = path_costs[current_node] + 1
				for neighbour in graph[current_node]:
					if path_costs[neighbour] > distance:
						path_costs[neighbour] = distance

				# When we are done considering all of the neighbors of the current
				# node, mark the current node as visited and remove it from the
				# unvisited set. A visited node will never be checked again.

				unvisited_list.remove(current_node)

				if len(unvisited_list) > 0:
					# Select the unvisited node that is marked with the
					# smallest tentative distance, set it as the new
					# "current node", and go back to the beginning of the loop.
					current_node = sorted(unvisited_list, key=cost_for_node)[0]

			if print_path_costs:
				print 'Path costs for each maze_first_explored space within the maze:'
				self.print_maze_consol((0,0), 'up', path_costs)

			optimal_path.append(target)
			current_node = target
			# Construct the optimal path by following the gradient of path costs
			# from the goal to the start.
			while start not in optimal_path:
				current_node = sorted(graph[current_node], key=cost_for_node)[0]
				optimal_path = [current_node] + optimal_path

		return optimal_path

	def convert_path_to_steps(self, path, initial_heading):
		'''
		Convert the given path to a list of step instructions
		(rotation, movement) given the initial heading.
		'''
		steps = []
		if len(path) > 0:
			start = path.pop(0)
			heading = initial_heading
			steps = []
			deltas = self.convert_path_to_deltas_max_3(start, path)
			for delta_x, delta_y in deltas:
				up    = heading  == 'up'
				right = heading  == 'right'
				down  = heading  == 'down'
				left  = heading  == 'left'
				rotation = 0
				if ((up and delta_y < 0) or (right and delta_x < 0) or
						(down and delta_y > 0) or (left and delta_x > 0)):
					movement = -max(abs(delta_x), abs(delta_y))
				else:
					if delta_y == 0:
						if delta_x > 0:
							if up:
								rotation = 90
							elif down:
								rotation = -90
						else:
							if up:
								rotation = -90
							elif down:
								rotation = 90
					else:
						if delta_y > 0:
							if left:
								rotation = 90
							elif right:
								rotation = -90
						else:
							if left:
								rotation = -90
							elif right:
								rotation = 90
					movement = max(abs(delta_x), abs(delta_y))
				steps.append((rotation, movement))
				heading = self.calculate_heading(heading, rotation)

		return steps

	def convert_path_to_deltas_max_3(self, start, path):
		'''
		Break down the path to the x/y difference between each node in the
		path with a maximum chnage of 3. This will ensure that maximum movement
		made by the Robot while navigating path is not exceeded.
		'''
		x, y = start
		deltas = []
		for node_x, node_y in path:
			if y == node_y:
				step = node_x - x
				while step > 3 or step < -3:
					if step > 0:
						deltas.append((3,0))
						step -= 3
					else:
						deltas.append((-3,0))
						step += 3
				deltas.append((step,0))
			else:
				step = node_y - y
				while step > 3 or step < -3:
					if step > 0:
						deltas.append((0,3))
						step -= 3
					else:
						deltas.append((0,-3))
						step += 3
				deltas.append((0,step))

			x, y = node_x, node_y
		return deltas

	def found_optimal_path(self):
		'''
		Determine whether the optimal path through the maze has been found.
		If this is the first time the optimal path has been found, save it.
		'''
		if not self.goal_found():
			return False
		goal_location = self.goal_location

		# print "Goal Location is: {}".format(goal_location)

		if self.optimal_path != None:
			return True

		known_maze_graph = self.build_graph_from_maze(True, True)
		if goal_location not in known_maze_graph.keys():
			print "Goal not yet navigable!"
			return False

		open_maze_graph = self.build_graph_from_maze(True, False)

		# Compare the best path through the maze assuming all unknown walls
		# are walls vs all unknown walls are opennings. If the path lengths are
		# the same, the optimal path has been found.
		shortest_known_path = self.Dijkstra_best_path(known_maze_graph,
											self.initial_location, goal_location)
		shortest_possible_path = self.Dijkstra_best_path(open_maze_graph,
											self.initial_location, goal_location)
		optimal_path_found = len(shortest_known_path) == len(shortest_possible_path)

		if optimal_path_found:
			self.optimal_path = shortest_known_path
			self.optimal_steps = self.convert_path_to_steps(self.optimal_path, 'up')
		return optimal_path_found

	def print_maze_with_path_costs(self):
		'''
		Print the maze_first_explored map including the path costs for each maze_first_explored cell.
		'''
		if not self.goal_found():
			print "Can not print maze with path costs. The goal has not been found."
			return False
		known_maze_graph = self.build_graph_from_maze(True, True)
		self.Dijkstra_best_path(known_maze_graph, self.initial_location,
												self.goal_location, True)

	# Navigation utility methods:

	def distance_between_nodes(self, a, b):
		''' Return the distance between the two given nodes. '''
		xa, ya = a
		xb, yb = b
		
		return math.hypot(xb-xa, yb-ya)

	def calculate_node(self, location, heading, instructions):
		'''
		Given a location and heading, determine which node a set of instructions
		would lead to.
		'''
		rotation, movement = instructions
		x, y  = location
		up, right, down, left, ccw, fwd, cw = self.heading_rotation_bools(heading, rotation)
		if (up and ccw) or (down and cw) or (left and fwd):
			x -= movement
		elif (up and fwd) or (right and ccw) or (left and cw):
			y += movement
		elif (up and cw) or (right and fwd) or (down and ccw) :
			x += movement
		elif (right and cw) or (down and fwd) or (left and ccw) :
			y -= movement

		return (x, y)

	def calculate_heading(self, heading, rotation):
		'''
		Given a heading and rotation, wwhat would the new heading be if the
		rotation was made?
		'''
		up, right, down, left, ccw, fwd, cw = self.heading_rotation_bools(heading, rotation)
		if fwd:
			return heading
		if (ccw and up) or (cw and down):
			return 'left'
		if (ccw and right) or (cw and left):
			return 'up'
		if (ccw and down) or (cw and up):
			return 'right'
		if (ccw and left) or (cw and right):
			return 'down'

	def heading_rotation_bools(self, heading, rotation):
		'''
		Convert the heading and rotation values to booleans.
		'''
		up    = heading  == 'up'
		right = heading  == 'right'
		down  = heading  == 'down'
		left  = heading  == 'left'
		# counterclockwise
		ccw   = rotation == -90
		# forward
		fwd   = rotation == 0
		# clockwise
		cw    = rotation == 90
		return up, right, down, left, ccw, fwd, cw
		
	def goal_found(self):
		''' Has the goal been found? '''
		return self.goal_location != None

	def initial_walls(self, maze_dim):
		'''
		Construct the initial state for the two-dimensional list that
		represents all of the walls in the maze including exterior walls.
		-1 = unknown
		0  = no wall
		1  = wall

		NB: The nature of storing both horizontal and vertical walls in the same
		two-dimensional list results in slightly unconventional indexing. For
		a given maze_dim, there will be 2 * maze_dim + 1 sets of walls: the
		vertical left exterior walls, maze_dim sets of horizontal walls
		(interior and exterior), maze_dim - 1 sets of interior vertical walls,
		and the right vertical extreior walls. This also results in the
		following additional quirk: lists representing vertical walls will
		have length maze_dim while lists representing horizontal walls will
		have length maze_dim + 1 (because of the exterior walls top and bottom).
		'''
		walls = [[-1] * maze_dim]
		for i in range(maze_dim):
			walls.append([-1] * (maze_dim + 1))
			walls.append([-1] * maze_dim)

		for i in range(2 * maze_dim + 1):
			for j in range(len(walls[i])):
				# One of the 4 sets of exterior walls?
				top    = i % 2 == 1 and j == len(walls[i]) - 1
				right  = i == maze_dim * 2
				bottom = i % 2 == 1 and j == 0
				left   = i == 0
				# One of the four openings interior to the goal area?
				goal_top    = i == maze_dim and j == maze_dim / 2
				goal_right  = i == maze_dim + 1 and j == maze_dim / 2
				goal_bottom = i == maze_dim and j == maze_dim / 2 - 1
				goal_left   = i == maze_dim - 1 and j == maze_dim / 2
				if top or right or bottom or left:
					walls[i][j] = 1
				elif goal_top or goal_right or goal_bottom or goal_left:
					walls[i][j] = 0

		return walls

	def initial_cell_possibilities(self, maze_dim):
		'''
		Construct the initial state of the two-dimensional list that represents
		the number of possibile shapes each cell could take. It is worth noting
		that corner, edge, and center cells have fewer initial possible shapes
		than other interior cells given the necasary structure of any maze
		(every maze is fully enclosed and the 2x2 goal area at the center of
		the maze has no interior walls).
		'''
		dim = maze_dim
		cell_possibilities = []
		for n in range(dim):
			cell_possibilities.append([15] * dim)

		for i in range(dim):
			for j in range(dim):
				# Is the cell an edge or corner?
				top    = j == dim - 1
				right  = i == dim - 1
				bottom = j == 0
				left   = i == 0
				# Is the cell inside the goal area?
				left_top     = i == dim / 2 - 1 and j == dim / 2
				right_top    = i == dim / 2 and j == dim / 2
				right_bottom = i == dim / 2 and j == dim / 2 - 1
				left_bottom  = i == dim / 2 - 1 and j == dim / 2 - 1
				if top or bottom:
					#                          corner                  edge
					cell_possibilities[i][j] = 2 if left or right else 7
				elif left or right:
					# edge
					cell_possibilities[i][j] = 7
				elif left_top or right_top or right_bottom or left_bottom:
					# goal area
					cell_possibilities[i][j] = 3

		return cell_possibilities

	def update_wall_knowledge(self, location, heading, sensors):
		'''
		Update map of walls and opennings based on current location, heading,
		and sensor readings.
		'''
		x, y = location
		left_reading, forward_reading, right_reading = sensors

		if heading == 'up':
			for i in range(left_reading + 1):
				wall_value = 1 if i == left_reading else 0
				self.walls[2 * (x - i)][y] = wall_value
			for j in range(forward_reading + 1):
				wall_value = 1 if j == forward_reading else 0
				self.walls[2 * x + 1][y + 1 + j] = wall_value
			for k in range(right_reading + 1):
				wall_value = 1 if k == right_reading else 0
				self.walls[2 * (x + 1 + k)][y] = wall_value

		elif heading == 'right':
			for i in range(left_reading + 1):
				wall_value = 1 if i == left_reading else 0
				self.walls[2 * x + 1][y + 1 + i] = wall_value
			for j in range(forward_reading + 1):
				wall_value = 1 if j == forward_reading else 0
				self.walls[2 * (x + 1 + j)][y] = wall_value
			for k in range(right_reading + 1):
				wall_value = 1 if k == right_reading else 0
				self.walls[2 * x + 1][y - k] = wall_value

		elif heading == 'down':
			for i in range(left_reading + 1):
				wall_value = 1 if i == left_reading else 0
				self.walls[2 * (x + 1 + i)][y] = wall_value
			for j in range(forward_reading + 1):
				wall_value = 1 if j == forward_reading else 0
				self.walls[2 * x + 1][y - j] = wall_value
			for k in range(right_reading + 1):
				wall_value = 1 if k == right_reading else 0
				self.walls[2 * (x - k)][y] = wall_value

		elif heading == 'left':
			for i in range(left_reading + 1):
				wall_value = 1 if i == left_reading else 0
				self.walls[2 * x + 1][y - i] = wall_value
			for j in range(forward_reading + 1):
				wall_value = 1 if j == forward_reading else 0
				self.walls[2 * (x - j)][y] = wall_value
			for k in range(right_reading + 1):
				wall_value = 1 if k == right_reading else 0
				self.walls[2 * x + 1][y + 1 + k] = wall_value

		self.update_cell_possibilities()

	def update_cell_possibilities(self):
		'''
		Given the current knowledge of the maze walls, update the
		two-dimensional list of possible cell shapes taking advantage
		of the knowledge that every cell must have at most 3 walls among its
		four sides.
		'''
		for i in range(self.maze_dim):
			for j in range(self.maze_dim):
				top    = self.walls[2 * i + 1][j + 1]
				right  = self.walls[2 * i + 2][j    ]
				bottom = self.walls[2 * i + 1][j    ]
				left   = self.walls[2 * i    ][j    ]
				wall_values = [top, right, bottom, left]

				top_unknown    = 1 if top    == -1 else 0
				right_unknown  = 1 if right  == -1 else 0
				bottom_unknown = 1 if bottom == -1 else 0
				left_unknown   = 1 if left   == -1 else 0
				num_unknown    = (top_unknown + right_unknown +
								  bottom_unknown + left_unknown)

				# If the robot knows that a space is srrounded by three walls
				# but doesn't know about the 4th, then the 4th must be an
				# openning.
				if num_unknown == 1 and sum(wall_values) == 2:
					self.cell_possibilities[i][j] = 1
					if top == -1:
						self.walls[2 * i + 1][j + 1] = 0
					elif right == -1:
						self.walls[2 * i + 2][j] = 0
					elif bottom == -1:
						self.walls[2 * i + 1][j] = 0
					elif left == -1:
						self.walls[2 * i][j] = 0
				else:
					self.cell_possibilities[i][j] = 2 ** num_unknown
					if 0 not in wall_values:
						self.cell_possibilities[i][j] -= 1

		self.check_goal_walls()

	def check_goal_walls(self):
		'''
		Check to see if the goal entrance has been discovered. If either the
		goal opening has been found or all 7 goal walls have been found then
		the remaining wall locations or opening can be inferred respectively.
		'''
		if self.goal_location != None:
			return
		dim = self.maze_dim

		vals = []
		for i, j in self.goal_wall_coords:
			vals.append(self.walls[i][j])

		if 0 in vals:
			# The goal openning has been found.
			opening_index = vals.index(0)
		elif len([x for x in vals if x != -1]) == 7:
			# All 7 walls surrounding the goal have been found.
			opening_index = vals.index(-1)
		else:
			return

		if opening_index in [0,1]:
			self.goal_location = (dim / 2, dim / 2)
		elif opening_index in [2,3]:
			self.goal_location = (dim / 2, dim / 2 - 1)
		elif opening_index in [4,5]:
			self.goal_location = (dim / 2 - 1, dim / 2 - 1)
		elif opening_index in [6,7]:
			self.goal_location = (dim / 2 - 1, dim / 2)

		for k in range(len(self.goal_wall_coords)):
			i, j = self.goal_wall_coords[k]
			if k == opening_index:
				self.walls[i][j] = 0
			else:
				self.walls[i][j] = 1		
	
	def print_maze_consol(self, location, heading, cell_values = None):
		'''
		Print the maze in consol.
		'''
		x, y = location
		
		vertical_wall_chars    = {-1: ':',  0: ' ',  1: '|'}
		horizontal_wall_chars  = {-1: '..', 0: '  ', 1: '--'}

		maze_rows = [''] * (2 * self.maze_dim + 1)
		for i in range(2 * self.maze_dim + 1):
			for j in range(len(self.walls[i])):
				if i % 2 == 0:
					# vertical walls
					maze_rows[2*j] += '+'
					maze_rows[2*j + 1] += vertical_wall_chars[self.walls[i][j]]
				else:
					# horizontal walls
					maze_rows[2*j] += horizontal_wall_chars[self.walls[i][j]]
					if 2*j + 1 < len(maze_rows):
						if cell_values == None:
							cell_value = '  '
							if (i - 1) / 2 == x and j == y:
									cell_value = 'xx'
						else:
							loc = ((i - 1) / 2, j)
							if loc in cell_values.keys():
								cell_value = str(cell_values[loc])
								if len(cell_value) == 1:
									cell_value = ' ' + cell_value
							else:
								cell_value = '??'
						maze_rows[2*j + 1] += cell_value
			if i % 2 == 0:
				maze_rows[-1] += '*'
		maze_rows.reverse()
		maze_drawing = ''
		for row in maze_rows:
			maze_drawing += row + "\n"
		print maze_drawing	

	def next_move(self, sensors):
		'''
		Use this function to determine the next move the robot should make,
		based on the input from the sensors after its previous move. Sensor
		inputs are a list of three distances from the robot's left, front, and
		right-facing sensors, in that order.

		Outputs should be a tuple of two values. The first value indicates
		robot rotation (if any), as a number: 0 for no rotation, +90 for a
		90-degree rotation clockwise, and -90 for a 90-degree rotation
		counterclockwise. Other values will result in no rotation. The second
		value indicates robot movement, and the robot will attempt to move the
		number of indicated squares: a positive number indicates forwards
		movement, while a negative number indicates backwards movement. The
		robot may move a maximum of three units per turn. Any excess movement
		is ignored.

		If the robot wants to end a run (e.g. during the first training run in
		the maze) then returing the tuple ('Reset', 'Reset') will indicate to
		the tester to end the run and return the robot to the start.
		'''

		rotation = 0
		movement = 0

		if self.racing:
			rotation, movement = self.optimal_steps[self.race_time_step]
			self.race_time_step += 1
			return rotation, movement
		'''
		Save the latest sensor readings and the mapper to update its knowledge
		about the maze walls.
		'''
		self.latest_sensor_reading = sensors
		self.update_wall_knowledge(self.location, self.heading, sensors)

		if self.goal_visited and self.found_optimal_path():
			self.racing = True
			#self.draw_solution_path(self.optimal_path)
			print "Best number of steps: {}\nbest path{}".format(len(self.optimal_steps),self.optimal_path)
			return 'Reset', 'Reset'

		return self.maze_first_explore()
		
		#return rotation, movement