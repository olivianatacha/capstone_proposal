# Maze.py
# Last update 20031115-214011

"""
implements class Maze

Three public methods are implemented:
  __init__(rows,cols)
  __str__()
  as_html_table()

Usage:
  maze = Maze( 20, 30 )  # create a maze 20 rows x 30 cols
  print maze             # print out the maze
  print "<html><body>%s</body></html>" % maze.as_html_table() # publish it

To do:
  1. Method find_path() :)
  2. Different algorithms for big mazes (>50x50) or iteration instead of recursion
"""

# Translation to Python (C) 2003 Georgy Pruss

# From http://www.siteexperts.com/tips/functions/ts20/mm.asp
# // Copyright 1999 Rajeev Hariharan. All Rights Reserved.

from __future__ import print_function
import random, sys
from random import randint


# Constants -- cell marks
BOTTOMWALL = 0
RIGHTWALL  = 1
VISITED    = 2

NORTH = 0
SOUTH = 1
WEST  = 2
EAST  = 3


class Maze:

  """Creates a maze and formattes it as text or HTML"""


  #*****************************************

  def __init__( self, n_rows, n_cols ):
    """Create a maze with given number of rows and cols.
    The path connects upper left and lower right cells.
    Actually, all cells are connected.
    Can raise 'MemoryError: Stack overflow' for big arguments, e.g. 100,100
    """

    self.n_rows = n_rows
    self.n_cols = n_cols
    self.maze = [None]*n_rows

    # Set up the hedge array - initially all walls intact
    for i in range(n_rows):
      self.maze[i] = [None]*n_cols
      for j in range(n_cols):
        self.maze[i][j] = [True,True,False] # BOTTOMWALL,RIGHTWALL,VISITED

    # Choose a random starting point
    currCol = 0
    currRow = 0
    #currCol = random.randrange(n_cols)
    #currRow = random.randrange(n_rows)

    # The searh can be quite deep
    if n_rows*n_cols > sys.getrecursionlimit():
      sys.setrecursionlimit( n_rows*n_cols+5 )

    # Recursively Remove Walls - Depth First Algorithm
    self._make_path( currRow, currCol )


  #*****************************************

  def _make_path( self, R, C, D=None ):

    maze = self.maze # speed up a bit

    # Knock out wall between this and previous cell
    maze[R][C][VISITED] = True;

    if   D==NORTH: maze[R]  [C]  [BOTTOMWALL] = False
    elif D==SOUTH: maze[R-1][C]  [BOTTOMWALL] = False
    elif D==WEST:  maze[R]  [C]  [RIGHTWALL]  = False
    elif D==EAST:  maze[R]  [C-1][RIGHTWALL]  = False

    # Build legal directions array
    directions = []
    if R>0            : directions.append(NORTH)
    if R<self.n_rows-1: directions.append(SOUTH)
    if C>0            : directions.append(WEST)
    if C<self.n_cols-1: directions.append(EAST)

    # Shuffle the directions array randomly
    dir_len = len(directions)
    for i in range(dir_len):
      j = random.randrange(dir_len)
      directions[i],directions[j] = directions[j],directions[i]

    # Call the function recursively
    for dir in directions:
      if dir==NORTH:
        if not maze[R-1][C][VISITED]:
          self._make_path( R-1,C,NORTH )
      elif dir==SOUTH:
        if not maze[R+1][C][VISITED]:
          self._make_path( R+1,C,SOUTH )
      elif dir==EAST:
        if not maze[R][C+1][VISITED]:
          self._make_path( R,C+1,EAST )
      elif dir==WEST:
        if not maze[R][C-1][VISITED]:
          self._make_path( R,C-1,WEST )
      #else: raise 'bug:you should never reach here'


  #*****************************************

  def __str__(self):
    """Return maze table in ASCII"""

    result = '.' + self.n_cols*'_.'
    result += '\n'

    for i in range(self.n_rows):
      result += '|'

      for j in range(self.n_cols):
        if i==self.n_rows-1 or self.maze[i][j][BOTTOMWALL]:
          result += '_'
        else:
          result += ' '
        if j==self.n_cols-1 or self.maze[i][j][RIGHTWALL]:
          result += '|'
        else:
          result += '.'

      result += '\n'

    return result


  #*****************************************

def convert_maze_from_graphiq_to_file(maze_table,maze_dim):
  maze_as_liste = []
  """
  for a simple use, convert the maze to list
  """
  for k in range (maze_dim+1):
    maze_as_liste.append(list(maze_table[k]))
    #print (list(maze_table[k]))

  #print (maze_as_liste)
  result = ""+str(maze_dim)+"\n"
  for j in range(maze_dim*2):
    if(j%2 == 1):
      row = ""
      left_wal = botom_wal = right_wal = top_wal = 1
      for i in range (maze_dim+1):
        
        if i > 0:
          #print ("odd")
          if maze_as_liste[i][j-1] == '|' :
            left_wal = 0
          if maze_as_liste[i][j-1] == '.' :
            left_wal = 1
          
          if maze_as_liste[i][j] == '_' :
            botom_wal = 0
          if maze_as_liste[i][j] == ' ' :
            botom_wal = 1

          if maze_as_liste[i][j+1] == '|':
            right_wal = 0
          if maze_as_liste[i][j+1] == '.':
            right_wal = 1

          if maze_as_liste[i-1][j] == '_':
            top_wal = 0
          if maze_as_liste[i-1][j] == ' ':
            top_wal = 1
          number = left_wal*8 + botom_wal*4 + right_wal*2 + top_wal*1

          if i==1 and j == 1:
            row = row + "1,"
          else :
            if i < maze_dim :
              row = row + str(number) + ","
            elif i == maze_dim :
              row = row + str(number)
      #print (row)
      result += row + "\n"

  return result


#*****************************************

if __name__ == "__main__":

  syntax = ( "Syntax: %s [[-]n_rows [n_cols]]\n\n"
             "If n_cols is missing, it will be the same as n_rows\n"
             "If n_rows is negative, html representation will be generated\n\n"
             "Examples:\n%s 50 39 -- text maze 50 rows by 39 columns\n"
             "%s -40   -- html code of 40 x 40 cell maze"
           )

  # parse arguments if any

  import os.path

  argc = len(sys.argv)
  name = os.path.basename( sys.argv[0] )

  if argc not in (2,3):
    print >>sys.stderr, syntax % (name,name,name)
    sys.exit(1)
  
  elif argc == 2:
    n_rows = n_cols = int(sys.argv[1])

  elif argc == 3:
    n_rows = int(sys.argv[1])
    n_cols = int(sys.argv[2])

  # create and print maze

  try:
    if n_rows > 0:
      for maze_counter in range (100):
        maze_t = Maze( n_rows, n_cols )
        maze_text = maze_t.__str__()
        #print (maze_text)
        center_coordinate = [(n_rows*2,n_rows*2),(n_rows*2-1,n_rows*2+1),(n_rows*2+1,n_rows*2+1),(n_rows*2-2,n_rows*2),(n_rows*2-3,n_rows*2+1)]
        maze_table = maze_text.splitlines()

        open_wall_center = randint(1,6)

        maze_table[n_rows/2-1] = maze_table[n_rows/2-1][:n_rows-1]+'_'+maze_table[n_rows/2-1][n_rows:] #top left wall
        maze_table[n_rows/2-1] = maze_table[n_rows/2-1][:n_rows+1]+'_'+maze_table[n_rows/2-1][n_rows+2:] #top right wall
        maze_table[n_rows/2] = maze_table[n_rows/2][:n_rows-2]+'|'+maze_table[n_rows/2][n_rows-1:] #left top wall
        maze_table[n_rows/2] = maze_table[n_rows/2][:n_rows+2]+'|'+maze_table[n_rows/2][n_rows+3:] #right top wall
        maze_table[n_rows/2+1] = maze_table[n_rows/2+1][:n_rows-2]+'|'+maze_table[n_rows/2+1][n_rows-1:] #left down wall
        maze_table[n_rows/2+1] = maze_table[n_rows/2+1][:n_rows+2]+'|'+maze_table[n_rows/2+1][n_rows+3:] #right down wall

        if open_wall_center == 1:
          maze_table[n_rows/2-1] = maze_table[n_rows/2-1][:n_rows-1]+' '+maze_table[n_rows/2-1][n_rows:] #top left wall
        if open_wall_center == 2:
          maze_table[n_rows/2-1] = maze_table[n_rows/2-1][:n_rows+1]+' '+maze_table[n_rows/2-1][n_rows+2:] #top right wall
        if open_wall_center == 3:
          maze_table[n_rows/2] = maze_table[n_rows/2][:n_rows-2]+' '+maze_table[n_rows/2][n_rows-1:] #left top wall
        if open_wall_center == 4:
          maze_table[n_rows/2] = maze_table[n_rows/2][:n_rows+2]+' '+maze_table[n_rows/2][n_rows+3:] #right top wall
        if open_wall_center == 5:
          maze_table[n_rows/2+1] = maze_table[n_rows/2+1][:n_rows-2]+' '+maze_table[n_rows/2+1][n_rows-1:] #left down wall
        if open_wall_center == 6:
          maze_table[n_rows/2+1] = maze_table[n_rows/2+1][:n_rows+2]+' '+maze_table[n_rows/2+1][n_rows+3:] #right down wall


        maze_table[n_rows/2-1] = maze_table[n_rows/2-1][:n_rows]+'.'+maze_table[n_rows/2-1][n_rows+1:]

        maze_table[n_rows/2] = maze_table[n_rows/2][:n_rows-1]+' '+maze_table[n_rows/2][n_rows:] 
        maze_table[n_rows/2] = maze_table[n_rows/2][:n_rows]+' '+maze_table[n_rows/2][n_rows+1:]
        maze_table[n_rows/2] = maze_table[n_rows/2][:n_rows+1]+' '+maze_table[n_rows/2][n_rows+2:]
        

        
        maze_table[n_rows/2+1] = maze_table[n_rows/2+1][:n_rows-1]+'_'+maze_table[n_rows/2+1][n_rows:]
        maze_table[n_rows/2+1] = maze_table[n_rows/2+1][:n_rows]+'.'+maze_table[n_rows/2+1][n_rows+1:]
        maze_table[n_rows/2+1] = maze_table[n_rows/2+1][:n_rows+1]+'_'+maze_table[n_rows/2+1][n_rows+2:]

        
        for i in range (n_rows+1):
          print (maze_table[i])

        converted_maze = convert_maze_from_graphiq_to_file(maze_table,n_rows)
        if maze_counter == 0:
          os.chdir("maze_"+str(n_rows)+"x"+str(n_rows))
        maze_file = open("random_maze_"+str(maze_counter)+"_"+str(n_rows)+"x"+str(n_rows)+".txt", "a")
        print (converted_maze,end="",file=maze_file)

    else:
      print ("bad number")
  except MemoryError:
    print ("Sorry, n_rows, n_cols were too big")


# EOF