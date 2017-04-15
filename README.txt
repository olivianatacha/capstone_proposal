===================================================================================
                          
===================================================================================

    .-._______                                                       _______.-.
---( )_)______)                                                     (______(_( )---
  (    ()___)                                                         (___()    )
       ()__)            Plot and Navigate a Virtual Maze               (__()
--(___()_)                                                               (_()___)--


===================================================================================
                             CAPSTONE PROJECT
===================================================================================

--------------------------------------oOOo----oOOo---------------------------------
							DESCRIPTION OF SUBMITTED FILES
-----------------------------------------------------------------------------------
			1-maze.py : the class describing a maze with its attributes
			and functions
			
			2-showmaze.py : The class to visualise the using turtle
			library
			
			3-robot.py : the class that we edited in which the robot is
			describe with its attributes, functions
			
			4-tester.py : The main class that uses all other in order to 
			test the robot in the maze.
			
			5-test_maze_01,02,03.txt : provided 12x12,14x14,16x16 dimension maze 
			for tests
			
			6-result_test_maze_01,02,03.txt : result of testing the 
			robot respectivly in test_maze_01, test_maze_02, test_maze_03
			
			7-tester_several_maze.py : an adapted copy of tester.py to 
			Successively run the maze in several given maze of same dimension
			
			8-maze_nxn : directory containing 100 random mazes of size nxn

--------------------------------------oOOo----oOOo---------------------------------					
							how to run the robot?
-----------------------------------------------------------------------------------
			1- for testing
				- first, open the terminal and direct it into the project directory
				-if you wish to visualise in the terminal :
					- run the Windows command "python tester.py test_maze_**.txt",
					where ** is one 01,02 or 03.
				else if you prefer it result in a file:
					run the Windows command "python tester.py test_maze_**.txt >> output_file_name.txt",
					where ** is one 01,02 or 03.
			2- for generalising : 
				- first, open the terminal and direct it into the project directory
				- run the Windows command "python tester_several_maze.py maze_nxn
					where nxn 12x12 or 14x14 or 16x16. Then open the out.txt file that is located in
					the maze_nxn folder you tested to see scores.