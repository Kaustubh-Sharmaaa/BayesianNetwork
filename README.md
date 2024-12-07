# BayesianNetwork

Name: Kaustubh Sharma
UTA ID: 1002138514

Programming Language Used: Python 3.8

Code Structure:
    The entire code is divided into smaller components, 
    The main function is responsible for the running of the entire application,
    - input_training_data() : responsible for reading the training data from the training file.
    - tally_counts() : calculates the frequency of the variables from the file.
    - computation() : calculates the conditional probability tables by making use of the output of the tally_counts function
    - calculate_joint_probability() : calculates the joint probability for the given variable values
    - sum_over_hidden() : calculates the sum of the probabilities over all the possible events
    - perform_query() : performs the query and returns the value.
    - interpret_query() : interprets the query by taking into consideration the input of the user
    - input_loop() : the main loop of the program that allows the user to input queries one after the other

How to run the code:
    Ensure that the training data file is saved in the same folder as the bnet.py file, and the name of the training data file is "training_data.txt".
command to run the code:
    python3 bnet.py

once the code is running, enter queries in the following format:
Sample Query:
    - Bt Gt given Ft (P(B = True, G = True)|P(F = True))
    - Bt Gt (P(B = True, G = True))
To exit the program, the user can enter "bye" as the query.