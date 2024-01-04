"""

@author: Jinal Shah

This script will create my database
and the table. Furthermore, I will 
have functions in this script to work with 
the database.

Database Schema:
Table essays:
RowID -> Int, Primary Key (Surrogate Key)
Prompt -> VarChar(250), the essay prompts, can be empty.
Essay -> VarChar(10000), the essay
WordCount -> Int, how many words are in the essay.
Generated -> Int, 0 or 1 depending on if the essay was written by a student or an LLM
"""
import mysql.connector
from mysql.connector import Error
import sys

# Appending the credentials path
sys.path.append('../')
from credentials import credentials

# Function to connect to mySQL Server
def connectToServer(host_name: str, username: str, pwd: str):
    """
    connectToServer

    A function that connects to the mySQL Server

    inputs:
    host -> a string representing the host
    user -> a string representing the user
    pwd -> a string representing the pwd

    outputs:
    connection -> a variable representing the connection to the server
    """
    connection = None

    # Trying to connect to the server
    try:
        connection = mysql.connector.connect(
            user=username,
            password=pwd,
            host=host_name
        )
        # Printing here to make sure that the server was connected to properly
        print('Connected to Server!')
    except Error as err:
        print('Error Occured in Connecting to Database:')
        print(f'{err}')
    
    # Returning the connection
    return connection

# A function to create the database
def create_database(server_connection, database_name: str):
    """
    create_database

    A function to create the database

    inputs:
    server_connection -> an object representing the connection to the server
    database_name -> a string for the database name

    outputs:
    None
    """
    cursor = server_connection.cursor()
    query = f'CREATE DATABASE {database_name}'

    # Executing the query
    try:
        cursor.execute(query)
        print('Database created successfully')
    except Error as err:
        print('Problem in Database Creation:')
        print(err)

# A function to connect to the database
def create_db_connection(host_name:str, user_name:str, password:str, db_name:str):
    """
    create_db_connection

    A function to connect to the database

    inputs:
    - host_name: string that describes the host
    - user_name: string for the username
    - password: string for the password
    - db_name: string for the database name

    outputs:
    - connection: an object establishing connection to the database.
    """
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=password,
            database=db_name
        )
        print(f"Connected to {db_name} Successfully")
    except Error as err:
        print('Error in Database Connection:')
        print(f"Error: '{err}'")
    return connection

# A function to execute queries
def modify_queries(query:str,connection):
    """
    modify_query

    A function to execute queries that modify the database

    inputs:
    - query: a string with the SQL query
    - connection: an object for the connection to the server
    
    outputs:
    - None
    """
    cursor = connection.cursor()

    # Trying to execute the query
    try:
        cursor.execute(query)
        connection.commit()
        print('Query ran')
    except Error as err:
        print('Query caused an error:')
        print(err)

# A function to other queries such as searching, etc
def querying(query:str, connection):
    """
    querying

    A function to query the database. 
    This function differs from modify_queries
    in the sense that modify_queries modifies
    the table. This one doesn't.

    inputs:
    - query: A string representing the query
    - connection: A connection to the database

    outputs:
    - result: the results of the query
    """
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")

# Main Method
if __name__ == '__main__':
    # Setting up the connection to the server
    server_connection = connectToServer(host_name=credentials['host'],
                                        username=credentials['user'],pwd=credentials['password'])

    # Creating the database
    create_database(server_connection,'AuthenticAI')

    # Create connection to database
    database_connection = create_db_connection(credentials['host'],
                                        credentials['user'],credentials['password'],'AuthenticAI')
    
    # Creating the table 
    create_table = """
    CREATE TABLE essays (
        row_id INT PRIMARY KEY AUTO_INCREMENT,
        prompt VARCHAR(250),
        essay VARCHAR(5000) NOT NULL,
        word_count INT NOT NULL,
        LLM_written INT NOT NULL
        );
    """
    modify_queries(create_table,database_connection)
    
    # Closing database connection
    database_connection.close()

    # Closing server connection
    server_connection.close()
