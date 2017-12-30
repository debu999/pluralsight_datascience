# -*- coding: utf-8 -*-

# Import Section
from dotenv import load_dotenv, find_dotenv
import os
import requests
from requests import session as ss
from cryptography.fernet import Fernet
import ast, logging

# Get Params
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

auth_key = ast.literal_eval(os.environ.get("AUTH_KEY"))
enc_pwd = ast.literal_eval(os.environ.get("ENC_PWD"))
user_auth_key = ast.literal_eval(os.environ.get("USER_AUTH_KEY"))
enc_uid = ast.literal_eval(os.environ.get("ENC_UID"))

cp_ak = Fernet(auth_key)
cp_uak = Fernet(user_auth_key)

# Functions and other variables
payload = {
    "action":"login",
    "username": cp_uak.decrypt(enc_uid),
    "password": cp_ak.decrypt(enc_pwd)
}

url_login = """https://www.kaggle.com/account/login"""

def extract_data(url, file_path):
    """Extract data from Kaggle."""
    
    with ss() as c:
        c.post(url_login,data=payload)
        with open(file_path,'wb') as fl_handle:

            res = c.get(url, stream=True)
            for block in res.iter_content(1024):
                fl_handle.write(block)
                

def main(proj_dir):
    """
    Main Method.
    """
    
    # get logger
    logger = logging.getLogger(__name__)
    logger.info("Getting Raw Data from Kaggle using Session Authentication.")
    
    # Fetch Data from Kaggle Invoke Funcitons
    url_train = "https://www.kaggle.com/c/titanic/download/train.csv"
    url_test = """https://www.kaggle.com/c/titanic/download/test.csv"""

    raw_data_path = os.path.join(proj_dir,'data','raw')
    train_data_path = os.path.join(raw_data_path,"train.csv")
    test_data_path = os.path.join(raw_data_path,"test.csv")

    # Extract Data and store in CSV
    extract_data(url_train, train_data_path)
    extract_data(url_test, test_data_path)
    logger.info("Data from Kaggle successfully downloaded.")
    
    os.listdir(raw_data_path)

if __name__ == "__main__":
    proj_dir = os.path.join(os.path.dirname(__file__),os.path.pardir,os.path.pardir)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(proj_dir)