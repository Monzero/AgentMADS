import requests
import os
import pandas as pd
import subprocess
import json
import ollama 
import re
from datetime import datetime
model = 'llamm3'

############################################################################################
#############           Defining custom functions                     #####################
############################################################################################


def upload_all_source(path, upload_everything_afresh = False):
    if not upload_everything_afresh and os.path.exists(path + '/96_results/all_source_id.csv'):
        print('Loading the source ids from the file')
        df_ids = pd.read_csv(path + '/96_results/all_source_id.csv')
        ids    = dict(zip(df_ids['source'], df_ids['source_id']))   
    else:
        ids        = create_source_id(path)
        df_ids     = pd.DataFrame(ids.items(), columns=['source', 'source_id'])
        df_ids.to_csv(path + '96_results/all_source_id.csv')
        print('Source ids are saved in the file')
    return(ids)

def create_source_id_url(path):
    """
    This function will upload the pdf from URL to chatpdf and return the source id for each file

    Args:
        path (_type_): patth is the common folder and it refers to the csv with all links and name of the source id
    """
    if not os.path.exists(path + '97_static/source_url.csv'):
        print('No source url file found')
        return
    
    df = pd.read_csv(path + '97_static/source_url.csv')
    ids = dict()
    
    headers = {
    'x-api-key'    : 'sec_c3FrcK3DSUfwnXo9M3OtqtS467NO3KOe',
    'Content-Type' : 'application/json'
    }
    
    for _, row in df.iterrows():
        source_id_name = row['source']
        url_link       = row['url']
    
        data = {'url'  : url_link}

        response = requests.post(
            'https://api.chatpdf.com/v1/sources/add-url', headers=headers, json=data)

        if response.status_code == 200:
            #print('Source ID:', response.json()['sourceId'])
            source_id      = response.json()['sourceId']
            ids[source_id_name] = source_id
        else:
            print('Status:', response.status_code)
            print('Error:', response.text)
    return(ids)
               
def create_source_id_upload(path):
    """This function will upload all the files to chatpdf and return the source id for each file

    Args:
        path (str): here the folder path is needed where all the files are stored. We assume the same structure as the one in the repo. 
    """
    path        = path + '98_data/' 
    all_files   = os.listdir(path)   
    all_files   = [x for x in all_files if x.endswith('.pdf')]
    all_files   = [path + file for file in all_files]
    all_files = [x for x in all_files if x.endswith('.pdf')]
    
    if len(all_files) == 0:
        print('No files found')
        return
    
    ids         = dict()    

    for file in all_files:  
        files = [
            ('file', ('file', open(file, 'rb'), 'application/octet-stream'))
        ]
        headers = {
            'x-api-key': 'sec_c3FrcK3DSUfwnXo9M3OtqtS467NO3KOe'
        }

        response = requests.post(
            'https://api.chatpdf.com/v1/sources/add-file', headers=headers, files=files)

        if response.status_code == 200:
            #print('Source ID:', response.json()['sourceId'])
            source_id_name = file.split('/')[-1].split('.')[0]
            source_id      = response.json()['sourceId']
            ids[source_id_name] = source_id
        else:
            print('Status:', response.status_code)
            print('Error:', response.text)
    
    return(ids)

def create_source_id(path):
    ids_upload = create_source_id_upload(path)
    ids_url    = create_source_id_url(path)
    ids        = {**ids_upload, **ids_url}
    return(ids)
    
def hit_chatpdf_api(path, load_all_fresh = False):
    """This function will load all the promps questions and hit the chatpdf api and get the results for the prompts.
    """
    prompts        = pd.read_csv(path + '/97_static/prompts.csv')
    
    if not load_all_fresh and os.path.exists(path + '/96_results/prompts_result.csv'):
        prompts_result = pd.read_csv(path + '/96_results/prompts_result.csv')
        que_covered    = prompts_result['sr_no'].unique()
        que_list       = prompts['sr_no'].unique()
        remaining_que  = [x for x in que_list if x not in que_covered]
        prompts        = prompts[ prompts['sr_no'].isin(remaining_que)]
    
    
    headers = {
        'x-api-key': 'sec_c3FrcK3DSUfwnXo9M3OtqtS467NO3KOe',
        "Content-Type": "application/json",
    }

    results = pd.DataFrame(columns=['sr_no','cat','que_no', 'source', 'message', 'result'])
    
    for _, row in prompts.iterrows():
        source    = ids[row['source']]
        message   = row['message']
        que_no    = row['que_no']
        cat       = row['cat']
        sr_no     = row['sr_no']
        data = {
            'sourceId': source,
            'messages': [
                {
                    'role': "user",
                    'content': message,
                }
            ]
        }
        response = requests.post(
            'https://api.chatpdf.com/v1/chats/message', headers=headers, json=data)

        if response.status_code == 200:
            print('Result:', response.json()['content'])
            result = response.json().get('content', '')  # Extract result 
        else:
            print('Status:', response.status_code)
            print('Error:', response.text)
            
            # Append the new row to the results DataFrame
        new_row = pd.DataFrame({'sr_no':[sr_no],'cat':[cat],'que_no':[que_no],'source': [row['source']], 'message': [message], 'result': [result]})
        results = pd.concat([results, new_row], ignore_index=True) 
    
    if not load_all_fresh and os.path.exists(path + '/96_results/prompts_result.csv'):
        results = pd.concat([prompts_result, results], ignore_index=True)
    
    results.to_csv(path + '/96_results/prompts_result.csv',  index=False)    

def score_q17(path):
    """This function scores question 17 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 17
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely


    if not content.strip():
        print("No content available for question 17.")
        return

    # Define the scoring criteria
    scoring_criteria = (
        "Score 0 if you get the impression that the policies are not publicly disclosed and the company has not provided information on the number of employee accidents or there have been labour fatalities on account of accidents in the work place. "
        "Score 1 if you get the impression that The policies are publicly disclosed or the company has provided information on the number of employee accidents  "
        "Score 2 if you get the impression that The company has provided information on the number of employee accidents and has publicly disclosed its health and safety policies"
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"I will specify the criteria for scoring the content. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string)."
        f"Other than JSON format, any other format will not be accepted."
        f"Fetch as much source information as possible to score the content. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        print(result["message"]["content"])

        response_text = result["message"]["content"].strip()

        match    = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            response = json.loads(match.group())
        else:
            raise ValueError("Failed to extract JSON response from the output.")
        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question 17: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    except json.JSONDecodeError:
        print("Failed to parse response as JSON. Raw output:")
        print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'category':['Corporate Governance'],'que_no':[17],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q1(path):
    """This function scores question 1 based on predefined criteria.
    
    Args:
        path (str): Path to the folder for the company.
        
    Question is : Does the board have directors with permanent boaerd seats?
    """
    
    # Load the dataset
    question_no = 1
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered      = pr[pr['que_no'] == question_no]['result']
    content          = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    
    if pr_filtered.empty:
        print(f"No content available for question {question_no}.")
        return

    print(f"Content for question {question_no}: {content}")
    
    if not content.strip():
        print(f"No content available for question {question_no}.")
        return

    # Define the scoring criteria
    scoring_criteria = (
        "Score 0 if one or more than one directors are marked as permanent board members"
        "Score 1 if the directors which are marked as permanent board members, but those are representatives of lenders. Remember that this case is applicable for financially distressed companies. So unless it is mentioned explicitly that lenders have sent those board members as representative, do not assume so."
        "Score 2 if All directors are marked as non-permanent board members"
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"I will specify the criteria for scoring the content. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string)."
        f"Other than JSON format, any other format will not be accepted."
        f"Fetch as much source information as possible to score the content. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Do not assume any details. If you dont find any information then it is not available. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        print(result["message"]["content"])

        response_text = result["message"]["content"].strip()

        match    = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            response = json.loads(match.group())
        else:
            raise ValueError("Failed to extract JSON response from the output.")
        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    except json.JSONDecodeError:
        print("Failed to parse response as JSON. Raw output:")
        print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'category':['Corporate Governance'],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)


############################################################################################
#############                         Script                           #####################
############################################################################################

path       = '/Users/monilshah/Documents/02_NWU/01_capstone/02_Code/'

#############                   Get the source ids                      #####################
ids       = upload_all_source(path, upload_everything_afresh=False)

#############          Get answers for the questions                  #####################
hit_chatpdf_api(path, load_all_fresh=True)

#############                   Score question 17                      #####################
score_q17(path)

#############                   Score question 1                      #####################
score_q1(path)






