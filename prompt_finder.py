import requests
import os
import pandas as pd
import subprocess
import json
import ollama 
import re
from datetime import datetime
from google import genai
from google.genai import types
import httpx
import pathlib
import time
import subprocess
import shutil
import fitz
import logging


os.environ["GOOGLE_API_KEY"] = "AIzaSyBwnynresyZ58PSB7c5wbRu-4yPlbs3Pck"
CHATPDF_API_KEY = 'sec_YW8SGWsJb26OCRTFW0hOxJquFCNhtpeW'
client = genai.Client()

model_to_use = 'llama3'
#model_to_use = 'deepseek-R1'


############################################################################################
#############           Defining custom functions                     #####################
############################################################################################

def create_gemini_source_from_url(path):
    """
    This function will download the pdf from URL and save in preset folder

    Args:
        path (_type_): patth is the common folder and it refers to the csv with all links and name of the source id
    """
    if not os.path.exists(path + '97_static/source_url.csv'):
        print('No source url file found')
        return
    
    df = pd.read_csv(path + '97_static/source_url.csv')
    ids = dict()
    for _, row in df.iterrows():
        source_id_name = row['source'] 
        url_link       = row['url']

        doc_url = url_link 
        save_as =  path + '/98_data/' + source_id_name + '.pdf'

        # Retrieve and encode the PDF byte
        filepath = pathlib.Path(save_as)
        filepath.write_bytes(httpx.get(doc_url).content)
        print(f"Downloaded the PDF from {doc_url} and saved it as {filepath}")

def split_pdf(input_pdf, overlap_pages=10):
    """Splits a PDF into two equal parts and saves them with modified names."""
    doc = fitz.open(input_pdf)
    total_pages = len(doc)

    if total_pages < 2:
        print(f"Skipping {input_pdf}: Not enough pages to split.")
        return  # Skip if there's only one page

    # Calculate the midpoint
    mid = total_pages // 2

    # Generate new file names
    base_name, ext = os.path.splitext(input_pdf)  # Separate name & extension
    output_pdf1 = f"{base_name}_1{ext}"
    output_pdf2 = f"{base_name}_2{ext}"

    # Create first half
    doc1 = fitz.open()
    doc1.insert_pdf(doc, from_page=0, to_page=mid - 1)
    doc1.save(output_pdf1)
    doc1.close()

    # Create second half
    doc2 = fitz.open()
    doc2.insert_pdf(doc, from_page=mid-overlap_pages, to_page=total_pages - 1)
    doc2.save(output_pdf2)
    doc2.close()
    if os.path.exists(output_pdf1) and os.path.exists(output_pdf2):
        os.remove(input_pdf)
        print(f"PDF split successfully and original deleted:\n  {output_pdf1} ({mid} pages)\n  {output_pdf2} ({total_pages - mid} pages)")
    else:
        print(f"Error: Splitting {input_pdf} failed, original file not deleted.")

def any_file_exceeds_size(path):
    
    path_tmp    = path + '98_data/' 
    all_files   = os.listdir(path_tmp)   
    all_files   = [x for x in all_files if x.endswith('.pdf')]
    all_files   = [path_tmp + file for file in all_files]
    all_files   = [x for x in all_files if x.endswith('.pdf')]
    flag        = False
    
    for file in all_files:
        if os.path.getsize(file) > 17 * 1024 * 1024:  # Check if file size > 20MB
            print(file)
            flag = True
    
    return flag
    
def make_pdf_in_right_size(path):
    
    path_tmp    = path + '98_data/' 
    all_files   = os.listdir(path_tmp)   
    all_files   = [x for x in all_files if x.endswith('.pdf')]
    all_files   = [path_tmp + file for file in all_files]
    all_files   = [x for x in all_files if x.endswith('.pdf')]
    
    for file in all_files:
        if os.path.getsize(file) > 15 * 1024 * 1024:  # Check if file size > 20MB
            print("working on ", file, "\n")
            split_pdf(file)
        else:
            print(f"PDF is less than 20 MB: {file}\n")            
            
def create_source_to_path_map(path):
    """This function will upload all the files to chatpdf and return the source id for each file

    Args:
        path (str): here the folder path is needed where all the files are stored. We assume the same structure as the one in the repo. 
    """
    path_tmp    = path + '98_data/' 
    all_files   = os.listdir(path_tmp)   
    all_files   = [x for x in all_files if x.endswith('.pdf')]
    all_files   = [path_tmp + file for file in all_files]
    all_files = [x for x in all_files if x.endswith('.pdf')]
    
    if len(all_files) == 0:
        print('No files found')
        return
    
    ids         = dict()    
    save_path   = path + '97_static/source_path_map.csv'
    
    for file in all_files:  
        print("Working on file : ", file)
        source_id_name = file.split('/')[-1].split('.')[0]
        ids[source_id_name]    = file
    
    df_ids     = pd.DataFrame(ids.items(), columns=['source', 'path'])
    df_ids.to_csv(save_path, index=False)
    print('Source ids are saved in the file') 
    return(ids)    
        
def hit_gemini(path, load_all_fresh = False ,sr_no_list= [9999]):
    """This function will load all the promps questions and hit the chatpdf api and get the results for the prompts.
        Here we check the answer and its relevance to the query and if found irrelevant then we skip the source
    """
    parent_path    = os.path.dirname(os.path.dirname(path))
    prompts        = pd.read_csv(parent_path + '/prompts.csv')
    
    if not load_all_fresh: 
        if os.path.exists(path + '/96_results/prompts_result.csv'):
            prompts_result = pd.read_csv(path + '/96_results/prompts_result.csv')
            que_covered    = prompts_result['sr_no'].unique()
            que_list       = prompts['sr_no'].unique()
            remaining_que  = [x for x in que_list if x not in que_covered]
            if sr_no_list != [9999]:
                remaining_que = [x for x in que_list if x in sr_no_list]
            prompts        = prompts[ prompts['sr_no'].isin(remaining_que)]
            prompts_result_tmp = prompts_result[~prompts_result['sr_no'].isin(remaining_que)]
        else:
            que_list       = prompts['sr_no'].unique()
            remaining_que = [x for x in que_list if x in sr_no_list]
            prompts        = prompts[ prompts['sr_no'].isin(remaining_que)]
            prompts_result_tmp = pd.DataFrame(columns=['run_time_stamp','sr_no','cat','que_no', 'source', 'message', 'result'])
    else: 
        prompts_result_tmp = pd.DataFrame(columns=['run_time_stamp','sr_no','cat','que_no', 'source', 'message', 'result'])
    
    for _, row in prompts.iterrows():
        message   = row['message']
        disp_msg  = row['disp_message']
        que_no    = row['que_no']
        cat       = row['cat']
        sr_no     = row['sr_no']
        all_sources = row['source']
        print("checking for sr no : ", sr_no)
        print("\n\n Here the question we are checking : ", disp_msg, "\n")
        if not all_sources=="ALL":
            all_sources = [values for key, values in ids.items() if key.startswith(row['source'])]
            #all_sources          = [ids[row['source']]]
            #all_sources          = filtered_source_path
        else:
            all_sources = list(ids.values())
        
        print("All sources we are using : ", all_sources, "\n")
        
        for source in all_sources:
            source_path = source    
            print("Source we are using :" + source_path)
            
            result = ask_gemini_doc_v2(source_path, message)
            
            print("\n Here is what Gemini got : ", result)
            
            ############ Attempting guard rail integration  #############
            print("\nAttempting guardrail checks\n")
             
            got_answer, source_mentioned = guardrail_agent(message, result)
        
            if got_answer == 'no':
                
                query_prompt = (
                        f"Original query was : {message}. "
                        f"Howver it could not fine the answer. You need to rephrase the question. "
                )
                modi_query = ollama.chat(
                                model=model_to_use,
                                messages=[
                                {
                                "role": "user",
                                "content": query_prompt,
                                },
                                ],
                            )

                modi_query = modi_query["message"]["content"].strip()
                
                print("We are using following modified query " , modi_query, "\n")
                
                result_attempt2 = ask_gemini_doc_v2(source_path, modi_query) 
                result    = result + result_attempt2
            
            if source_mentioned == 'no':
                
                query_prompt = (
                        f"Original query was : {message}. "
                        f"The answer was received. "
                        f"However, it could not page number from {source_path}. Make a query to ask on which page number following information is found {result} "
                )
                modi_query = ollama.chat(
                                model=model_to_use,
                                messages=[
                                {
                                "role": "user",
                                "content": query_prompt,
                                },
                                ],
                            )

                modi_query = modi_query["message"]["content"].strip()
                
                print("We are using following modified query " , modi_query, "\n")
                
                result_attempt2 = ask_gemini_doc_v2(source_path, modi_query) 
                result    = result + result_attempt2
            
            print("\nSaving following answer : ", result, "\n")
            
            # Append the new row to the results DataFrame
            run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'sr_no':[sr_no],'cat':[cat],'que_no':[que_no],'source': [row['source']], 'message': [message], 'result': [result]})
            prompts_result_tmp = pd.concat([prompts_result_tmp, new_row], ignore_index=True)
            prompts_result_tmp.to_csv(path + '/96_results/prompts_result.csv',  index=False)    

def ask_gemini_doc_v2(doc_path, prompt):
    """This function will ask the gemini model with the document and prompt and get the result.
        If the result is not found then it will retry after 60 seconds.
    """
    print("Okay, asking Gemini...")
    global ids_chatpdf
    policy_filepath = pathlib.Path(doc_path)    
    try:
        response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=[
                        types.Part.from_bytes(
                        data=policy_filepath.read_bytes(),
                        mime_type='application/pdf',
                        ),
                    prompt])
        return response.text
    except Exception as e:
        if not "Request payload size exceeds the limit" in str(e):
            print(f"Error: {e}. Retrying in 60 seconds...")
            time.sleep(60)
            try:
                response = client.models.generate_content(
                                model="gemini-1.5-flash",
                                contents=[
                                types.Part.from_bytes(
                                data=policy_filepath.read_bytes(),
                                mime_type='application/pdf',
                                ),
                            prompt])
                return response.text
            except Exception as e:
                print(f"Error: {e}. Asking Gemini failed. Trying chatPDF...")
                path_temp,temp_id =upload_this_doc_to_chatpdf(path, doc_path)
                ids_chatpdf[path_temp] = temp_id
                cp_try  = hit_chatpdf_api(doc_path, prompt)
                return cp_try
        else:
            print(f"Too big a document for Gemini. Trying chatPDF...")
            path_temp,temp_id =upload_this_doc_to_chatpdf(path, doc_path)
            ids_chatpdf[path_temp] = temp_id
            cp_try  = hit_chatpdf_api(doc_path, prompt)
            return cp_try

def guardrail_agent(query,ans):
    
    promptcr = (
            f"You are a smart LLM output assessor. "
            f"We asked following query to one LLM: {query}. "
            f"The LLM came up with the following answer: {ans}. "
            f"Did LLM got the answer or it failed? "
            f"Does the answer do justice to the question ?  "
            f"Does the answer mention the page number of section number from which it has got the answer? "
            f"To make answer reliable, the source should say from annual report page number xx, we found that..."
            f"Answer should be in Json format. For example: {'{"got_answer": "yes/no", "source_mentioned": "yes/no"}'}"
            f"Answer other than JSON format will not be accepted. ")
            
    try:
        
        result = ollama.chat(
            model=model_to_use,
            messages=[
                {
                    "role": "user",
                    "content": promptcr,
                },
            ],
        )
        print("Guardrail response : ",result["message"]["content"])
        response_text = result["message"]["content"].strip()
    
        match    = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            print("In guardrail verdict, JSON object found\n")
            response = json.loads(match.group())
    
            ans = response.get('got_answer', 'N/A')
            modi_query = response.get('source_mentioned', 'N/A')
        else:
            ans = 'na'
            modi_query= 'na'
            
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    except json.JSONDecodeError:
        print("Failed to parse response as JSON. Raw output:")
        ans = 'na'
        modi_query= 'na'
        
    return(ans,modi_query)
            
def llama_answer_checker(query, 
                   ans, 
                   critera = "Does it have hints of halluciantion or off topic points?"):
    
    promptcr = (
            f"You are a smart LLM output assessor."
            f"We asked following query to one LLM: {query}."
            f"The LLM came up with the following answer: {ans}."
            f"Based on the criteria mentioned below, you have to assess whether the answer is correct or not."
            f"Here is a guidance on how to assess the answer: {critera}."
            f"Please provide your answer in 'yes' or 'no' format."
            f"No preamble or extra information is required."
            f"Answer should be in Json format. For example: {'{"answer": "yes"}'}")

    try:
        
        result = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": promptcr,
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
        ans = response.get('answer', 'N/A')
       

        print(f"Verdict: {ans}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    except json.JSONDecodeError:
        print("Failed to parse response as JSON. Raw output:")
        ans = 'no'
    
    return(ans)

def create_chatpdf_source_id_upload(path):
    """This function will upload ALL the files to chatpdf and return the source id for each file

    Args:
        path (str): here the folder path is needed where all the files are stored. We assume the same structure as the one in the repo. 
    """
    path_data   = path + '98_data/' 
    all_files   = os.listdir(path_data)   
    all_files   = [x for x in all_files if x.endswith('.pdf')]
    all_files   = [path_data + file for file in all_files]
    all_files = [x for x in all_files if x.endswith('.pdf')]
    
    if len(all_files) == 0:
        print('No files found')
        return
    
    ids         = dict()    

    for file in all_files:  
        print("Working on file : ", file)
        
        files = [
            ('file', ('file', open(file, 'rb'), 'application/octet-stream'))
        ]
        headers = {
            'x-api-key': CHATPDF_API_KEY
        }

        response = requests.post(
            'https://api.chatpdf.com/v1/sources/add-file', headers=headers, files=files)

        if response.status_code == 200:
            #print('Source ID:', response.json()['sourceId'])
            source_id_name = file                            #file.split('/')[-1].split('.')[0]
            source_id      = response.json()['sourceId']
            ids[source_id_name] = source_id
            df_ids     = pd.DataFrame(ids.items(), columns=['source', 'source_id'])
            df_ids.to_csv(path + '/97_static/chatpdf_source_id_map.csv', index=False)
        else:
            print('Status:', response.status_code)
            print('Error:', response.text)
    
    return(ids)

def hit_chatpdf_api(doc_path, prompt):
    """This function will load all the promps questions and hit the chatpdf api and get the results for the prompts.
        here we first summarize the source and see if the query is relevant to the source or not. If it is relevant then only we hit the chatpdf api
    """
    source_id = ids_chatpdf[doc_path]
    
    headers = {
        'x-api-key': CHATPDF_API_KEY,
        "Content-Type": "application/json",
    }
            
    data = {
        'sourceId': source_id,
        'messages': [
            {
                'role': "user",
                'content': prompt,
            }
        ]
    }
    response = requests.post(
        'https://api.chatpdf.com/v1/chats/message', headers=headers, json=data)

############################################################################################
    if response.status_code == 200:
        #print('Result:', response.json()['content'])
        result = response.json().get('content', '')  # Extract result 
    else:
        print('Status:', response.status_code)
        print('Error:', response.text)
        result = 'could not find the result'
    return result

def upload_this_doc_to_chatpdf(path, file):
    print("Uploading file on chatPDF : ", file)
    chatpdf_file = path + '/97_static/chatpdf_source_id_map.csv'
    
    if os.path.exists(chatpdf_file):
        chatpdf_ids = pd.read_csv(chatpdf_file)
        if file in chatpdf_ids['source'].values:
            print("This source is uploaded already")
            return file,chatpdf_ids[chatpdf_ids['source']==file]['source_id'].values[0]  
    else:
        chatpdf_ids = pd.DataFrame(columns=['source', 'source_id'])
        
    ids   = dict()    
    files = [
        ('file', ('file', open(file, 'rb'), 'application/octet-stream'))
    ]
    headers = {
        'x-api-key': CHATPDF_API_KEY
    }

    response = requests.post(
        'https://api.chatpdf.com/v1/sources/add-file', headers=headers, files=files)

    if response.status_code == 200:
        #print('Source ID:', response.json()['sourceId'])
        source_id_path       = file                            #file.split('/')[-1].split('.')[0]
        source_id            = response.json()['sourceId']
        ids[source_id_path]  = source_id
        df_ids               = pd.DataFrame(ids.items(), columns=['source', 'source_id'])
        df_ids               = pd.concat([df_ids,chatpdf_ids],axis=0, ignore_index=True)
        df_ids.to_csv(chatpdf_file, index=False)
    else:
        print('Status:', response.status_code)
        print('Error:', response.text)

    return source_id_path, source_id

def aggregate_prompt_scoring(path):
    parent_path    = os.path.dirname(os.path.dirname(path))
    folder_names = [f.name for f in os.scandir(parent_path) if f.is_dir()]
    all_prompt_result    = pd.DataFrame()
    que_result           = pd.DataFrame() 
    
    for folder in folder_names:    
        prompt_result_path  = parent_path + f'/{folder}' + '/96_results/prompts_result.csv'
        all_prompt_result   = pd.concat([all_prompt_result,pd.read_csv(prompt_result_path)])
        
        que_result_path     = parent_path + f'/{folder}' + '/96_results/que_wise_scores.csv'
        que_result          = pd.concat([que_result,pd.read_csv(que_result_path)])
        
    all_prompt_result.to_csv(parent_path + '/all_promp_results.csv')
    que_result.to_csv(parent_path + '/que_results.csv')       
    
    
############################################################################################
#############           Scoring Agents                                #####################
############################################################################################

    
def score_q4(path):
    """This function scores question 4 based on predefined criteria.
    
    Args:
        path (str): Path to the folder for the company.
        
    Question is : Does the board have directors with permanent boaerd seats?
    """
    
    # Load the dataset
    question_no = 4
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered      = pr[pr['que_no'] == question_no]['result']
    content          = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]
    
    if pr_filtered.empty:
        print(f"No content available for question {question_no}.")
        return

    print(f"Content for question {question_no}: {content}")
    
    if not content.strip():
        print(f"No content available for question {question_no}.")
        return

    # Define the scoring criteria
    scoring_criteria = (
        "Score 0 if any one of the directors is marked as permanent board members as well as they are not explicitly mentioned to be representatives of lenders."
        "Score 1 if the directors which are marked as permanent board members, but those are representatives of lenders. Remember that usually this case is applicable for financially distressed companies. So unless it is mentioned explicitly that lenders have sent those board members as representative, do not assume so."
        "Score 2 if All directors are marked as non-permanent board members"

    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Do not assume any details. If you dont find any information then it is not available. "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number and report quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q7(path):
    """This function scores question 7 based on predefined criteria.
    
    Args:
        path (str): Path to the folder for the company.
        
    Question is : Does the board have directors with permanent boaerd seats?
    """
    
    # Load the dataset
    question_no = 7
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered      = pr[pr['que_no'] == question_no]['result']
    content          = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]
    
    if pr_filtered.empty:
        print(f"No content available for question {question_no}.")
        return
    
    print(f"Content for question {question_no}: {content}")
    
    if not content.strip():
        print(f"No content available for question {question_no}.")
        return

    # Define the scoring criteria
    scoring_criteria = (
        "Score 0 if the content finds the gap between FYE (financial year ending) and AGM more than 6 months. Or any such details is not available."
        "Score 1 if the content finds the gap between FYE (financial year ending) and AGM is between 4 to 6 months."
        "Score 2 if the content finds that the gap between FYE (financial year ending) and AGM is less than 4 months."
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Keep the justification as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Do not assume any details. If you dont find any information then consider it as not available. "
        f"The content on which you have to score is: {content}. "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number and report quoted in the content. Make sure you don't miss that. "
        f"Write elaborate justification for the score you provide."
    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q10(path):
    """This function scores question 3 based on predefined criteria.
    
    Args:
        path (str): Path to the folder for the company.
        
    Question is : Does the board have directors with permanent boaerd seats?
    """
    
    # Load the dataset
    question_no = 10
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered      = pr[pr['que_no'] == question_no]['result']
    content          = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]
    
    if pr_filtered.empty:
        print(f"No content available for question {question_no}.")
        return

    
    print(f"Content for question {question_no}: {content}")
    
    if not content.strip():
        print(f"No content available for question {question_no}.")
        return

    # Define the scoring criteria
    scoring_criteria = (
        "The content here is about information regarding the conflict of interest with stake holders. Two main things to check are the policy and adherence to the policy. "
        "Score 0 if you find there is not even policy which talks about the conflict of interest with stake holders or you find incidences or information which suggests that the company has incidents of stakeholder conflict of interest."
        "Score 1 if there is policy but it does not mentioned explicitly that it covers suppliers and vendors."
        "Score 2 if no conflict of interest with stake holders is found as well as policy covers all stake holders including the suppliers and vendors."
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Do not assume any details. If you dont find any information then consider it as not available. "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number and report quoted in the content. Make sure you don't miss that. "
        f"The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q12(path):
    """This function scores question 12 based on predefined criteria.
    
    Args:
        path (str): Path to the folder for the company.
        
    Question is : Does the board have directors with permanent boaerd seats?
    """
    
    # Load the dataset
    question_no = 12
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered      = pr[pr['que_no'] == question_no]['result']
    content          = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]
    
    if pr_filtered.empty:
        print(f"No content available for question {question_no}.")
        return

    
    print(f"Content for question {question_no}: {content}")
    
    if not content.strip():
        print(f"No content available for question {question_no}.")
        return

    ##################### Applying question specific post processing #####################
    
    pp_prompt = (
        f"This context has a json file which lists the information about the related party transactions."
        f"Please check the json file and filter all the transactions which corresponds to the Royalty related transactions."
        f"Aggregate the amount of all the transactions and provide the total amount of Royalty transactions."
        f"The profit of the company is also mentioned in the content. Please companre total amount of Royalty transactions with the profit of the company."
        f"If royalty is not mentioned, assume it 0"
        f"The context text is : {content}"
    )
    
    response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[pp_prompt])
    
    content = response.text    
    print(f"Content after post processing: {content}")
    
    
    ##################### Applying question specific post processing #####################
    
    # Define the scoring criteria
    scoring_criteria = (
        "Here we are comparing the royalty transactions with the related parties to the profit of the company. "
        "Score 0 if ratio of the royalty payments to the related parties to the profit of the company is more than 0.2."
        "Score 1 if ratio of the royalty payments to the related parties to the profit of the company is more than 0.1 but less than 0.2."
        "Score 2 if if ratio of the royalty payments to the related parties to the profit of the company is less than 0.1."
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Do not assume any details. If you dont find any information then consider it as not available. "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q16(path):
    """This function scores question 16 based on predefined criteria.
    
    Args:
        path (str): Path to the folder for the company.
        
    Question is : Does the board have directors with permanent boaerd seats?
    """
    
    # Load the dataset
    question_no = 16
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered      = pr[pr['que_no'] == question_no]['result']
    content          = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]
    
    if pr_filtered.empty:
        print(f"No content available for question {question_no}.")
        return

    
    print(f"Content for question {question_no}: {content}")
    
    if not content.strip():
        print(f"No content available for question {question_no}.")
        return

    # Define the scoring criteria
    scoring_criteria = (
        "The content here is about structures of mechanisms that could violate the minority stake holders rights. "
        "Score 0 if you get the impression that yes the structure or mechanisms are such that it could be unfair treatments of minority stakeholders and their rights could be violated."
        "Score 2 if you get the impression that there is no evidence that structure or mechanisms could violate minority stakeholders' rights."
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Do not assume any details. If you dont find any information then consider it as not available. "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q18(path):
    """This function scores question 18 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 18
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print(f"No content available for question {question_no}.")
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are assessing the company on health, safety, and welfare related aspects."
        "Score 0 if you get the impression that the policies are not publicly disclosed and the company has not provided information on the number of employee accidents or there have been labour fatalities on account of accidents in the work place. "
        "Score 1 if you get the impression that The policies are publicly disclosed or the company has provided information on the number of employee accidents  "
        "Score 2 if you get the impression that The company has provided information on the number of employee accidents and has publicly disclosed its health and safety policies"
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q19(path):
    """This function scores question 19 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 19
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question ", question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "Score 0 if you get the impression that the policy regarding sexual harrassment has not been publicly disclosed and the company has not provided information on the number of sexual harassment incidents. "
        "Score 1 if you get the impression that the policy regarding sexual harrassment is publicly disclosed or the company has provided information on the sexual harassment incidents. "
        "Score 2 if you get the impression that the company has provided information on the number of sexual harassment incidents and has publicly disclosed its prevention of sexual harassment policy. "
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria}. "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no} : {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q20(path):
    """This function scores question 20 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 20
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "Check if complay has policies for either supplier/vendor management or selection. Score 0 if you get the impression that policies are not available for both supplier/contractor management and selection"
        "Score 1 if you get the impression that policies are available either for supplier/contractor management or selection and not both."
        "Score 2 if you get the impression that policies are available for both supplier/contractor management and selection"
        "While checking, it is not necessary that the policies should have any metrics or KPIs. The mere presence of policies is sufficient."
    )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q21(path):
    """This function scores question 21 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 21
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "Check if in whole content there is any indication whether company has delayed the payment to any lender, vendor or supplier. Please do not include disputed cases in this."
        "Score 0 if you find that company has delayed the payment to any lenders. Or delay in payment has resulted into rating decrease for the company"
        "Score 1 if you find that company has delayed the payment to suppliers, vendors, but not to any lenders."
        "Score 2 if you find that company has not delayed the payment to any lenders, suppliers or vendors."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q22(path):
    """This function scores question 22 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 22
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "Based on the information provided in the context, we are trying to assess the ethical standards and anti-corruption policies of the company."
        "We only need overall score and justification. Ignore other information provided to you "
        "Score 0 if you find in the context policy regarding the ethical standards or anti corruption is not available."
        "Score 1 if you find in the context policy regarding the anti curryption is available but it does not mention any anti-corruption or anti-bribary measures."
        "Score 2 if you find in the context policy regarding the ethics and anti-corruption and anti-bribary are available with appropriate meausures."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with only two keys. The keys are 'score' (integer) and 'justification' (string). Let the value of the justification should be as elaborate as possible. "
        f"Other than JSON format with keys 'score' and 'justification', any other format will not be accepted."
        f"If you don't find information you are looking for, or if you cannot just put in required JSON format ->  make score = 0 and justification = 'I could not make judgement'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q23(path):
    """This function scores question 23 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 23
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat          = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess the social responsibility of the company and are taking CSR spending as proxy."
        "Score 0 if net profit is positive number and yet have not spent on CSR activities"
        "Score 1 if ratio of amount spent in CSR activities to the net profit is between 0 and 0.02 (exclusive)."
        "Score 2 if ratio of amount spent in CSR activities to the net profit is more than 0.02 or negative."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q28(path):
    """This function scores question 28 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 28
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess if auditors have raised any concern or not."
        "Score 0 if Auditor has issued a qualified opinion or the financial statements have been restated or the auditor has resigned due to differences in accounting opinion "
        "Score 1 if Auditor has raised an emphasis of matter."
        "Score 2 if Auditor has issued an unqualified opinion without any matter of emphasis."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. Let the justification be as elaborate as possible. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string)."
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q32(path):
    """This function scores question 28 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 32
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess if company has related party transaction policy and if yes, how comprehensive it is."
        "Score 0 if there is no policy available for related party transactions."
        "Score 1 if the policy for related party transactions is available but either of definition of ordinary business, definition of materiality of transaction or reuiqement of such transaction reviewed by external auditor is missing."
        "Score 2 if the policy for related party transaction is available all of following is mentioned: definition of ordinary business, definition of materiality of transaction and reuiqement of such transaction reviewed by external auditor."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q36(path):
    """This function scores question 28 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 36
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to see if company transperantly discloses the information about the shareholding pattern."
        "Score 0 if shareholding pattern is not available and the annual report does not have the top 10 shareholder names."
        "Score 1 if either of shareholding pattern is available or the annual report has the top 10 shareholder names."
        "Score 2 if both the shareholding pattern and the top 10 shareholder names are mentioned in annual report."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q37(path):
    """This function scores question 37 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 37
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to see if company transperantly discloses the information about the shareholding pattern for board members and key management personals."
        "Score 0 if shareholding information is not available for both board members and key management personals"
        "Score 1 if shareholding information is available for either of board members or key management personals."
        "Score 2 if shareholding information is available for both board members and key management personals."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q38(path):
    """This function scores question 38 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 38
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to see if company has dividend distribution policy and it talks about the dividend distribution ratio and follows it or not."
        "Score 0 if dividend distribution policy is not available or if it is available clear guidance on payout ratio is not given."
        "Score 1 if there is dividend distribution policy and the guidance for the payout ratio is available, but in last 3 years the ratio is not rigourously followed. There have been deviations."
        "Score 2 if the policy is publicly available, specifies a target payout ratio; and there have not been any deviations from the policy in the past three years or the rationale for deviation has been clearly provided"
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q44(path):
    """This function scores question 38 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 44
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to see if board members and key management personals are well qualified or not."
        "Score 0 if qualifications, field of experties and number of years of experience in the field is not available for both board members and key management personals."
        "Score 1 if qualifications, field of experties and number of years of experience in the field is available for either board members or key management personals."
        "Score 2 if qualifications, field of experties and number of years of experience in the field is available for both board members and key management personals."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible."
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q45(path):
    """This function scores question 38 based on predefined criteria."""
    
    # Load the dataset
    question_no  = 45
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    
    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to see if board members and key management personals were fined by regulators or stock exchanges."
        "Score 0 if there is any mention of either board members or key management personals being fined by regulators or stock exchanges."
        "Score 2 if there is no mention of either board members or key management personals being fined by regulators or stock exchanges."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a corporate governance scoring expert. "
        f"You have to score the content based on the criteria mentioned below. "
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible." 
        f"Other than JSON format, any other format will not be accepted."
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q48(path):
    """This function scores question 38 based on predefined criteria."""
    # Load the dataset
    question_no  = 48
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to see if board members attend meeting regularly or not."
        "We will evaluate the attendance ratio (number of meetings they attended compared to number of total meetings that occured) of all board members."
        "Score 0 if overall attendance ratio for all board members on an average is less than 0.75"
        "Score 1 if overall attendance for all board members on an average is between 0.75 and 0.90."
        "Score 2 if overall attendance for all board members on an average is more than 0.90."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a scoring agent." 
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible."
        f"Score is either 0,1 or 2, nothing else. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
    #    return
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q49(path):
    """This function scores question 49 based on predefined criteria."""
    # Load the dataset
    question_no  = 49
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to see if board members met enough number of times or not."
        "Score 0 if board met less than 4 times in the financial year."
        "Score 1 if board met exactly 4 times in the financial year."
        "Score 2 if board met more than 4 times in the financial year."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a scoring agent." 
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Score is either 0,1 or 2, nothing else. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
    #    return
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q51(path):
    """This function scores question 49 based on predefined criteria."""
    # Load the dataset
    question_no  = 51
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess the experties of board collectively in area among [Legal, financial, Marketing, General Management, Supply chain, Operations, Industry relevant experience, other]."
        "Score 1 if collectively board has strong experties (more than 10 years in each) in less than 3 areas and at least one member has prior working experience in the major industry in which company operates."
        "Score 2 if collectively board has strong experties (more than 10 years in each) in at least 3 areas and at least one member has prior working experience in the major industry in which company operates."
        "Score 0 otherwise."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a scoring agent." 
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Score is either 0,1 or 2, nothing else. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
    #    return
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q52(path):
    """This function scores question 52 based on predefined criteria."""
    # Load the dataset
    question_no  = 52
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess gender ratio of independent board members."
        "Score 0 if there are no independent board members who is female i.e the gender ratio for independent director is 0."
        "Score 1 if independent board members gender ratio is less than 0.3 and more than 0"
        "Score 2 if independent board members gender ratio is more than 0.3."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a scoring agent." 
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Score is either 0,1 or 2, nothing else. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
    #    return
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q53(path):
    """This function scores question 53 based on predefined criteria."""
    # Load the dataset
    question_no  = 53
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess gender ratio of whole company's workforce."
        "Score 0 if there is not information available or female to total workforce ratio is less than 10%."
        "Score 1 if female to total workforce ratio is less between 10% (inclusive) and 30%"
        "Score 2 if female to total workforce ratio is more than or equal to 30%"
        )
    
    # Construct the prompt
    prompt = (
        f"You are a scoring agent." 
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Score is either 0,1 or 2, nothing else. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
    #    return
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q54(path):
    """This function scores question 54 based on predefined criteria."""
    # Load the dataset
    question_no  = 54
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess independence of the board."
        "Score 0 board chainman is executive/promoter and independence of the board is less than 50%."
        "Score 0 board chainman is not executive/promoter and independence of the board is less than 33%."
        "Score 1 if for board whose chairman is executive/promoter, independence of the board is 0.5 or more or if chairman is not executive/promoter, independence of the board is 0.33 or more. If you see there are vintage members on board, but they do not affirm to independence annually. If there are no confirmed vintage members, then do not bother."
        "Score 2 if for board whose chairman is executive/promoter, independence of the board is 0.5 or more or if chairman is not executive/promoter, independence of the board is 0.33 or more. "
        )
    
    # Construct the prompt
    prompt = (
        f"You are a scoring agent." 
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Score is either 0,1 or 2, nothing else. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
    #    return
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q55(path):
    """This function scores question 54 based on predefined criteria."""
    # Load the dataset
    question_no  = 55
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess if committess formation is fine."
        "Criteria 1 is Audit committee should have more than or equal to 3 members."
        "Criterial 2 is Audit committee and NRC committee members should be independent"
        "Criteria 3 is members on either Audit committee or NRC committee should not be on board for more than 10 years."
        "Score 0 if 2 or more criteria is violated."
        "Score 1 if only one criteria is violated."
        "Score 2 if no criteria is violated."
        )
    
    # Construct the prompt
    prompt = (
        f"You are a scoring agent." 
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Score is either 0,1 or 2, nothing else. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
    #    return
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def score_q63(path):
    """This function scores question 63 based on predefined criteria."""
    # Load the dataset
    question_no  = 63
    file_path    = os.path.join(path, '96_results', 'prompts_result.csv')
    pr           = pd.read_csv(file_path)
    
    # Extract the relevant content
    pr_filtered  = pr[pr['que_no'] == question_no]['result']
    content      = '\n'.join(pr_filtered.dropna().tolist())  # Handle NaN values safely
    que_cat      = pr[pr['que_no'] == question_no]['cat'].values[0]

    if not content.strip():
        print("No content available for question " ,   question_no)
        return

    print(f"Content for question {question_no}: {content}")
    
    # Define the scoring criteria
    scoring_criteria = (
        "We are trying to assess if CEO compensation is fine."
        "Score 0 if Total compensation of CEO is more than 0.05 of net profits."
        "Otherwise score 2"
        )
    
    # Construct the prompt
    prompt = (
        f"You are a scoring agent." 
        f"Answer in JSON format with keys as 'score' (integer) and 'justification' (string). Let the justification be as elaborate as possible. "
        f"Score is either 0,1 or 2, nothing else. "
        f"Other than JSON format, any other format will not be accepted. "
        f"If you don't find information you are looking for make score = 0 and justification = 'No information available'. "
        f"The scoring criteria is as follows: {scoring_criteria} "
        f"Whatever references or source information you find in content, make sure you retain that along with document name and page number. "
        f"Make sure you quote page number along with document quoted in the content. Make sure you don't miss that. "
        f"And The content on which you have to score is: {content}. "

    )
    if os.path.exists(path + '/96_results/que_wise_scores.csv'):
        output_df = pd.read_csv(path + '/96_results/que_wise_scores.csv')
    else:
        output_df = pd.DataFrame(columns=['run_time_stamp','company','category','que_no', 'score', 'justification'])
    
    # Call the local Ollama model with the prompt
    try:
        
        result = ollama.chat(
            model=model_to_use,
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
            #raise ValueError("Failed to extract JSON response from the output.")
            print("Failed to extract JSON response from the output. So falling back.")
            response = {"score": 0, "justification": f"I could not make judgement, however content is : {content}"}

        
        # Extract score and justification
        score = response.get('score', 'N/A')
        justification = response.get('justification', 'No justification provided.')

        print(f"Score for question {question_no}: {score}")
        print(f"Justification: {justification}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
    #except json.JSONDecodeError:
    #    print("Failed to parse response as JSON. Raw output:")
    #    print(result.stdout)
    #    return
        
    # Append the new row to the results DataFrame
    run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = pd.DataFrame({'run_time_stamp':[run_time_stamp],'company':[company_sym],'category':[que_cat],'que_no':[question_no],'score': [score], 'justification': [justification]})
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(path + '/96_results/que_wise_scores.csv',  index=False)

def answer_category1():

    print("\n\n#############                   Board permanency                      #####################\n\n")
    score_q4(path)

    print("\n\n#############                   AGM delays if any                     #####################\n\n")
    score_q7(path)

    print('\n\n#############                   conflict of interest                  #####################\n\n')
    score_q10(path)

    print('\n\n#############                   Royalty in related party tx           #####################\n\n')
    score_q12(path) 

    print('\n\n#############           Violation of minority shareholders rights     #####################\n\n')
    score_q16(path)

def answer_category2():
    print("\n\n#############                   health, safety, and welfare            #####################\n\n")
    score_q18(path)

    print("\n\n#############                   Sexual harrassment policy              #####################\n\n")
    score_q19(path)

    print("\n\n#############                   Supplier/vendor selection/Management    #####################\n\n")
    score_q20(path)

    print("\n\n#############                   Delay in payment to stackholders        #####################\n\n")
    score_q21(path)

    print("\n\n#############                   Anti curruption/Anti bribary            #####################\n\n")
    score_q22(path)

    print("\n\n#############                   CSR Spent                               #####################\n\n")
    score_q23(path)

def answer_category3():
    print("\n\n#############                   Auditor's opinion                      #####################\n\n")
    score_q28(path)

    print("\n\n#############                   RPT policy                             #####################\n\n")
    score_q32(path)

    print("\n\n#############                   Shareholding pattern                    #####################\n\n")
    score_q36(path)

    print("\n\n#############                   Shareholding pattern board/KMP          #####################\n\n")
    score_q37(path)

    print("\n\n#############                   Dividend distribution policy          #####################\n\n")
    score_q38(path)

    print("\n\n#############                   board qualification                   #####################\n\n")
    score_q44(path)

    print("\n\n#############                   check for any fines                   #####################\n\n")
    score_q45(path)

def answer_category4():
    print("\n\n#############            Attendence percentage in the meeting           #####################\n\n")
    score_q48(path)

    print("\n\n#############                   Nr times board met                     #####################\n\n")
    score_q49(path)

    print("\n\n#############                   Board experties                       #####################\n\n")
    score_q51(path)

    print("\n\n#############                   Gender diversity on board             #####################\n\n")
    score_q52(path)

    print("\n\n#############                   Gender diversity on workforce         #####################\n\n")
    score_q53(path)

    print("\n\n#############                   board independence                     #####################\n\n")
    score_q54(path)

    print("\n\n#############                   Committee checks                     #####################\n\n")
    score_q55(path)

    print("\n\n#############                   CEO compensation                      #####################\n\n")
    score_q63(path)

############################################################################################
#############                         Script                           #####################
############################################################################################

company_sym = 'TATAMOTORS'
path       = f'/Users/monilshah/Documents/02_NWU/01_capstone/04_Code_v3/{company_sym}/'


create_gemini_source_from_url(path)

# Split big PDF
while any_file_exceeds_size(path):
    make_pdf_in_right_size(path)

ids_chatpdf = dict()
ids         =  create_source_to_path_map(path)
#ids_chatpdf = create_chatpdf_source_id_upload(path)



hit_gemini(path, load_all_fresh = False, sr_no_list=[44])   

###########################################################################################
#############   Category 1 : Rights and equitable treatment of shareholders  ##############
###########################################################################################

answer_category1()

###########################################################################################
#################         Category 2: Role of stakeholders             ####################
###########################################################################################

answer_category2()

###########################################################################################
#################         Category 3: Transperancy and disclosure       ####################
###########################################################################################

answer_category3()

###########################################################################################
#################         Category 4: Responsibility of the board      ####################
###########################################################################################

answer_category4()


