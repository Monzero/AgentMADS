import requests
import os
import pandas as pd

path       = '/Users/monilshah/Documents/02_NWU/01_capstone/02_Code/98_data/'
prompts    = pd.read_csv('/Users/monilshah/Documents/02_NWU/01_capstone/02_Code/97_static/prompts.csv')
ids        = create_source_id_upload(path)

def create_source_id_upload(path):
    all_files   = os.listdir(path)   
    all_files = [x for x in all_files if x.endswith('.pdf')]
    all_files = [path + file for file in all_files]

    ids         = dict()
    all_files = [x for x in all_files if x.endswith('.pdf')]

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
 
 
   
    # ######
    # ids         = dict()
    # all_files   = os.listfiles(path)
    
    # all_files = [x for x in all_files if x.endswith('.pdf')]

    # for file in all_files:  
    #     files = [
    #         ('file', ('file', open(file, 'rb'), 'application/octet-stream'))
    #     ]
    #     headers = {
    #         'x-api-key': 'sec_c3FrcK3DSUfwnXo9M3OtqtS467NO3KOe'
    #     }

    #     response = requests.post(
    #         'https://api.chatpdf.com/v1/sources/add-file', headers=headers, files=files)

    #     if response.status_code == 200:
    #         #print('Source ID:', response.json()['sourceId'])
    #         source_id_name = file.split('.')[0]
    #         source_id      = response.json()['sourceId']
    #         ids[source_id_name] = source_id
    #     else:
    #         print('Status:', response.status_code)
    #         print('Error:', response.text)

    # return ids
    
    ##### CSV format
    
    #promt1 | source | message 
    
def hit_chatpdf_api():
    headers = {
        'x-api-key': 'sec_c3FrcK3DSUfwnXo9M3OtqtS467NO3KOe',
        "Content-Type": "application/json",
    }

    results = pd.DataFrame(columns=['source', 'message', 'result'])
    
    for _, row in prompts.iterrows():
        source     = ids[row['source']]
        message   = row['message']
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
            #prompts['result'] = response.json()['content']
            result = response.json().get('content', '')  # Extract result safely
        else:
            print('Status:', response.status_code)
            print('Error:', response.text)
            
            # Append the new row to the results DataFrame
        new_row = pd.DataFrame({'source': [source], 'message': [message], 'result': [result]})
        results = pd.concat([results, new_row], ignore_index=True) 
    
    results.to_csv('/Users/monilshah/Documents/02_NWU/01_capstone/02_Code/97_static/prompts_result.csv')  

hit_chatpdf_api()