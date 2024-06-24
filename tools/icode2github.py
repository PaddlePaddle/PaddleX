# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
import re  

NEW_COPYRIGHT = '# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.\n\
# \n\
# Licensed under the Apache License, Version 2.0 (the "License");\n\
# you may not use this file except in compliance with the License.\n\
# You may obtain a copy of the License at\n\
#\n\
#    http://www.apache.org/licenses/LICENSE-2.0\n\
#\n\
# Unless required by applicable law or agreed to in writing, software\n\
# distributed under the License is distributed on an "AS IS" BASIS,\n\
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
# See the License for the specific language governing permissions and\n\
# limitations under the License.\n' 
  
def replace_copyright_in_file(file_path, new_copyright):  
    """
    Replace copyright information in single Python file.
    
    Args:
        file_path (str): The path of the file to be processed. 
        new_copyright (str): The new copyright information. 
    
    Returns:
        None
    
    """
    print(f"Processing file: {file_path}")    
    try:  
        with open(file_path, 'r+', encoding='utf-8') as file:  
            content = file.read()  
            pattern = re.compile(r'(# !/usr/bin/env python3[\s\S]*?Authors\s*\n""")', re.MULTILINE)  
            new_content = pattern.sub(new_copyright + '\n', content)  
            if new_content != content:  
                print(f"Copyright information replaced in {file_path}") 
                file.seek(0)  # Reset the file pointer to the beginning of the file.
                file.write(new_content)  
                file.truncate()    
    except Exception as e:  
        print(f"Error processing file {file_path}: {e}")  
   
  
def replace_copyright_in_directory(directory, new_copyright):  
    """
    Replace copyright information in Python files under the specified directory.
    
    Args:
        directory (str): The directory path where Python files are located.
        new_copyright (str): The new copyright information to be replaced.
    
    Returns:
        None.
    
    """
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith('.py'):  
                file_path = os.path.join(root, file)  
                replace_copyright_in_file(file_path, new_copyright)  
  
if __name__ == '__main__':  
    replace_copyright_in_directory('./', NEW_COPYRIGHT)
 
