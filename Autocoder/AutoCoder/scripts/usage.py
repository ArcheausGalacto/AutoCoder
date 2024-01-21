import openai
import os
import subprocess

def query_gpt4_chat(prompt, model="gpt-4-1106-preview"):
    with open("F:\\AutoCoder\\AutoCoder\\api_key.txt", "r") as file:
        api_key = file.read().strip()

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message['content'].strip()

def get_last_script(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.py')]
    if not files:
        return None
    return max(files, key=os.path.getctime)

def get_next_version_number(directory, base_filename):
    version = 1
    while os.path.exists(os.path.join(directory, f"{base_filename}_{version}.txt")):
        version += 1
    return version

def run_script(script_command):
    try:
        result = subprocess.run(script_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr

def save_debug_output(script_name, output, directory="F:\\AutoCoder\\AutoCoder\\debug_output"):
    base_filename = f"debug_{os.path.splitext(script_name)[0]}"  # Remove .py extension
    version = get_next_version_number(directory, base_filename)
    filename = os.path.join(directory, f"{base_filename}_{version}.txt")
    
    with open(filename, "w") as file:
        file.write(output)
    print(f"Debug output saved as {filename}")

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
def main():
    output_scripts_dir = "F:\\AutoCoder\\AutoCoder\\output_scripts"
    last_script_path = get_last_script(output_scripts_dir)

    if last_script_path:
        script_name = os.path.basename(last_script_path)
        with open(last_script_path, 'r') as file:
            script_content = file.read()
        
        original_prompt = read_file("F:\\AutoCoder\\AutoCoder\\original_prompt.txt")
        # Including the script name in the prompt
        usage_prompt = f"this is the name of the script as it appears in F:\AutoCoder\AutoCoder\output_scripts: '{script_name}'. Make sure when you call the script you include the full path. Output an example usage of the script and nothing else, for instance, a good example output to show response structure would be 'python {script_name} 1 10'. Do not output your own summary. If there are fields which need input, enter your own example based on the original_prompt. Output your command as plain text. Do not include ``` or similar. Here is the original prompt for formulating an example to test your script by command line: '{original_prompt}'"
        example_usage = query_gpt4_chat(f"{script_content}\n{usage_prompt}")

        # This should be the full command, e.g., "python output_1.py 1 10"
        example_usage_command = example_usage.strip()

        # Execute the example usage command
        output = run_script(example_usage_command)
        
        # Save the debug output with a name based on the script being debugged
        save_debug_output(script_name, output)

if __name__ == '__main__':
    main()
