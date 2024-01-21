import openai
import os

def query_gpt4_chat(prompt, model="gpt-4-1106-preview"):
    with open("F:\\AutoCoder\\AutoCoder\\api_key.txt", "r") as file:
        api_key = file.read().strip()

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message['content'].strip()

def generate_skeleton(guide_prompt, user_prompt):
    final_prompt = guide_prompt + "\n" + user_prompt
    return query_gpt4_chat(final_prompt)

def get_next_version_number(directory, base_filename):
    version = 1
    while os.path.exists(os.path.join(directory, f"{base_filename}_{version}.py")):
        version += 1
    return version

def save_to_file(code, directory="F:\\AutoCoder\\AutoCoder\\output_scripts", base_filename="output"):
    version = get_next_version_number(directory, base_filename)
    filename = os.path.join(directory, f"{base_filename}_{version}.py")
    
    lines = code.split('\n')
    # Remove first and last line
    filtered_code = "\n".join(lines[1:-1])
    
    with open(filename, "w") as file:
        file.write(filtered_code)
    print(f"Saved as {filename}")

def run_autocoder(guide_prompt, user_prompt):
    code = generate_skeleton(guide_prompt, user_prompt)
    save_to_file(code)

def save_original_prompt(original_prompt, original_prompt_file="F:\\AutoCoder\\AutoCoder\\original_prompt.txt"):
    with open(original_prompt_file, "w") as file:
        file.write(original_prompt)

# This section can be commented out if you're calling run_autocoder from another script
# guide_prompt = "Your guide prompt here"
# user_prompt = "Your user prompt here"
# run_autocoder(guide_prompt, user_prompt)
