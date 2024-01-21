import os
import big

def get_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    return max(files, key=os.path.getctime)

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def main():
    debug_output_dir = "F:\\AutoCoder\\AutoCoder\\debug_output"
    original_prompt_file = "F:\\AutoCoder\\AutoCoder\\original_prompt.txt"

    latest_debug_output_file = get_latest_file(debug_output_dir)
    original_prompt = read_file(original_prompt_file)

    if latest_debug_output_file:
        latest_debug_output = read_file(latest_debug_output_file)

        evaluation_prompt = (
            f"Based on the original prompt: '{original_prompt}', "
            f"does the following output satisfy the requirements? \n\n{latest_debug_output}\n\n"
            "Answer with 'yes' or 'no' only. IF THERE IS AN ERROR TRACEBACK, OUTPUT 'no'!!!!!!!!!!. As long as it satisfies the original prompt requirements EXACTLY, say yes."
        )

        evaluation = big.query_gpt4_chat(evaluation_prompt)
        print(evaluation)
        return evaluation.strip()

if __name__ == "__main__":
    main()
