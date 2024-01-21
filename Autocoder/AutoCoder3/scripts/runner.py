import big
import usage
import validate
import os
import re

def get_latest_file_matching_pattern(directory, pattern):
    # List all files in the directory matching the pattern
    files = [os.path.join(directory, f) for f in os.listdir(directory) 
             if pattern in f and os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    return max(files, key=os.path.getctime)  # Return the latest file

def read_file(file_path):
    # Read and return the content of a file
    with open(file_path, "r") as file:
        return file.read().strip()

def sanitize_output(debug_output):
    """
    Sanitizes the debug output to remove any special characters 
    that could be misinterpreted as commands.
    """
    # Remove backticks and other patterns like 'bash' or 'shell' following backticks
    sanitized_output = re.sub(r'```[a-zA-Z]*', '', debug_output)
    return sanitized_output

guide_prompt = ("take the user prompt and output the complete script. For instance, your output should be all of the necessary functions and comments explaining their usage. DO NOT output anything but code. There should be no summary at the end!!! Here is an example of disallowed output, which was found at the end of a previous output. '```To run this script from the command line, you would save it to a python file, for example, `something.py`, and then execute it with...: ``` python something.py [input]' Do NOT give your analysis, just the code! Make it so that the script can be run from the command line. If there are dependencies which need to be downloaded, please include the necessary script commands to install them.")

def main():
    validation_result = 'no'
    iteration_count = 0
    original_prompt = read_file("F:\\AutoCoder\\AutoCoder2\\original_prompt.txt")

    while validation_result.lower() != 'yes':
        user_prompt = read_file("F:\\AutoCoder\\AutoCoder2\\prompt.txt")

        if iteration_count >= 6:
            recovery_prompt = input("Enter your feedback: ")
            user_prompt = "Here is the most recent suggestion: " + recovery_prompt + "\n" + user_prompt

        big.run_autocoder(guide_prompt, user_prompt)
        usage.main()  # Directly calling the main function of usage.py

        # Assume validate.main() can accept the original prompt for validation
        validation_result = validate.main()  # Modify this line as needed

        if validation_result and validation_result.lower() == 'no':
            output_scripts_dir = "F:\\AutoCoder\\AutoCoder2\\output_scripts"
            last_script_path = usage.get_last_script(output_scripts_dir)

            if last_script_path:
                script_name = os.path.basename(last_script_path)
                debug_output_dir = "F:\\AutoCoder\\AutoCoder2\\debug_output"
                debug_file_pattern = f"debug_{os.path.splitext(script_name)[0]}"
                latest_debug_output_file = get_latest_file_matching_pattern(debug_output_dir, debug_file_pattern)

                if latest_debug_output_file:
                    debug_output = read_file(latest_debug_output_file)
                    sanitized_debug_output = sanitize_output(debug_output)
                    last_script_content = read_file(last_script_path)

                    combined_prompt = (
                        "Modify the script based on the following debug output: " + sanitized_debug_output +
                        "\nHere is the latest script: \n" + last_script_content +
                        "\nHere is the original prompt: \n" + original_prompt
                    )
                    # Write the modified prompt back to the file
                    with open("F:\\AutoCoder\\AutoCoder2\\prompt.txt", "w") as file:
                        file.write(combined_prompt)

        iteration_count += 1

if __name__ == "__main__":
    main()
