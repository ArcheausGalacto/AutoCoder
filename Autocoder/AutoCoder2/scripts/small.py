import openai

def read_api_key(file_path="F:\\AutoCoder\\AutoCoder2\\api_key.txt"):
    with open(file_path, "r") as file:
        return file.read().strip()

def query_gpt35_turbo(prompt):
    api_key = read_api_key()
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message['content'].strip()

def main():
    # Example usage:
    test_prompt = "Translate 'Hello, world!' into French."
    print(query_gpt35_turbo(test_prompt))

if __name__ == "__main__":
    main()
